import argparse
import json
from pathlib import Path

from datasets import DatasetDict
import torch

from lexetta_lcp.CompLexPerAnnotator.data import load_dataset, get_user_histories
from lexetta_lcp.CompLexPerAnnotator.model import load_trained, predict_batch
from lexetta_lcp.CompLexPerAnnotator.schema import TrainingConfig, RetrieverType
from lexetta_lcp.CompLexPerAnnotator.train import get_retriever, compute_eval_metrics


def evaluate_model(
    model,
    tokenizer,
    dataset: DatasetDict,
    config,
    retriever_type: RetrieverType | None = None,
    split: str = "test",
) -> tuple[float, dict[str, float]]:
    """
    Full evaluation pipeline for a loaded model.

    Note: because RandomRetriever / CorpusRetriever are non-deterministic, results
    may differ slightly from training-time eval which used a fixed tokenized dataset.
    For exact reproducibility during training use compute_eval_metrics directly on
    the already-tokenized split.

    Args:
        model: Trained model
        tokenizer: Tokenizer matching the model
        dataset: Raw (untokenized) DatasetDict with 'train', 'validation', and 'test' splits
        config: TrainingConfig used during training
        retriever_type: Override the retriever type from config (optional)
        split: Dataset split to evaluate on ('test' or 'validation')

    Returns:
        Tuple of (overall_pearson_r, per_annotator_pearson_r_list)
    """
    retriever_type = retriever_type or config.retriever_type
    user_histories = get_user_histories(dataset)
    retriever_map = {
        aid: get_retriever(retriever_type, history)
        for aid, history in user_histories.items()
    }
    print(f"Built {retriever_type.name} retrievers for {len(retriever_map)} annotators")

    split_ds = dataset[split]
    sentences = split_ds["sentence"]
    tokens = split_ds["token"]
    annotator_ids = split_ds["annotator_id"]
    labels = split_ds["complexity"]

    print(f"Retrieving histories for {len(split_ds)} examples...")
    histories = [
        retriever_map[row["annotator_id"]](sample=row, n=config.user_history_length)
        for row in split_ds
    ]

    print("Running inference...")
    preds = predict_batch(model, tokenizer, sentences, tokens, histories)
    return compute_eval_metrics(preds, labels, annotator_ids)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a per-annotator model")
    parser.add_argument("path", help="Run directory of the trained per-annotator model")
    parser.add_argument("output_dir", help="Directory to save results (must not already exist)")
    parser.add_argument(
        "--split",
        choices=["test", "validation"],
        default="test",
        help="Dataset split to evaluate on (default: test)",
    )
    args = parser.parse_args()

    run_dir = Path(args.path)
    output_dir = Path(args.output_dir)

    if output_dir.exists():
        parser.error(f"Output directory already exists: {output_dir}")

    print("Loading config...")
    with open(run_dir / "config.json") as f:
        config = TrainingConfig.model_validate_json(f.read())

    print("Loading dataset...")
    dataset = load_dataset(seed=config.seed, val_size=config.val_split, test_size=config.test_split)

    print("Loading model...")
    model, tokenizer = load_trained(str(run_dir / "model"))
    if torch.cuda.is_available():
        model = model.to("cuda")
    else:
        print("Warning: no GPU found!")

    overall_r, per_annotator_r = evaluate_model(
        model, tokenizer, dataset, config,
        retriever_type=config.retriever_type,
        split=args.split,
    )
    print(f"Overall Pearson r: {overall_r:.4f}")
    for aid, r in per_annotator_r.items():
        print(f"  {aid}: {r:.4f}")

    output_dir.mkdir(parents=True)
    results = {
        "config": config.model_dump(),
        "overall_pearson_r": overall_r,
        "per_annotator_pearson_r": per_annotator_r,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
