import argparse
import json
from pathlib import Path

from CompLexPerAnnotator.data import load_dataset, preprocess_data
from CompLexPerAnnotator.model import load_trained
from CompLexPerAnnotator.schema import TrainingConfig, RetrieverType
from CompLexPerAnnotator.train import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate a per-annotator model")
    parser.add_argument("path", help="Run directory of the trained per-annotator model")
    parser.add_argument("output_dir", help="Directory to save results (must not already exist)")
    # These must match what was used during training. load_dataset() defaults are seed=42, test_size=0.2.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--retriever",
        type=int,
        choices=[r.value for r in RetrieverType],
        default=None,
        help=f"Override the retriever type from the training config. {', '.join(f'{r.value}={r.name}' for r in RetrieverType)}",
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
    dataset = load_dataset(seed=args.seed, test_size=args.test_size)
    dataset = preprocess_data(dataset)

    print("Loading model...")
    model, tokenizer = load_trained(str(run_dir / "model"))

    retriever_type = RetrieverType(args.retriever) if args.retriever else None

    overall_r, per_annotator_r = evaluate_model(model, tokenizer, dataset, config, retriever_type=retriever_type)
    print(f"Overall Pearson r: {overall_r:.4f}")
    for aid, r in per_annotator_r.items():
        print(f"  {aid}: {r:.4f}")

    output_dir.mkdir(parents=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump({"overall_pearson_r": overall_r, "per_annotator_pearson_r": per_annotator_r}, f, indent=4)
    print(f"Results saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
