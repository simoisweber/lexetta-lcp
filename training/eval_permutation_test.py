"""
Permutation test: does the personalized model actually use annotator identity?

Procedure:
  1. Run inference once to get (pred, label, annotator_id, task_id) per test row.
  2. Compute the true mean per-annotator Pearson r.
  3. Repeatedly permute predictions among annotators *within each task*
     (so task structure is preserved, but annotator identity is destroyed).
     Recompute mean per-annotator r each time.
  4. If the true r is significantly higher than the shuffled distribution, the
     model is using annotator-specific information. If it's inside the shuffled
     distribution, the model's apparent personalization is just task structure.

This is the strongest available test for "is personalization real?" since it
keeps every other source of signal fixed.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from transformers import Trainer, TrainingArguments

from CompLexPerAnnotator.data import load_dataset, tokenize_per_annotator_dataset, get_user_histories
from CompLexPerAnnotator.model import load_trained
from CompLexPerAnnotator.schema import TrainingConfig, RetrieverType
from CompLexPerAnnotator.train import get_retriever


def mean_per_annotator_r(preds: np.ndarray, labels: np.ndarray, annotator_ids: np.ndarray) -> float:
    grouped: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for p, l, aid in zip(preds, labels, annotator_ids):
        grouped[aid][0].append(p)
        grouped[aid][1].append(l)
    rs = []
    for p, l in grouped.values():
        if len(p) > 1 and np.std(p) > 0 and np.std(l) > 0:
            r, _ = stats.pearsonr(p, l)
            rs.append(r)
    return float(np.mean(rs)) if rs else float("nan")


def run_inference(model, tokenizer, dataset, config, retriever_type) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    user_histories = get_user_histories(dataset)
    retriever_map = {aid: get_retriever(retriever_type, hist) for aid, hist in user_histories.items()}
    print(f"Built {retriever_type.name} retrievers for {len(retriever_map)} annotators")

    print("Tokenizing test set...")
    tokenized = tokenize_per_annotator_dataset(
        dataset, tokenizer=tokenizer,
        retriever_map=retriever_map,
        user_history_length=config.user_history_length,
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="/tmp/eval", per_device_eval_batch_size=16, report_to="none"),
    )

    print("Running inference...")
    output = trainer.predict(tokenized["test"])
    preds = output.predictions.squeeze()
    labels = output.label_ids
    annotator_ids = np.array(dataset["test"]["annotator_id"])
    task_ids = np.array(dataset["test"]["task_id"])
    return preds, labels, annotator_ids, task_ids


def permute_within_task(preds: np.ndarray, task_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle predictions among the annotators of each task.

    Task-level structure is preserved (the same multiset of predictions stays on
    each task), but the assignment of which prediction belongs to which annotator
    is randomized. This is the key permutation: it destroys exactly the
    annotator-specific signal we want to test for.
    """
    permuted = preds.copy()
    # group row indices by task
    task_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(task_ids):
        task_to_indices[t].append(i)
    for indices in task_to_indices.values():
        if len(indices) > 1:
            shuffled = rng.permutation(indices)
            permuted[indices] = preds[shuffled]
    return permuted


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="Run directory of the trained per-annotator model")
    parser.add_argument("--seed", type=int, default=42, help="Dataset split seed (must match training)")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--retriever", type=int, choices=[r.value for r in RetrieverType], default=None)
    parser.add_argument("--num-permutations", type=int, default=1000)
    parser.add_argument("--perm-seed", type=int, default=0, help="Seed for the permutation RNG")
    args = parser.parse_args()

    run_dir = Path(args.path)

    print("Loading config...")
    with open(run_dir / "config.json") as f:
        config = TrainingConfig.model_validate_json(f.read())

    print("Loading dataset...")
    dataset = load_dataset(seed=args.seed, test_size=args.test_size)

    print("Loading model...")
    model, tokenizer = load_trained(str(run_dir / "model"))
    if torch.cuda.is_available():
        model = model.cuda()

    retriever_type = RetrieverType(args.retriever) if args.retriever else config.retriever_type

    preds, labels, annotator_ids, task_ids = run_inference(model, tokenizer, dataset, config, retriever_type)

    true_r = mean_per_annotator_r(preds, labels, annotator_ids)
    print(f"\nTrue mean per-annotator Pearson r: {true_r:.4f}")

    rng = np.random.default_rng(args.perm_seed)
    print(f"Running {args.num_permutations} within-task permutations...")
    shuffled_rs = np.empty(args.num_permutations)
    for i in range(args.num_permutations):
        permuted = permute_within_task(preds, task_ids, rng)
        shuffled_rs[i] = mean_per_annotator_r(permuted, labels, annotator_ids)
        if (i + 1) % max(1, args.num_permutations // 10) == 0:
            print(f"  [{i + 1:>5}/{args.num_permutations}] running mean = {shuffled_rs[:i+1].mean():.4f}")

    mean_shuf = shuffled_rs.mean()
    std_shuf = shuffled_rs.std()
    p_value = (shuffled_rs >= true_r).sum() / len(shuffled_rs)
    z = (true_r - mean_shuf) / std_shuf if std_shuf > 0 else float("inf")

    print()
    print("=" * 60)
    print(f"True per-annotator r:        {true_r:.4f}")
    print(f"Shuffled mean ± std:         {mean_shuf:.4f} ± {std_shuf:.4f}")
    print(f"Shuffled min / max:          {shuffled_rs.min():.4f} / {shuffled_rs.max():.4f}")
    print(f"z-score of true vs shuffled: {z:.2f}")
    print(f"p-value (one-sided):         {p_value:.4f}  ({(shuffled_rs >= true_r).sum()}/{len(shuffled_rs)} shuffles >= true)")
    print("=" * 60)

    if true_r > shuffled_rs.max():
        print("Result: model uses annotator identity (true r exceeds every permutation).")
    elif p_value < 0.05:
        print("Result: model uses annotator identity (p < 0.05).")
    else:
        print("Result: NO evidence the model uses annotator identity beyond task structure.")


if __name__ == "__main__":
    main()
