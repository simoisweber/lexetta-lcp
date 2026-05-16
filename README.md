# lexetta-lcp

LCP (Lexical Complexity Prediction) library for the Lexetta Project.

## Installation

Clone the repository and install the package:

```bash
git clone <repo-url> lexetta-lcp
cd lexetta-lcp
pip install -e .
```

## Commands

Installing the package exposes the following console scripts:

| Command | Description |
| --- | --- |
| `train-complex <config.json> <run_name>` | Train a model on the CompLex dataset. Writes config, checkpoints, results, and the final model to `outputs/<run_name>/`. |
| `train-per-annotator <config.json> <run_name>` | Train a per-annotator personalized model on the CompLex (per-annotator) dataset. Writes outputs to `outputs/<run_name>/`. |
| `tune-per-annotator` | Run a Ray Tune hyperparameter sweep for the per-annotator model. Trial outputs land under `outputs/tune/per_annotator_tune/`. |
| `eval-per-annotator <run_dir> <output_dir> [--split test\|validation]` | Evaluate a trained per-annotator model on the test (default) or validation split and save overall + per-annotator Pearson r to `<output_dir>/results.json`. |
| `eval-permutation-test <run_dir> [--num-permutations N] [--perm-seed S]` | Permutation test that checks whether a per-annotator model actually uses annotator identity by shuffling predictions within each task and comparing the true mean per-annotator Pearson r to the shuffled distribution. |
