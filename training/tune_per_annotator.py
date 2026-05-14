import sys
from pathlib import Path

from ray import tune

from CompLexPerAnnotator import (
    TrainingConfig,
    load_dataset,
    run_single_training,
)
from CompLexPerAnnotator.schema import RetrieverType

PROJECT_ROOT = Path(__file__).resolve().parent

def objective(config):
    # Ray passes a flat dict; rebuild the typed TrainingConfig from it.
    training_config = TrainingConfig(
        rank=config["rank"],
        alpha=config["alpha"],
        target_modules=config["target_modules"],
        lora_dropout=config["lora_dropout"],
        retriever_type=RetrieverType(config["retriever_type"]),
        user_history_length=config["user_history_length"],
        num_epochs=config["num_epochs"],
        learning_rate=config["learning_rate"],
        batch_size=config["batch_size"],
    )

    data = load_dataset(
        cache_dir=str(PROJECT_ROOT / "data" / "per_annotator"),
        test_size=training_config.test_split,
        seed=training_config.seed,
    )

    # Ray gives each trial its own working dir; checkpoints land there.
    trial_dir = Path(tune.get_context().get_trial_dir())
    _, run = run_single_training(
        config=training_config,
        dataset=data,
        output_dir=str(trial_dir / "checkpoints"),
    )

    last_eval = next(l for l in reversed(run.metrics.logs) if "eval_pearson_r" in l)

    return {
        "pearson_r": last_eval["eval_pearson_r"],
        "mean_per_annotator_pearson_r": last_eval.get(
            "eval_mean_per_annotator_pearson_r", float("nan")
        ),
        "final_train_loss": run.metrics.final_train_loss,
        "final_test_loss": run.metrics.final_test_loss,
        "train_time_s": run.metrics.train_time_s,
        "peak_vram_mb": run.metrics.peak_vram_mb,
    }


search_space = {
    # Tuned axes
    "learning_rate": tune.grid_search([1e-4, 2e-4, 5e-4]),
    "rank": tune.grid_search([8, 16, 32]),

    # Held fixed -- change to grid_search to expand the sweep
    "alpha": 16,
    "lora_dropout": 0.1,
    "target_modules": ["query", "key", "value"],
    "retriever_type": int(RetrieverType.WORD_FREQUENCY),
    "user_history_length": 20,
    "num_epochs": 1,
    "batch_size": 8,
}

METRIC = "mean_per_annotator_pearson_r"
MODE = "max"


if __name__ == "__main__":
    tuner = tune.Tuner(
        # 1 GPU per trial -- each trial trains a full model. Set to fractional
        # values (e.g. 0.5) only if multiple trials can share a GPU.
        tune.with_resources(objective, resources={"gpu": 1, "cpu": 4}),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric=METRIC,
            mode=MODE,
            num_samples=1,
        ),
        run_config=tune.RunConfig(
            name="per_annotator_tune",
            storage_path=str(PROJECT_ROOT / "outputs" / "tune"),
        ),
    )

    results = tuner.fit()

    best = results.get_best_result(metric=METRIC, mode=MODE)
    print("\n=== Best trial ===")
    print(f"config:  {best.config}")
    print(f"metrics: {best.metrics}")
    print(f"path:    {best.path}")
