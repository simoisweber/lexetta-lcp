import torch
import time
from typing import Any
from pathlib import Path
from scipy import stats
from transformers import Trainer, TrainingArguments
from datasets import DatasetDict

from CompLexPerAnnotator.schema import TrainingConfig, TrainingRun, Metrics, RetrieverType
from CompLexPerAnnotator.data import tokenize_per_annotator_dataset, get_user_histories
from CompLexPerAnnotator.model import create_trainer_per_annotator, create_base_model, apply_lora
from CompLexPerAnnotator.retriever import RandomRetriever, WordFrequencyRetriever, Retriever, CorpusRetriever

def get_trainable_params(model: Any) -> tuple[int, int]:
    """
    Get trainable parameter statistics.
    
    Args:
        model: The model to analyze
        
    Returns:
        Tuple of (trainable params, total params)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total



def extract_losses(trainer: Trainer) -> tuple[float, float]:
    """
    Extract final training and evaluation losses from trainer logs.
    
    Args:
        trainer: The Trainer instance after training
        
    Returns:
        Tuple of (final train loss, final eval loss)
    """
    logs = trainer.state.log_history
    final_train_loss = next(l["loss"] for l in reversed(logs) if "loss" in l)
    final_eval_loss = next(l["eval_loss"] for l in reversed(logs) if "eval_loss" in l)
    return final_train_loss, final_eval_loss


def save_results(
    data: TrainingRun,
    filepath: Path | str
) -> None:
    """
    Save training results to a JSON file.
    
    Args:
        data: The TrainingRun data to save
        filepath: Path to save the results
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, "w") as f:
        f.write(data.model_dump_json(indent=4))
    
    print(f"Results saved to {filepath}")

def train_model(trainer: Trainer) -> tuple[float, float]:
    """
    Train the model and collect timing/memory metrics.
    
    Args:
        trainer: The Trainer instance
        
    Returns:
        Tuple of (training time in seconds, peak VRAM in MB)
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    train_start = time.time()
    trainer.train()
    train_end = time.time()
    
    train_time = train_end - train_start
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
    
    return train_time, peak_vram_mb


def run_single_training(
    config: TrainingConfig,
    dataset: DatasetDict,
    output_dir: str = None,
) -> tuple[Trainer, TrainingRun]:
    """
    Run a complete fine-tuning pipeline with a single configuration.
    
    Args:
        config: Training configuration
        dataset: Pre-loaded dataset
        output_dir: Directory for training outputs
    Returns:
        TrainingRun with all metrics
    """
    print(f"Starting training with config: {config}")

    # Create model
    print("Creating model with LoRA adapters...")
    model, tokenizer = create_base_model()
    model = apply_lora(model, config)
    trainable, total = get_trainable_params(model)
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # Create Retriever for each annotator
    retriever_map = {}
    user_histories = get_user_histories(dataset)
    for aid, history in user_histories.items():
        retriever_map[aid] = get_retriever(retriever_type=config.retriever_type, history=history)
    print(f"Built {config.retriever_type.name} retrievers for {len(retriever_map)} annotators")

    # Tokenize
    print("Tokenizing dataset...")
    tokenized_dataset = tokenize_per_annotator_dataset(dataset, tokenizer=tokenizer, retriever_map=retriever_map, user_history_length=config.user_history_length)

    # Train
    trainer = create_trainer_per_annotator(
        model=model,
        config=config,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        output_dir=output_dir,
    )

    print("Training...")
    train_time, peak_vram = train_model(trainer)
    print(f"Training done in {train_time:.1f}s, peak VRAM: {peak_vram:.0f} MB")

    # Extract final metrics from the last epoch's eval (no extra forward pass needed)
    final_train_loss, final_eval_loss = extract_losses(trainer)
    logs = trainer.state.log_history
    pearson_r = next(l["eval_pearson_r"] for l in reversed(logs) if "eval_pearson_r" in l)
    print(f"Final train loss: {final_train_loss:.4f}, final eval loss: {final_eval_loss:.4f}")
    print(f"Pearson r: {pearson_r:.4f}")

    # Create result object
    metrics = Metrics(
        train_time_s=train_time,
        peak_vram_mb=peak_vram,
        params_trainable=trainable,
        params_total=total,
        final_test_loss=final_eval_loss,
        final_train_loss=final_train_loss,
        logs=logs,
    )
    
    result = TrainingRun(
        config=config,
        metrics=metrics,
        version="1"
    )    
    return trainer, result

def get_retriever(retriever_type: RetrieverType, history: list) -> Retriever:
    match retriever_type:
        case RetrieverType.RANDOM:
            return RandomRetriever(history=history)
        case RetrieverType.WORD_FREQUENCY:
            return WordFrequencyRetriever(history=history)
        case RetrieverType.CORPUS:
            return CorpusRetriever(history=history)
        case _:
            raise NotImplementedError(f"No Retriever implemented for {retriever_type}")


def compute_eval_metrics(preds, labels, annotator_ids=None) -> tuple[float, float]:
    """
    Compute overall Pearson r and mean per-annotator Pearson r.

    Per-annotator r is computed only over annotators with >1 example and
    non-zero variance in both predictions and labels (others are skipped to
    keep the metric defined).

    Args:
        preds: Model predictions (1-D array, length N)
        labels: Ground-truth labels (1-D array, length N)
        annotator_ids: Annotator ID for each example (length N). If None,
            mean per-annotator r is returned as NaN.

    Returns:
        Tuple of (overall_pearson_r, mean_per_annotator_pearson_r)
    """
    import numpy as np
    from collections import defaultdict

    overall_r, _ = stats.pearsonr(preds, labels)

    if annotator_ids is None:
        return float(overall_r), float("nan")

    grouped: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for p, l, aid in zip(preds, labels, annotator_ids):
        grouped[aid][0].append(float(p))
        grouped[aid][1].append(float(l))

    per_annotator_r = []
    for p, l in grouped.values():
        if len(p) > 1 and np.std(p) > 0 and np.std(l) > 0:
            r, _ = stats.pearsonr(p, l)
            per_annotator_r.append(r)
    mean_r = float(np.mean(per_annotator_r)) if per_annotator_r else float("nan")
    return float(overall_r), mean_r


def evaluate_model(
    model,
    tokenizer,
    dataset: DatasetDict,
    config,
    retriever_type: RetrieverType | None = None,
) -> tuple[float, float]:
    """
    Full evaluation pipeline for a loaded model.

    Builds retrievers, tokenizes the dataset, runs inference, and returns both
    Pearson r metrics. Use this when you need to evaluate a model from scratch
    (e.g. in the standalone eval script).

    Note: because RandomRetriever / CorpusRetriever are non-deterministic, results
    may differ slightly from training-time eval which used a fixed tokenized dataset.
    For exact reproducibility during training use compute_eval_metrics directly on
    the already-tokenized test split.

    Args:
        model: Trained model
        tokenizer: Tokenizer matching the model
        dataset: Raw (untokenized) DatasetDict with 'train' and 'test' splits
        config: TrainingConfig used during training
        retriever_type: Override the retriever type from config (optional)

    Returns:
        Tuple of (overall_pearson_r, mean_per_annotator_pearson_r)
    """
    retriever_type = retriever_type or config.retriever_type
    user_histories = get_user_histories(dataset)
    retriever_map = {
        aid: get_retriever(retriever_type, history)
        for aid, history in user_histories.items()
    }
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
    annotator_ids = dataset["test"]["annotator_id"]
    return compute_eval_metrics(preds, output.label_ids, annotator_ids)

