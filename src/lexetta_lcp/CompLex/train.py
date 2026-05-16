import torch
import time
from typing import Optional, Any, Tuple
from pathlib import Path
from transformers import Trainer
from datasets import DatasetDict

from lexetta_lcp.CompLex.schema import TrainingConfig, TrainingRun, Metrics
from lexetta_lcp.CompLex import tokenize_complex_dataset, create_trainer_complex, create_base_model, apply_lora


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

    # Tokenize
    print("Tokenizing dataset...")
    tokenized_dataset = tokenize_complex_dataset(dataset, tokenizer=tokenizer, max_length=config.max_input_length)
    
    # Train
    trainer = create_trainer_complex(
        model=model,
        config=config,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        output_dir=output_dir
    )

    print("Training...")    
    # evaluate once at the start for a baseline
    pre_train_eval = trainer.evaluate()
    # start the actual training
    train_time, peak_vram = train_model(trainer)
        
    # Extract metrics
    final_train_loss, final_eval_loss = extract_losses(trainer)

    # Create result object
    metrics = Metrics(
        train_time_s=train_time,
        peak_vram_mb=peak_vram,
        params_trainable=trainable,
        params_total=total,
        final_test_loss=final_eval_loss,
        final_train_loss=final_train_loss,
        logs=[pre_train_eval] + trainer.state.log_history
    )
    
    result = TrainingRun(
        config=config,
        metrics=metrics,
        version="1"
    )    
    return trainer, result
