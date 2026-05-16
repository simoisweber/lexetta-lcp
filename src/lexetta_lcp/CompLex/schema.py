from typing import Literal
from pydantic import BaseModel, Field
from enum import Enum

class TrainingConfig(BaseModel):
    rank: int # lora rank (higher rank -> higher parameter count)
    alpha: int # precision (no effect on parameter count)
    target_modules: list[str] # which parts of the llm we want to fine tune (query, key, value, ...)
    lora_dropout: float
    
    max_input_length: int # maximum number of tokens as input
    num_epochs: int
    learning_rate: float
    batch_size: int # batch size during training, higher values allow higher learning rates but also increase vram usage


class Metrics(BaseModel):
    train_time_s: float # time in seconds it took to train this model
    peak_vram_mb: float 

    params_trainable: int # number of trainable parameters
    params_total: int # number of parameters in total

    final_train_loss: float
    final_test_loss: float

    logs: list[dict] # logs from trainer.state.log_history

class TrainingRun(BaseModel):
    config: TrainingConfig
    metrics: Metrics
    version: Literal["1"] 