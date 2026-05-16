from typing import Literal
from pydantic import BaseModel, Field
from enum import IntEnum

class RetrieverType(IntEnum):
    RANDOM = 1
    WORD_FREQUENCY = 2
    CORPUS = 3

class TrainingConfig(BaseModel):
    model_name: str = "bert-base-uncased" # HF model id; switches the pipeline between encoder and decoder backbones
    max_input_length: int = 512 # tokenizer max length; raise for decoders with longer context windows

    rank: int # lora rank (higher rank -> higher parameter count)
    alpha: int # precision (no effect on parameter count)
    target_modules: list[str] # which parts of the llm we want to fine tune (query, key, value, ...)
    lora_dropout: float

    retriever_type: RetrieverType
    user_history_length: int # maximum number of tokens as input
    val_split: float = 0.15 # fraction of users to use for validation
    test_split: float = 0.15 # fraction of users to use for testing
    seed: int = 42 # random seed for reproducible train/val/test splits
    num_epochs: int
    learning_rate: float
    warmup_ratio: float = 0.0 # fraction of total training steps used for linear LR warmup; 0 disables warmup
    batch_size: int # batch size during training, higher values allow higher learning rates but also increase vram usage


class Metrics(BaseModel):
    train_time_s: float # time in seconds it took to train this model
    peak_vram_mb: float

    params_trainable: int # number of trainable parameters
    params_total: int # number of parameters in total

    final_train_loss: float
    final_eval_loss: float # loss on the validation split

    logs: list[dict] # logs from trainer.state.log_history

class TrainingRun(BaseModel):
    config: TrainingConfig
    metrics: Metrics
    version: Literal["2"]