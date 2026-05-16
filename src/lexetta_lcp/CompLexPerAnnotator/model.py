
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from scipy import stats
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer, DataCollatorWithPadding

from lexetta_lcp.CompLexPerAnnotator.schema import TrainingConfig
from lexetta_lcp.CompLexPerAnnotator.data import encode_batch, encode


def compute_eval_metrics(preds, labels, annotator_ids=None) -> tuple[float, dict[str, float]]:
    """
    Compute overall Pearson r and per-annotator Pearson r values.

    Per-annotator r is computed only over annotators with >1 example and
    non-zero variance in both predictions and labels (others are skipped to
    keep the metric defined).

    Args:
        preds: Model predictions (1-D array, length N)
        labels: Ground-truth labels (1-D array, length N)
        annotator_ids: Annotator ID for each example (length N). If None,
            per-annotator r is returned as an empty dict.

    Returns:
        Tuple of (overall_pearson_r, {annotator_id: pearson_r})
    """
    overall_r, _ = stats.pearsonr(preds, labels)

    if annotator_ids is None:
        return float(overall_r), {}

    grouped: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for p, l, aid in zip(preds, labels, annotator_ids):
        grouped[aid][0].append(float(p))
        grouped[aid][1].append(float(l))

    per_annotator_r = {}
    for aid, (p, l) in grouped.items():
        if len(p) > 1 and np.std(p) > 0 and np.std(l) > 0:
            r, _ = stats.pearsonr(p, l)
            per_annotator_r[aid] = float(r)
    return float(overall_r), per_annotator_r

def create_base_model(
    model_name: str = "bert-base-uncased",
    max_input_length: int = 512,
) -> Any:
    """
    Create the base model.

    Works for both encoder (BERT-style) and decoder (Llama/Qwen/GPT-style)
    backbones. For decoders we configure pad token + left padding so that
    last-token pooling in AutoModelForSequenceClassification gives a usable
    regression representation.

    Params:
        model_name: Name of the pretrained model
        max_input_length: Sequence length cap stored on the tokenizer so that
            encode/encode_batch can read it without an extra argument.

    Returns:
        A tuple of the base model and the tokenizer
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.model_max_length = max_input_length

    if "token_type_ids" not in tokenizer.model_input_names:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer

def apply_lora(model: Any, config: TrainingConfig) -> Any:
    """
    Apply LoRA adapters to the model.

    Params:
        model: The base model
        config: Training configuration with LoRA parameters

    Returns:
        Model with LoRA adapters
    """
    lora_config = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        use_rslora=True, # was proven to work better for different ranks https://doi.org/10.48550/arXiv.2312.03732
    )
    
    model = get_peft_model(model, lora_config)
    return model

def create_trainer_per_annotator(
    model: Any,
    tokenizer: Any,
    config: TrainingConfig,
    train_dataset: Any,
    eval_dataset: Any,
    eval_annotator_ids: list = None,
    output_dir: str = None,
) -> Trainer:
    """
    Create a Trainer instance.

    Params:
        model: The model to train
        config: Training configuration
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        output_dir: Directory for outputs

    Returns:
        Configured Trainer instance
    """

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = preds.squeeze()  # (batch, 1) -> (batch,)
        overall_r, per_annotator_dict = compute_eval_metrics(preds, labels, annotator_ids=eval_annotator_ids)
        mean_per_annotator_r = (
            sum(per_annotator_dict.values()) / len(per_annotator_dict)
            if per_annotator_dict else float("nan")
        )
        return {
            "pearson_r": overall_r,
            "mean_per_annotator_pearson_r": mean_per_annotator_r,
            "per_annotator_pearson_r": per_annotator_dict,
        }

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=config.batch_size,
        bf16=True,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        logging_steps=100,
    )
    
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

def load_trained(model_dir) -> tuple:
    """
    Load a trained model and its tokenizer from a directory.

    Reads the base model name from the saved PEFT adapter config so this works
    for any backbone the adapter was trained on (encoder or decoder).

    Params:
        model_dir: Path to the saved PEFT model directory

    Returns:
        A tuple of the model and the tokenizer
    """
    from peft import PeftConfig

    peft_config = PeftConfig.from_pretrained(model_dir)
    base_model, tokenizer = create_base_model(model_name=peft_config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model=base_model, model_id=model_dir)
    return model, tokenizer


@torch.no_grad()
def predict(
    model: Any,
    tokenizer: Any,
    sentence: str,
    token: str,
    user_history: list[dict],
) -> float:
    """
    Predict lexical complexity for a single (sentence, token) pair given user history.

    Params:
        model: Trained model (e.g. from load_trained)
        tokenizer: Tokenizer matching the model
        sentence: The context sentence containing the target token
        token: The target token whose complexity is being predicted
        user_history: Already-retrieved list of history items to include in the
            prompt. Each item must have 'token' and 'complexity' (0..1) keys.
            The caller is responsible for any selection/retrieval logic.

    Returns:
        Predicted complexity in [0, 1]
    """
    model.eval()
    enc = encode(tokenizer, user_history, sentence, token)
    inputs = {k: torch.tensor(v).unsqueeze(0).to(model.device) for k, v in enc.items()}
    return model(**inputs).logits.squeeze().item()


@torch.no_grad()
def predict_batch(
    model: Any,
    tokenizer: Any,
    sentences: list[str],
    tokens: list[str],
    user_histories: list[list[dict]],
    batch_size: int = 16,
) -> list[float]:
    """
    Batched version of predict. All three lists must have the same length.

    Runs the forward pass in chunks of batch_size to bound GPU memory.

    Returns:
        List of predicted complexities in [0, 1], one per input.
    """
    from tqdm.auto import tqdm

    model.eval()
    out: list[float] = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="Predicting", unit="batch"):
        enc = encode_batch(
            tokenizer,
            user_histories[i : i + batch_size],
            sentences[i : i + batch_size],
            tokens[i : i + batch_size],
        )
        inputs = {k: torch.tensor(v).to(model.device) for k, v in enc.items()}
        logits = model(**inputs).logits.squeeze(-1)
        out.extend(logits.cpu().tolist())
    return out
