
from typing import Any

from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import torch

from lexetta_lcp.CompLexPerAnnotator.schema import TrainingConfig


def create_base_model(
    model_name: str = "bert-base-uncased",
) -> Any:
    """
    Create the base model.

    Params:
        model_name: Name of the pretrained model

    Returns:
        A tuple of the base model and the tokenizer
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
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
    from CompLexPerAnnotator.train import compute_eval_metrics

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
        per_device_eval_batch_size=16,
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
    )

def load_trained(model_dir) -> tuple:
    """
    Load a trained model and its tokenizer from a directory.

    Params:
        model_dir: Path to the saved PEFT model directory

    Returns:
        A tuple of the model and the tokenizer
    """
    base_model, tokenizer = create_base_model()
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
    from CompLexPerAnnotator.data import encode

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
    from CompLexPerAnnotator.data import encode_batch
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
