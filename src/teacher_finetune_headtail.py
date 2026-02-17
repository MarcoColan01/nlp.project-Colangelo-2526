from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)


def bf16_supported() -> bool:
    """Ritorna True se la GPU supporta bf16 in modo affidabile."""
    if not torch.cuda.is_available():
        return False
    if hasattr(torch.cuda, "is_bf16_supported"):
        try:
            return bool(torch.cuda.is_bf16_supported())
        except Exception:
            pass
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


@dataclass(frozen=True)
class TeacherModelConfig:
    # Per il tuo caso: bert-base-uncased
    model_name: str = "bert-base-uncased"
    hidden_dropout_prob: Optional[float] = None
    attention_probs_dropout_prob: Optional[float] = None
    gradient_checkpointing: bool = False
    # torch 2.x: 'sdpa' spesso è più veloce; usa 'eager' se incontri problemi
    attn_implementation: str = "sdpa"


def build_teacher_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def build_teacher_model(cfg: TeacherModelConfig):
    """BERT binario con *un solo logit* e loss BCEWithLogits.

    In HuggingFace, BCE è attivata impostando:
      - num_labels=1
      - problem_type='multi_label_classification'
    """
    config = AutoConfig.from_pretrained(
        cfg.model_name,
        num_labels=1,
        problem_type="multi_label_classification",
        attn_implementation=cfg.attn_implementation,
    )
    if cfg.hidden_dropout_prob is not None:
        config.hidden_dropout_prob = float(cfg.hidden_dropout_prob)
    if cfg.attention_probs_dropout_prob is not None:
        config.attention_probs_dropout_prob = float(cfg.attention_probs_dropout_prob)

    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, config=config)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


def build_collator(tokenizer):
    # padding dinamico -> meno memoria
    return DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")


def get_llrd_optimizer_parameters(model, learning_rate, weight_decay, layer_decay=0.95):
    """Crea gruppi parametri con Layer-wise Learning Rate Decay (LLRD)."""
    opt_parameters = []
    named_parameters = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    lr = learning_rate

    # 1) Classifier & pooler
    head_names = ["classifier", "pooler"]
    opt_parameters.append(
        {
            "params": [
                p
                for n, p in named_parameters
                if any(h in n for h in head_names) and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
            "lr": lr,
        }
    )
    opt_parameters.append(
        {
            "params": [
                p
                for n, p in named_parameters
                if any(h in n for h in head_names) and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": lr,
        }
    )

    # 2) Encoder layers (dall'ultimo al primo)
    if hasattr(model, "bert") and hasattr(model.bert, "encoder"):
        n_layers = len(model.bert.encoder.layer)
        for i in range(n_layers - 1, -1, -1):
            lr *= layer_decay
            layer_prefix = f"encoder.layer.{i}."
            opt_parameters.append(
                {
                    "params": [
                        p
                        for n, p in named_parameters
                        if layer_prefix in n and not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": weight_decay,
                    "lr": lr,
                }
            )
            opt_parameters.append(
                {
                    "params": [
                        p
                        for n, p in named_parameters
                        if layer_prefix in n and any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                    "lr": lr,
                }
            )

    # 3) Embeddings (LR minimo)
    lr *= layer_decay
    opt_parameters.append(
        {
            "params": [
                p
                for n, p in named_parameters
                if "embeddings" in n and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
            "lr": lr,
        }
    )
    opt_parameters.append(
        {
            "params": [
                p
                for n, p in named_parameters
                if "embeddings" in n and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": lr,
        }
    )
    return opt_parameters


def load_flat_dataset(parquet_path: str) -> Dataset:
    """Carica parquet *già tokenizzato* e normalizza labels a float (0/1)."""
    ds = load_dataset("parquet", data_files=str(parquet_path))["train"]
    if "label" in ds.column_names and "labels" not in ds.column_names:
        ds = ds.rename_column("label", "labels")
    ds = ds.map(lambda ex: {"labels": float(ex["labels"])})

    cols = ["input_ids", "attention_mask", "token_type_ids", "labels"]
    ds = ds.select_columns([c for c in cols if c in ds.column_names])
    return ds


def compute_pos_weight(train_ds: Dataset, label_col: str = "labels") -> float:
    """pos_weight = N_neg / N_pos per BCEWithLogitsLoss(pos_weight=...)."""
    labels = np.asarray(train_ds[label_col], dtype=np.float32)
    n_pos = float(labels.sum())
    n_tot = float(labels.shape[0])
    n_neg = n_tot - n_pos
    if n_pos <= 0:
        return 1.0
    return n_neg / n_pos


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def find_best_f1_threshold(
    probs: np.ndarray, labels: np.ndarray, steps: int = 201
) -> Tuple[float, float, float, float]:
    """Cerca la soglia che massimizza F1 sulla classe positiva.

    Ritorna: (best_threshold, best_f1, precision, recall)
    """
    probs = np.asarray(probs).reshape(-1)
    labels = np.asarray(labels).reshape(-1).astype(int)

    thresholds = np.linspace(0.0, 1.0, steps)
    best_t = 0.5
    best_f1 = -1.0
    best_pr = 0.0
    best_rc = 0.0

    for t in thresholds:
        preds = (probs >= t).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
            best_pr = float(pr)
            best_rc = float(rc)

    return best_t, best_f1, best_pr, best_rc


def compute_metrics(p) -> Dict[str, float]:
    """Metriche su validation durante training.

    Include:
      - f1@0.5 (logging)
      - best_f1 + best_threshold (ottimizzazione soglia su validation)
      - ROC-AUC e PR-AUC (threshold-free)
    """
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    logits = np.asarray(logits).reshape(-1)
    probs = _sigmoid(logits)
    labels = np.asarray(p.label_ids).reshape(-1)

    # threshold fisso 0.5
    preds_05 = (probs >= 0.5).astype(int)
    acc = accuracy_score(labels, preds_05)
    pr, rc, f1, _ = precision_recall_fscore_support(labels, preds_05, average="binary", zero_division=0)

    # threshold-free
    try:
        roc_auc = roc_auc_score(labels, probs)
    except Exception:
        roc_auc = 0.5
    try:
        pr_auc = average_precision_score(labels, probs)
    except Exception:
        pr_auc = 0.0

    # best threshold (validation)
    best_t, best_f1, best_pr, best_rc = find_best_f1_threshold(probs, labels)

    return {
        "accuracy_0.5": float(acc),
        "precision_0.5": float(pr),
        "recall_0.5": float(rc),
        "f1_0.5": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "best_threshold": float(best_t),
        "best_precision": float(best_pr),
        "best_recall": float(best_rc),
        "best_f1": float(best_f1),
    }


class WeightedBCETrainer(Trainer):
    """Trainer con BCEWithLogitsLoss e pos_weight per gestire sbilanciamento."""

    def __init__(self, *args, pos_weight: Optional[float] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        logits = logits.view(-1)
        labels = labels.view(-1).float()

        if self.pos_weight is not None:
            w = torch.tensor([float(self.pos_weight)], device=logits.device)
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=w)
        else:
            loss_fct = nn.BCEWithLogitsLoss()

        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def predict_probs(trainer: Trainer, ds: Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """Ritorna (probs, labels) usando sigmoid su logits."""
    out = trainer.predict(ds)
    logits = out.predictions[0] if isinstance(out.predictions, tuple) else out.predictions
    logits = np.asarray(logits).reshape(-1)
    probs = _sigmoid(logits)
    labels = np.asarray(out.label_ids).reshape(-1).astype(int)
    return probs, labels


def evaluate_with_threshold(trainer: Trainer, ds: Dataset, threshold: float) -> Dict[str, float]:
    """Valuta un dataset usando una soglia fissata (scelta su validation)."""
    probs, labels = predict_probs(trainer, ds)
    preds = (probs >= threshold).astype(int)

    acc = accuracy_score(labels, preds)
    pr, rc, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)

    try:
        roc_auc = roc_auc_score(labels, probs)
    except Exception:
        roc_auc = 0.5
    try:
        pr_auc = average_precision_score(labels, probs)
    except Exception:
        pr_auc = 0.0

    return {
        "accuracy": float(acc),
        "precision": float(pr),
        "recall": float(rc),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "threshold": float(threshold),
    }
