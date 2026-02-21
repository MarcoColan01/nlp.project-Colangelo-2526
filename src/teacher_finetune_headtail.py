from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List, Union

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)

def bf16_supported() -> bool:
    """Check se la GPU supporta bf16 (Ampere o superiori)."""
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability()[0] >= 8:
        return True
    return False

@dataclass(frozen=True)
class TeacherModelConfig:
    model_name: str = "bert-base-uncased"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    gradient_checkpointing: bool = True # Cruciale per VRAM 8GB

def build_teacher_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)

def build_teacher_model(cfg: TeacherModelConfig):
    """Costruisce il modello per classificazione binaria (1 output logit)."""
    config = AutoConfig.from_pretrained(
        cfg.model_name,
        num_labels=2, 
        hidden_dropout_prob=cfg.hidden_dropout_prob,
        attention_probs_dropout_prob=cfg.attention_probs_dropout_prob,
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, config=config)
    
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        
    return model

def get_llrd_optimizer_parameters(model, learning_rate, weight_decay, layer_decay=0.95):
    """
    Layer-wise Learning Rate Decay (LLRD).
    Il LR decresce man mano che si scende verso i primi layer (embedding).
    """
    opt_parameters = []
    named_parameters = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    
    # 1. Head (Classifier + Pooler) - LR Pieno
    head_lr = learning_rate
    head_params = {"params": [], "weight_decay": weight_decay, "lr": head_lr}
    head_params_no_decay = {"params": [], "weight_decay": 0.0, "lr": head_lr}
    
    for n, p in named_parameters:
        if "classifier" in n or "pooler" in n:
            if any(nd in n for nd in no_decay):
                head_params_no_decay["params"].append(p)
            else:
                head_params["params"].append(p)
    
    opt_parameters.extend([head_params, head_params_no_decay])

    # 2. Encoder Layers - Decay progressivo
    # Bert ha 12 layer (0-11). Partiamo dall'11 (più alto) scendendo allo 0.
    if hasattr(model, "bert") and hasattr(model.bert, "encoder"):
        layers = model.bert.encoder.layer
        curr_lr = head_lr
        
        for i in range(len(layers) - 1, -1, -1):
            curr_lr *= layer_decay
            layer_params = {"params": [], "weight_decay": weight_decay, "lr": curr_lr}
            layer_params_no_decay = {"params": [], "weight_decay": 0.0, "lr": curr_lr}
            
            prefix = f"encoder.layer.{i}."
            for n, p in named_parameters:
                if prefix in n:
                    if any(nd in n for nd in no_decay):
                        layer_params_no_decay["params"].append(p)
                    else:
                        layer_params["params"].append(p)
            
            opt_parameters.extend([layer_params, layer_params_no_decay])

    # 3. Embeddings - LR Minimo
    curr_lr *= layer_decay
    emb_params = {"params": [], "weight_decay": weight_decay, "lr": curr_lr}
    emb_params_no_decay = {"params": [], "weight_decay": 0.0, "lr": curr_lr}
    
    for n, p in named_parameters:
        if "embeddings" in n:
             if any(nd in n for nd in no_decay):
                emb_params_no_decay["params"].append(p)
             else:
                emb_params["params"].append(p)
                
    opt_parameters.extend([emb_params, emb_params_no_decay])
    
    return opt_parameters

class WeightedCETrainer(Trainer):
    """
    Trainer custom per gestire la Weighted Binary Cross Entropy.
    """
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        if self.class_weights is not None:
             print(f"Trainer initialized with Weighted Loss. Class Weights: {self.class_weights.tolist()}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels").long()
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Flattening
        #logits = logits.view(-1)
        #labels = labels.view(-1).float()
        
        if self.class_weights is not None:
            w = self.class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=w)
        else:
            loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics(p):
        predictions, labels = p
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        logits = predictions  # [N,2]
        labels = labels.reshape(-1)

        # prob classe 1
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]

        preds = (probs >= 0.5).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
        acc = accuracy_score(labels, preds)

        roc_auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
        pr_auc = average_precision_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0

        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall, "roc_auc": roc_auc, "pr_auc": pr_auc}

def calculate_class_weights(dataset: Dataset, label_col: str = "label") -> torch.Tensor:
        labels = np.array(dataset[label_col])
        pos = labels.sum()
        neg = len(labels) - pos
        if pos == 0:
            return torch.tensor([1.0, 1.0], dtype=torch.float)
        w0 = 1.0
        w1 = float(neg / pos)
        return torch.tensor([w0, w1], dtype=torch.float)