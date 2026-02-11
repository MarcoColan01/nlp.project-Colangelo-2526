from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoConfig, AutoModelForSequenceClassification, AutoTokenizer,
    Trainer, DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
def bf16_supported() -> bool:
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
    model_name: str = "bert-large-uncased"
    hidden_dropout_prob: Optional[float] = None
    attention_probs_dropout_prob: Optional[float] = None
    gradient_checkpointing: bool = False
    attn_implementation: str = "sdpa" # Usa 'eager' se sdpa dà errori su vecchie torch

def build_teacher_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)
def build_teacher_model(cfg: TeacherModelConfig):
    config = AutoConfig.from_pretrained(cfg.model_name, num_labels=1, problem_type="multi_label_classification")
    if cfg.hidden_dropout_prob: config.hidden_dropout_prob = float(cfg.hidden_dropout_prob)
    if cfg.attention_probs_dropout_prob: config.attention_probs_dropout_prob = float(cfg.attention_probs_dropout_prob)
    
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, config=config)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model

def build_collator(tokenizer):
    return DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")

def get_llrd_optimizer_parameters(model, learning_rate, weight_decay, layer_decay=0.95):
    """
    Crea gruppi di parametri con Learning Rate decrescente.
    Top Layers (Classifier) -> LR Pieno
    Bottom Layers (Embeddings) -> LR Molto basso
    
    FIX: Filtra per nome (stringa) invece che per identità del tensore per evitare RuntimeError.
    """
    opt_parameters = []
    named_parameters = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    
    # Init LR
    lr = learning_rate
    
    # === 1. Classifier & Pooler (LR MASSIMO) ===
    # Filtriamo direttamente usando i nomi
    head_names = ["classifier", "pooler"]
    
    # Gruppo con Weight Decay
    opt_parameters.append({
        "params": [
            p for n, p in named_parameters 
            if any(h in n for h in head_names) 
            and not any(nd in n for nd in no_decay)
        ],
        "weight_decay": weight_decay,
        "lr": lr
    })
    
    # Gruppo senza Weight Decay (Bias, LayerNorm)
    opt_parameters.append({
        "params": [
            p for n, p in named_parameters 
            if any(h in n for h in head_names) 
            and any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.0,
        "lr": lr
    })
    
    # === 2. Encoder Layers (Decay all'indietro da 23 a 0) ===
    if hasattr(model, "bert") and hasattr(model.bert, "encoder"):
        n_layers = len(model.bert.encoder.layer)
        for i in range(n_layers - 1, -1, -1):
            lr *= layer_decay # Riduciamo LR per il prossimo strato
            layer_prefix = f"encoder.layer.{i}."
            
            # Parametri di QUESTO specifico layer
            opt_parameters.append({
                "params": [
                    p for n, p in named_parameters 
                    if layer_prefix in n 
                    and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
                "lr": lr
            })
            opt_parameters.append({
                "params": [
                    p for n, p in named_parameters 
                    if layer_prefix in n 
                    and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": lr
            })

    # === 3. Embeddings (LR MINIMO) ===
    lr *= layer_decay
    
    opt_parameters.append({
        "params": [
            p for n, p in named_parameters 
            if "embeddings" in n 
            and not any(nd in n for nd in no_decay)
        ],
        "weight_decay": weight_decay,
        "lr": lr
    })
    opt_parameters.append({
        "params": [
            p for n, p in named_parameters 
            if "embeddings" in n 
            and any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.0,
        "lr": lr
    })
    
    return opt_parameters

def load_flat_dataset(parquet_path: str):
    ds = load_dataset("parquet", data_files=str(parquet_path))["train"]
    if "label" in ds.column_names and "labels" not in ds.column_names:
        ds = ds.rename_column("label", "labels")
    ds = ds.map(lambda ex: {"labels": float(ex["labels"])})
    
    cols = ["input_ids", "attention_mask", "token_type_ids", "labels"]
    ds = ds.select_columns([c for c in cols if c in ds.column_names])
    return ds

def compute_metrics(p):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int).reshape(-1)
    labels = p.label_ids.reshape(-1)
    
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    try: auc = roc_auc_score(labels, probs)
    except: auc = 0.5
        
    return {"accuracy": acc, "f1": f1, "precision": p, "recall": r, "auc": auc}