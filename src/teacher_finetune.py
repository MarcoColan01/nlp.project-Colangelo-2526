from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    DataCollatorWithPadding,
    EvalPrediction
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

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
    config = AutoConfig.from_pretrained(
        cfg.model_name,
        num_labels=1,
        problem_type="multi_label_classification",
    )    
    if cfg.hidden_dropout_prob is not None:
        config.hidden_dropout_prob = float(cfg.hidden_dropout_prob)
    if cfg.attention_probs_dropout_prob is not None:
        config.attention_probs_dropout_prob = float(cfg.attention_probs_dropout_prob)

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name, 
            config=config, 
            attn_implementation=cfg.attn_implementation
        )
    except TypeError:
        # Fallback per versioni transformers vecchie
        model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, config=config)        

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    return model

def freeze_bert_layers(model: nn.Module, freeze_embeddings: bool = True, freeze_layers: int = 0) -> None:
    """
    Congela gli embeddings e i primi N layer dell'encoder di BERT.
    BERT-Large ha 24 layer. Un freeze_layers=20 lascia allenabili solo gli ultimi 4.
    """
    # 1. Congela Embeddings
    if freeze_embeddings and hasattr(model, "bert") and hasattr(model.bert, "embeddings"):
        for param in model.bert.embeddings.parameters():
            param.requires_grad = False
        print("✅ Embeddings FROZEN.")

    # 2. Congela Encoder Layers
    if freeze_layers > 0 and hasattr(model, "bert") and hasattr(model.bert, "encoder"):
        encoder_layers = model.bert.encoder.layer
        total_layers = len(encoder_layers)
        
        if freeze_layers > total_layers:
            freeze_layers = total_layers
            
        for i in range(freeze_layers):
            for param in encoder_layers[i].parameters():
                param.requires_grad = False
        
        print(f"✅ First {freeze_layers}/{total_layers} encoder layers FROZEN.")

    # Statistiche finali
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Params Status: {trainable_params/1e6:.1f}M Trainable / {total_params/1e6:.1f}M Total ({(trainable_params/total_params)*100:.1f}%)")
     
def build_collator(tokenizer):
    return DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")

def compute_metrics(p: EvalPrediction):
    logits = p.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    
    # Sigmoid per ottenere probabilità (poiché usiamo BCEWithLogitsLoss)
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int).reshape(-1)
    labels = p.label_ids.reshape(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)
    
    try:
        auc = roc_auc_score(labels, probs)
    except:
        auc = 0.5

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc": auc
    }

def _load_chunk_counts(chunk_counts_parquet: Path) -> Dict[int, int]:
    df = pd.read_parquet(chunk_counts_parquet)
    return {int(rid): int(nc) for rid, nc in zip(df["review_id"].tolist(), df["n_chunks"].tolist())}

def load_chunks_dataset(
        chunks_parquet: Path,
        chunk_counts_parquet: Optional[Path] = None,
        add_sample_weight: bool = False,
):
    ds = load_dataset("parquet", data_files=str(chunks_parquet))["train"]
    # Rimuoviamo colonne non necessarie al modello, ma teniamo labels
    drop_cols = [c for c in ds.column_names if c in ["movie_id", "review_id", "chunk_index"]]
    
    # Se dobbiamo calcolare i pesi, ci servono gli ID temporaneamente
    if not add_sample_weight and drop_cols:
        ds = ds.remove_columns(drop_cols)
    
    # Rinomina label -> labels per compatibilità HF Trainer
    if "label" in ds.column_names and "labels" not in ds.column_names:
        ds = ds.rename_column("label", "labels")

    # Cast a float per BCE loss
    ds = ds.map(lambda ex: {"labels": float(ex["labels"])}, desc="Cast labels")

    if add_sample_weight:
        if chunk_counts_parquet is None:
            raise ValueError("No chunk_counts_parquet provided but add_sample_weight=True")
        
        # Carichiamo mappa dei conteggi in memoria (è piccola, ~100k int)
        counts = _load_chunk_counts(chunk_counts_parquet)

        def _add_w(ex):
            rid = int(ex["review_id"])
            nc = counts.get(rid, 1)
            # Peso inverso: se ho 10 chunk, il peso è 0.1.
            # Se ho 1 chunk, il peso è 1.0.
            weight = 1.0 / max(1, float(nc))
            return {"sample_weight": weight}

        ds = ds.map(_add_w, desc="Add sample_weight")
        
        # Ora possiamo rimuovere le colonne ID, tenendo sample_weight
        cols_to_keep = {"input_ids", "attention_mask", "token_type_ids", "labels", "sample_weight"}
        remove = [c for c in ds.column_names if c not in cols_to_keep]
        if remove:
            ds = ds.remove_columns(remove)

    return ds

class WeightedBCETrainer(Trainer):
    def compute_loss(self, model, inputs: Dict[str, Any], return_outputs: bool = False, **kwargs):
        labels = inputs.pop("labels")
        sample_weight = inputs.pop("sample_weight", None)

        # Filtra input per passare solo ciò che BERT si aspetta
        model_inputs = {k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
        outputs = model(**model_inputs)
        
        logits = outputs.logits
        
        # Squeeze se necessario [Batch, 1] -> [Batch]
        if logits.dim() == 2 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        labels = labels.float().squeeze() if labels.dim() == 2 else labels.float()

        # Reduction none per poter applicare i pesi per-sample
        loss_fct = nn.BCEWithLogitsLoss(reduction="none")
        loss_vec = loss_fct(logits, labels)

        if sample_weight is not None:
            sample_weight = sample_weight.float().squeeze() if sample_weight.dim() == 2 else sample_weight.float()
            loss_vec = loss_vec * sample_weight

        loss = loss_vec.mean()
        return (loss, outputs) if return_outputs else loss