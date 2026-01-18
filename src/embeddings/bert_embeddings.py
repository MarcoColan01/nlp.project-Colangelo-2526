# src/embeddings/bert_backend.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

PoolMode = Literal["cls", "mean", "last_token"]
LayerMode = Literal["last", "last4_avg", "layer_k"]


@dataclass
class BertBundle:
    name: str
    tokenizer: any
    model: any
    device: str


def load_bert_base_uncased(
    device: str = "cpu",
    model_name: str = "bert-base-uncased",
) -> BertBundle:
    """
    Loads bert-base-uncased tokenizer + model with hidden states enabled.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()
    model.to(device)
    return BertBundle(name=model_name, tokenizer=tokenizer, model=model, device=device)


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.maximum(norm, eps)
    return x / norm


def _select_layer_hidden_states(
    hidden_states: Tuple[torch.Tensor, ...],
    layer_mode: LayerMode,
    layer_k: Optional[int],
) -> torch.Tensor:
    """
    hidden_states is a tuple of tensors: (embeddings, layer1, ..., layerL)
    Each tensor has shape: [B, T, H]
    """
    if layer_mode == "last":
        return hidden_states[-1]
    if layer_mode == "last4_avg":
        # average over last 4 layers
        last4 = hidden_states[-4:]
        return torch.stack(last4, dim=0).mean(dim=0)  # [B, T, H]
    if layer_mode == "layer_k":
        if layer_k is None:
            raise ValueError("layer_k must be provided when layer_mode='layer_k'")
        # allow negative indexing like -1, -2...
        return hidden_states[layer_k]
    raise ValueError(f"Unknown layer_mode={layer_mode}")


def _pool_sequence(
    token_embeddings: torch.Tensor,  # [B, T, H]
    attention_mask: torch.Tensor,    # [B, T]
    pool: PoolMode,
) -> torch.Tensor:
    if pool == "cls":
        return token_embeddings[:, 0, :]  # [B, H]

    if pool == "last_token":
        # last *non-padding* token
        lengths = attention_mask.sum(dim=1)  # [B]
        idx = (lengths - 1).clamp(min=0)     # [B]
        b = torch.arange(token_embeddings.size(0), device=token_embeddings.device)
        return token_embeddings[b, idx, :]

    if pool == "mean":
        # mean pooling over non-padding tokens
        mask = attention_mask.unsqueeze(-1).type_as(token_embeddings)  # [B, T, 1]
        summed = (token_embeddings * mask).sum(dim=1)                   # [B, H]
        denom = mask.sum(dim=1).clamp(min=1e-6)                         # [B, 1]
        return summed / denom

    raise ValueError(f"Unknown pool={pool}")


@torch.no_grad()
def embed_texts_bert(
    bundle: BertBundle,
    texts: Sequence[str],
    *,
    max_length: int = 64,
    batch_size: int = 16,
    pool: PoolMode = "cls",
    layer_mode: LayerMode = "last",
    layer_k: Optional[int] = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    Returns embeddings with shape [N, H] as float32 numpy array.

    Notes:
      - Uses hidden states, selectable by layer_mode.
      - Uses pooling strategy for sequence -> vector.
      - Recommended defaults: pool='cls' or 'mean'; layer_mode='last4_avg' is often more stable.
    """
    all_vecs: list[np.ndarray] = []
    device = bundle.device

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = bundle.tokenizer(
            list(batch),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        out = bundle.model(**enc)
        hidden_states = out.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden_states. Check output_hidden_states=True.")

        token_emb = _select_layer_hidden_states(hidden_states, layer_mode, layer_k)  # [B,T,H]
        pooled = _pool_sequence(token_emb, enc["attention_mask"], pool)              # [B,H]

        vec = pooled.detach().cpu().to(torch.float32).numpy()
        all_vecs.append(vec)

    X = np.concatenate(all_vecs, axis=0).astype(np.float32)
    if normalize:
        X = _l2_normalize(X)
    return X
