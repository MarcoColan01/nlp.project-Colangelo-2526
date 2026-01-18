# src/embeddings/api.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Union

import numpy as np

from src.utils import get_device

from .bert_embeddings import (
    BertBundle,
    embed_texts_bert,
    load_bert_base_uncased,
)

ModelName = Literal["bert-base-uncased"]


@dataclass
class ModelBundle:
    """
    Thin wrapper so the rest of the code can handle multiple model families uniformly.
    """
    model_name: str
    backend: str
    bundle: object


def load_model(
    model_name: ModelName = "bert-base-uncased",
    *,
    device: Optional[str] = None,
) -> ModelBundle:
    device = device or get_device(prefer_cuda=True)

    if model_name == "bert-base-uncased":
        b = load_bert_base_uncased(device=device, model_name=model_name)
        return ModelBundle(model_name=model_name, backend="bert", bundle=b)

    raise ValueError(f"Unsupported model_name={model_name}")


def embed_text(
    model: ModelBundle,
    texts: Sequence[str],
    *,
    max_length: int = 64,
    batch_size: int = 16,
    pool: str = "cls",
    layer_mode: str = "last",
    layer_k: Optional[int] = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    Unified embedding API.
    For now supports only bert-base-uncased.
    """
    if model.backend == "bert":
        return embed_texts_bert(
            model.bundle,  # type: ignore[arg-type]
            texts,
            max_length=max_length,
            batch_size=batch_size,
            pool=pool,              # type: ignore[arg-type]
            layer_mode=layer_mode,  # type: ignore[arg-type]
            layer_k=layer_k,
            normalize=normalize,
        )
    raise ValueError(f"Unsupported backend={model.backend}")
