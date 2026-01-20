from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from utils import (
    project_paths,
    ensure_dir,
    save_json,
    load_json,
    save_npy,
    load_npy,
    short_hash,
)
from embeddings.api import ModelBundle, embed_text

AggMode = Literal["mean", "median"]


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n


@dataclass(frozen=True)
class ConceptEmbeddingConfig:
    model_name: str
    pool: str
    layer_mode: str
    layer_k: Optional[int]
    max_length: int
    batch_size: int
    agg: AggMode
    normalize_each: bool
    normalize_final: bool
    set_id: str = "default"

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "pool": self.pool,
            "layer_mode": self.layer_mode,
            "layer_k": self.layer_k,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "agg": self.agg,
            "normalize_each": self.normalize_each,
            "normalize_final": self.normalize_final,
            "set_id": self.set_id,
        }


def _cache_paths(cfg: ConceptEmbeddingConfig) -> Tuple[Path, Path]:
    paths = project_paths(create=True)
    base = paths.results / "cache" / "concept_embeddings"
    ensure_dir(base)

    cfg_hash = short_hash(cfg.to_dict(), length=16)
    meta_path = base / f"{cfg_hash}.meta.json"
    emb_path = base / f"{cfg_hash}.embeddings.npy"
    return meta_path, emb_path


def _templates_fingerprint(concept_to_texts: Dict[str, Sequence[str]]) -> str:
    # stable ordering of concepts + templates
    items = [(c, list(concept_to_texts[c])) for c in sorted(concept_to_texts.keys())]
    payload = {"concepts": items}
    return short_hash(payload, length=20)


def build_concept_embeddings(
    model: ModelBundle,
    concept_to_texts: Dict[str, Sequence[str]],
    *,
    pool: str = "cls",
    layer_mode: str = "last4_avg",
    layer_k: Optional[int] = None,
    max_length: int = 64,
    batch_size: int = 16,
    agg: AggMode = "mean",
    normalize_each: bool = True,
    normalize_final: bool = True,
    set_id: str = "default",
    use_cache: bool = True,
) -> Tuple[List[str], np.ndarray]:
    """
    Returns:
      concepts: sorted list of concept names
      X: concept embeddings [C, D] float32

    Cache hit requires:
      - same config
      - same sorted concepts list
      - same templates fingerprint
    """
    concepts = sorted(concept_to_texts.keys())

    cfg = ConceptEmbeddingConfig(
        model_name=model.model_name,
        pool=pool,
        layer_mode=layer_mode,
        layer_k=layer_k,
        max_length=max_length,
        batch_size=batch_size,
        agg=agg,
        normalize_each=normalize_each,
        normalize_final=normalize_final,
        set_id=set_id,
    )
    meta_path, emb_path = _cache_paths(cfg)

    templates_fp = _templates_fingerprint({c: concept_to_texts[c] for c in concepts})

    if use_cache and meta_path.exists() and emb_path.exists():
        meta = load_json(meta_path)
        if meta.get("templates_fingerprint") == templates_fp and meta.get("concepts") == concepts:
            X = load_npy(emb_path).astype(np.float32)
            return concepts, X

    # Flatten templates for a single model call
    flat_texts: List[str] = []
    concept_idx: List[int] = []
    for ci, c in enumerate(concepts):
        texts = list(concept_to_texts[c])
        if not texts:
            raise ValueError(f"Concept '{c}' has empty template list.")
        for t in texts:
            flat_texts.append(t)
            concept_idx.append(ci)

    T = embed_text(
        model,
        flat_texts,
        max_length=max_length,
        batch_size=batch_size,
        pool=pool,
        layer_mode=layer_mode,
        layer_k=layer_k,
        normalize=normalize_each,
    )  # [N_templates, D]

    C = len(concepts)
    D = T.shape[1]
    out = np.zeros((C, D), dtype=np.float32)

    idx_arr = np.asarray(concept_idx, dtype=np.int32)
    for ci in range(C):
        rows = T[idx_arr == ci]
        if agg == "mean":
            v = rows.mean(axis=0)
        elif agg == "median":
            v = np.median(rows, axis=0).astype(np.float32)
        else:
            raise ValueError(f"Unknown agg={agg}")
        out[ci] = v.astype(np.float32)

    if normalize_final:
        out = _l2_normalize(out)

    meta = {
        "config": cfg.to_dict(),
        "concepts": concepts,
        "templates_fingerprint": templates_fp,
        "n_concepts": C,
        "dim": D,
    }
    save_json(meta_path, meta)
    save_npy(emb_path, out)

    return concepts, out
