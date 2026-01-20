# src/maps/similarity.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Row-wise L2 normalization. Safe even if rows contain zeros.
    """
    X = X.astype(np.float32, copy=False)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms


def cosine_similarity_matrix(X: np.ndarray, assume_normalized: bool = True) -> np.ndarray:
    """
    Compute cosine similarity matrix S = X X^T (if normalized).
    Returns float32 [N, N].
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D [N,D], got shape {X.shape}")
    Xn = X if assume_normalized else l2_normalize(X)
    S = (Xn @ Xn.T).astype(np.float32)
    # numerical guard
    np.clip(S, -1.0, 1.0, out=S)
    return S


def topk_neighbors_from_sim(
    concepts: Sequence[str],
    S: np.ndarray,
    k: int = 5,
    include_self: bool = False,
) -> List[List[Tuple[str, float]]]:
    """
    For each concept i, return list of (neighbor_name, sim) length k.
    Uses similarity matrix S [N,N].

    include_self=False excludes i itself from candidates.
    """
    n = len(concepts)
    if S.shape != (n, n):
        raise ValueError(f"S shape {S.shape} doesn't match n={n}")

    out: List[List[Tuple[str, float]]] = []

    for i in range(n):
        sims = S[i].copy()

        if not include_self:
            sims[i] = -np.inf

        # argpartition for speed then sort
        kk = min(k, n - (0 if include_self else 1))
        idx = np.argpartition(-sims, range(kk))[:kk]
        idx = idx[np.argsort(-sims[idx])]

        neigh = [(concepts[j], float(sims[j])) for j in idx]
        out.append(neigh)

    return out


def topk_neighbors(
    concepts: Sequence[str],
    X: np.ndarray,
    k: int = 5,
    assume_normalized: bool = True,
    include_self: bool = False,
) -> List[List[Tuple[str, float]]]:
    """
    Convenience wrapper: compute sim matrix then top-k neighbors.
    """
    S = cosine_similarity_matrix(X, assume_normalized=assume_normalized)
    return topk_neighbors_from_sim(concepts, S, k=k, include_self=include_self)
