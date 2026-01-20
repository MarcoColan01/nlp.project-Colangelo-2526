# src/tests/test_similarity_smoke.py
from __future__ import annotations

import numpy as np

from utils import set_seed
from embeddings import load_model
from embeddings.concept_embedding import build_concept_embeddings
from maps.similarity import cosine_similarity_matrix, topk_neighbors_from_sim


def main() -> None:
    set_seed(42)
    model = load_model("bert-base-uncased", device="cpu")

    concept_to_texts = {
        "joy": [
            "I feel joy.",
            "She was filled with joy.",
            "Joy is an emotion.",
        ],
        "sadness": [
            "I feel sadness.",
            "He was overwhelmed by sadness.",
            "Sadness is an emotion.",
        ],
        "anger": [
            "I feel anger.",
            "They were consumed by anger.",
            "Anger is an emotion.",
        ],
        "happiness": [
            "I feel happiness.",
            "She experienced happiness.",
            "Happiness is an emotion.",
        ],
    }

    concepts, X = build_concept_embeddings(
        model,
        concept_to_texts,
        pool="cls",
        layer_mode="last4_avg",
        max_length=32,
        batch_size=8,
        agg="mean",
        normalize_each=True,
        normalize_final=True,
        set_id="sim_smoke_v1",
        use_cache=True,
    )

    S = cosine_similarity_matrix(X, assume_normalized=True)

    # Shape checks
    n = len(concepts)
    assert S.shape == (n, n), f"Expected {(n,n)}, got {S.shape}"

    # Symmetry
    assert np.allclose(S, S.T, atol=1e-5), "Similarity matrix not symmetric"

    # Diagonal ~ 1
    diag = np.diag(S)
    assert np.allclose(diag, 1.0, atol=1e-3), f"Diagonal not ~1: {diag}"

    # Top-k neighbors
    nn = topk_neighbors_from_sim(concepts, S, k=2, include_self=False)
    for i, c in enumerate(concepts):
        names = [name for name, _ in nn[i]]
        assert c not in names, f"Self included in neighbors for {c}"
        assert len(names) == 2, f"Expected 2 neighbors for {c}, got {len(names)}"

    # Print neighbors for manual sanity
    print("Concepts:", concepts)
    for i, c in enumerate(concepts):
        print(f"{c:10s} -> {nn[i]}")

    print("OK similarity smoke: passed")


if __name__ == "__main__":
    main()
