from __future__ import annotations

import numpy as np

from utils import set_seed
from embeddings import load_model
from embeddings.concept_embedding import build_concept_embeddings
from embeddings.api import embed_text


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
    }
    concepts_sorted = sorted(concept_to_texts.keys())

    # 1) Normal build + final normalization
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
        set_id="smoke_v1",
        use_cache=True,
    )
    assert concepts == concepts_sorted
    assert X.shape[0] == len(concepts)
    assert X.ndim == 2
    assert X.dtype == np.float32

    norms = np.linalg.norm(X, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), f"Final embeddings not normalized: {norms}"
    print("OK: built normalized concept embeddings:", X.shape)

    # 2) Aggregation correctness when no normalization
    concepts2, X2 = build_concept_embeddings(
        model,
        concept_to_texts,
        pool="cls",
        layer_mode="last",
        max_length=32,
        batch_size=8,
        agg="mean",
        normalize_each=False,
        normalize_final=False,
        set_id="smoke_v2_no_norm",
        use_cache=False,
    )
    assert concepts2 == concepts

    flat = []
    idx = []
    for ci, c in enumerate(concepts):
        for t in concept_to_texts[c]:
            flat.append(t)
            idx.append(ci)

    T = embed_text(
        model,
        flat,
        max_length=32,
        batch_size=8,
        pool="cls",
        layer_mode="last",
        normalize=False,
    )

    idx_arr = np.asarray(idx)
    for ci, c in enumerate(concepts):
        rows = T[idx_arr == ci]
        manual = rows.mean(axis=0)
        diff = np.max(np.abs(manual - X2[ci]))
        assert diff < 1e-4, f"Mean aggregation mismatch for {c}: maxdiff={diff}"
    print("OK: aggregation mean matches manual mean (no normalization).")

    # 3) Cache hit should return identical
    concepts3, X3 = build_concept_embeddings(
        model,
        concept_to_texts,
        pool="cls",
        layer_mode="last4_avg",
        max_length=32,
        batch_size=8,
        agg="mean",
        normalize_each=True,
        normalize_final=True,
        set_id="smoke_v1",
        use_cache=True,
    )
    assert concepts3 == concepts
    assert np.allclose(X3, X), "Cache load mismatch."
    print("OK: cache returns identical embeddings.")

    print("OK concept embedding smoke: passed")


if __name__ == "__main__":
    main()
