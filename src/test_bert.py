# test_bert_smoke.py
from __future__ import annotations

import numpy as np

from utils import set_seed
from embeddings import load_model, embed_text


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # assumes a,b already L2 normalized
    return float(np.dot(a, b))


def main() -> None:
    set_seed(42)

    model = load_model("bert-base-uncased", device="cpu")


    texts = [
        "I feel joy.",
        "I feel happiness.",
        "I feel sadness.",
        "The cat sits on the mat.",
    ]

    X = embed_text(
        model,
        texts,
        max_length=32,
        batch_size=4,
        pool="cls",
        layer_mode="last4_avg",
        normalize=True,
    )

    # Basic checks
    assert isinstance(X, np.ndarray)
    assert X.shape[0] == len(texts), f"Expected {len(texts)} rows, got {X.shape[0]}"
    assert X.ndim == 2, f"Expected 2D array, got shape {X.shape}"
    assert X.dtype == np.float32, f"Expected float32, got {X.dtype}"

    # Norm ~ 1 after normalize
    norms = np.linalg.norm(X, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), f"Embeddings not normalized. norms={norms}"

    # Sanity: "joy" should be closer to "happiness" than to "sadness" (not guaranteed but common)
    sim_joy_happy = cosine_sim(X[0], X[1])
    sim_joy_sad = cosine_sim(X[0], X[2])
    sim_joy_cat = cosine_sim(X[0], X[3])
    assert sim_joy_happy >= sim_joy_cat + 0.05


    print("cos(joy, happiness) =", sim_joy_happy)
    print("cos(joy, sadness)   =", sim_joy_sad)



    # Soft assertion: allow small margin, but usually should hold
    assert sim_joy_happy >= sim_joy_sad - 0.02, (
        "Unexpected similarity ordering: joy not closer to happiness than sadness "
        f"(joy-happy={sim_joy_happy:.4f}, joy-sad={sim_joy_sad:.4f})"
    )

    print("OK bert smoke: embeddings shape/norm/sanity passed")


if __name__ == "__main__":
    main()
