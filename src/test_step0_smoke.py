# test_step0_smoke.py
from __future__ import annotations

from pathlib import Path
import numpy as np

from utils import (
    project_paths,
    set_seed,
    save_json,
    load_json,
    save_npy,
    load_npy,
    short_hash,
)


def main() -> None:
    paths = project_paths(create=True)
    assert paths.src.exists(), f"src folder not found at {paths.src}"
    assert paths.results.exists(), f"results folder not created at {paths.results}"

    # --- JSON roundtrip
    obj1 = {"b": 2, "a": 1, "nested": {"z": 0, "y": 1}}
    json_path = paths.results / "_smoke" / "test.json"
    save_json(json_path, obj1)
    obj2 = load_json(json_path)
    assert obj2 == obj1, "JSON roundtrip failed"

    # --- NPY roundtrip
    arr1 = np.random.randn(5, 3).astype(np.float32)
    npy_path = paths.results / "_smoke" / "test.npy"
    save_npy(npy_path, arr1)
    arr2 = load_npy(npy_path)
    assert np.allclose(arr1, arr2), "NPY roundtrip failed"

    # --- stable hash (order should not matter)
    conf_a = {"model": "bert", "pooling": "cls", "layer": 12, "seed": 42}
    conf_b = {"seed": 42, "layer": 12, "pooling": "cls", "model": "bert"}
    ha = short_hash(conf_a)
    hb = short_hash(conf_b)
    assert ha == hb, "Stable hashing failed (key order changed hash)"

    # --- seed sanity (not strict determinism, but should set env and numpy)
    set_seed(123)
    x1 = np.random.rand(3)
    set_seed(123)
    x2 = np.random.rand(3)
    assert np.allclose(x1, x2), "NumPy seed reproducibility failed"

    print("OK step0: paths/io/hashing/seed working")


if __name__ == "__main__":
    main()
