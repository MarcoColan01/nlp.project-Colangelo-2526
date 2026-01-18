# src/utils/config.py
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Paths:
    root: Path
    src: Path
    results: Path


def find_project_root(start: Optional[Path] = None) -> Path:
    """
    Find repository/project root by walking upwards until a directory
    containing 'src' is found.
    """
    cur = (start or Path(__file__).resolve()).parent
    for _ in range(10):
        if (cur / "src").exists() and (cur / "src").is_dir():
            return cur
        cur = cur.parent
    # Fallback: assume src/utils/config.py -> root is 2 levels up from src
    # (root/src/utils/config.py)
    return Path(__file__).resolve().parents[3]


def project_paths(root: Optional[Path] = None, create: bool = True) -> Paths:
    r = root or find_project_root()
    src = r / "src"
    results = src / "results"
    if create:
        results.mkdir(parents=True, exist_ok=True)
    return Paths(root=r, src=src, results=results)


def set_seed(seed: int) -> None:
    """
    Best-effort reproducibility:
    - python random
    - numpy
    - torch (if installed), incl. some determinism flags
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Best-effort determinism (doesn't guarantee full determinism in all ops)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        # torch not installed or not usable: ignore
        pass


def get_device(prefer_cuda: bool = True) -> str:
    """
    Returns 'cuda' if available and prefer_cuda=True, else 'cpu'.
    """
    if not prefer_cuda:
        return "cpu"
    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
