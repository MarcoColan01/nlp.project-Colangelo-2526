# src/utils/io.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np


PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: PathLike, obj: Any, indent: int = 2) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    return p


def load_json(path: PathLike) -> Any:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_npy(path: PathLike, arr: np.ndarray) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    np.save(p, arr)
    return p


def load_npy(path: PathLike) -> np.ndarray:
    p = Path(path)
    return np.load(p, allow_pickle=False)


def save_text(path: PathLike, text: str) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        f.write(text)
    return p


def load_text(path: PathLike) -> str:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return f.read()
