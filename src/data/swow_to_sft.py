from __future__ import annotations
from dataclasses import dataclass 
from pathlib import Path 
from  typing import Dict, List, Sequence, Tuple, Union
import json 
import numpy as np 
import pandas as pd 

def _norm_ws(s: str) -> str:
    return " ".join(str(s).strip().split())

DEFAULT_PROMPT = (
    "You are a helpful assistant. Given a cue word, output ONE likely associated word. "
    "Return ONLY the associated word."
)

@dataclass(frozen=True)
class SWOWBuildConfig:
    system_prompt: str = DEFAULT_PROMPT
    seed: int = 1234
    val_frac: float = 0.02
    test_frac: float = 0.02
    max_train_examples: int = 50_000
    max_val_examples: int = 2_000
    max_test_examples: int = 2_000
    min_len: int = 1

def load_swow_csv(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SWOW CSV not found: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")

def to_pairs(df: pd.DataFrame) -> pd.DataFrame:
    cols = {column.lower().strip(): column for column in df.columns}
    