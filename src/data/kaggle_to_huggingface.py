'''
This module contains utilities to convert datasets from Kaggle format to Hugging Face Datasets format.
'''

from __future__ import annotations 
from dataclasses import dataclass
from pathlib import Path 
from typing import Dict, List, Optional, Sequence, Tuple, Union
import pandas as pd
import numpy as np
import json 
import hashlib 

DEFAULT_ALIASES = {
    "date": ["date", "puzzle_date", "day"],
    "category": ["category", "answerdescription", "group", "label"],
    "w1": ["w1", "word1", "word_1"],
    "w2": ["w2", "word2", "word_2"],    
    "w3": ["w3", "word3", "word_3"],
    "w4": ["w4", "word4", "word_4"],
}

@dataclass(frozen=True)
class KaggleToHFConfig:
    seed: int = 1234
    require_four_groups: bool = True 
    require_four_words: bool = True
    drop_incomplete_dates: bool = True
    normalize_whitespace: bool = True
    shuffle_words: bool = True

def _normalize_ws(s: str) -> str: 
    return " ".join(str(s).strip().split()) 

def _find_column(df: pd.DataFrame, logical_name: str, aliases: Dict[str, List[str]]) -> str:
    cols_lower = {column.lower(): column for column in df.columns} 
    for alias in aliases.get(logical_name, []):
        if alias.lower() in cols_lower:
            return cols_lower[alias.lower()]
    raise KeyError(
        f"Missing column for '{logical_name}'. Expected one of: {aliases.get(logical_name)}. "
        f"Found: {list(df.columns)}"
    ) 

def load_kaggle_csv(csv_path: Union[str, Path], *, encoding: Optional[str] = None) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return pd.read_csv(csv_path, encoding=encoding)

