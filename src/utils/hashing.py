from __future__ import annotations
from typing import Sequence 
import hashlib 

def _norm_word(w: str) -> str:
    return " ".join(str(w).strip().split()).casefold()

def puzzle_hash(words16: Sequence[str]) -> str:
    norm = sorted([_norm_word(word) for word in words16])
    return hashlib.sha256(("|".join(norm)).encode("utf-8")).hexdigest()