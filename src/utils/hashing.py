# src/utils/hashing.py
from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_json_dumps(obj: Any) -> str:
    """
    Deterministic JSON string (sorted keys, no whitespace noise).
    Works for JSON-serializable objects.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def short_hash(obj: Any, length: int = 12) -> str:
    """
    Convenient short hash for caching keys (e.g. config dictionaries).
    """
    return sha256_of(stable_json_dumps(obj))[:length]
