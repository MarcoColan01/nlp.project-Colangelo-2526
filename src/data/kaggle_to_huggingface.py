"""
Kaggle -> HuggingFace-like conversion for NYT Connections datasets.

Supports This Kaggle CSV format:

(B) "word-per-row" (16 rows per date) like Connections_Data.csv:
    Game ID, Puzzle Date, Word, Group Name, Group Level, Starting Row, Starting Column

This module groups rows by date and emits one puzzle record per date in a HF-like schema:
{
  "puzzle_id": "kaggle_YYYY-MM-DD",
  "date": "YYYY-MM-DD",
  "words": [... 16 ...],           # deterministic shuffle by default
  "answers": [
      {"answerDescription": "...", "words": ["w1","w2","w3","w4"]},
      ... (4)
  ],
  "metadata": {"source": "kaggle", ...}
}

Includes optional filtering by minimum date (inclusive) to avoid overlap with HF dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import pandas as pd
import numpy as np
import json
import hashlib


DEFAULT_ALIASES = {
    "date": ["puzzle date", "puzzle_date", "date"],
    "word": ["word", "w"],
    "group": ["group name", "group_name", "category", "group"],
    "game_id": ["game id", "game_id", "id"],
}


@dataclass(frozen=True)
class KaggleToHFConfig:
    seed: int = 1234
    normalize_whitespace: bool = True
    shuffle_words: bool = True

    # filtering
    min_date_inclusive: Optional[str] = None  # e.g., "2025-03-25"

    # strictness / cleanup
    drop_incomplete_dates: bool = True
    require_four_groups: bool = True
    require_sixteen_words: bool = True


def _normalize_ws(s: str) -> str:
    return " ".join(str(s).strip().split())


def _find_column(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in cols_lower:
            return cols_lower[a.lower()]
    return None


def _parse_date_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date.astype(str)


def _apply_min_date_filter(df: pd.DataFrame, date_col: str, min_date_inclusive: Optional[str]) -> pd.DataFrame:
    if not min_date_inclusive:
        return df
    md = pd.to_datetime(min_date_inclusive).date()
    d = pd.to_datetime(df[date_col], errors="coerce").dt.date
    return df[d >= md].copy()


def load_kaggle_csv(csv_path: Union[str, Path], *, encoding: Optional[str] = None) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return pd.read_csv(csv_path, encoding=encoding)


def kaggle_rows_to_puzzles(df: pd.DataFrame, *, config: KaggleToHFConfig = KaggleToHFConfig()) -> List[Dict]:
    # Detect format B
    date_b = _find_column(df, DEFAULT_ALIASES["date"])
    word_b = _find_column(df, DEFAULT_ALIASES["word"])
    group_b = _find_column(df, DEFAULT_ALIASES["group"])
    if date_b and word_b and group_b:
        return _format_b(df, config=config, date_col=date_b, word_col=word_b, group_col=group_b)


    raise KeyError(
        "Could not auto-detect Kaggle format. "
        "Expected Puzzle Date + Word + Group Name. "
        f"Columns found: {list(df.columns)}"
    )


def _make_words_list(all_words: List[str], *, rng: np.random.Generator, shuffle: bool) -> List[str]:
    if shuffle:
        idx = rng.permutation(len(all_words))
        return [all_words[i] for i in idx]
    return sorted(all_words, key=lambda x: x.casefold())


def _format_b(df: pd.DataFrame, *, config: KaggleToHFConfig, date_col: str, word_col: str, group_col: str) -> List[Dict]:
    game_col = _find_column(df, DEFAULT_ALIASES["game_id"])  # optional

    cols = [date_col, word_col, group_col] + ([game_col] if game_col else [])
    work = df[cols].copy()

    if config.normalize_whitespace:
        work[word_col] = work[word_col].astype(str).map(_normalize_ws)
        work[group_col] = work[group_col].astype(str).map(_normalize_ws)

    work[date_col] = _parse_date_series(work[date_col])
    work = _apply_min_date_filter(work, date_col, config.min_date_inclusive)

    rng = np.random.default_rng(config.seed)
    puzzles: List[Dict] = []

    group_keys = [date_col, game_col] if game_col else [date_col]

    for key, g in work.groupby(group_keys, sort=True):
        if game_col:
            date, game_id = key
        else:
            date, game_id = key, None

        g = g.dropna()

        grouped = g.groupby(group_col, sort=True)[word_col].apply(list)

        if config.require_four_groups and len(grouped) != 4:
            if config.drop_incomplete_dates:
                continue
            raise ValueError(f"{key}: expected 4 groups, found {len(grouped)}")

        answers = []
        all_words: List[str] = []
        ok = True
        for grp_name, words in grouped.items():
            words = [str(w) for w in words]
            if any(w.strip() == "" or w.lower() == "nan" for w in words) or len(words) != 4:
                ok = False
                break
            answers.append({"answerDescription": str(grp_name), "words": words})
            all_words.extend(words)

        if not ok:
            if config.drop_incomplete_dates:
                continue
            raise ValueError(f"{key}: incomplete group(s)")

        if config.require_sixteen_words and len(all_words) != 16:
            if config.drop_incomplete_dates:
                continue
            raise ValueError(f"{key}: expected 16 words, got {len(all_words)}")

        pid = f"kaggle_{date}" if game_id is None else f"kaggle_{date}_{game_id}"
        puzzles.append(
            {
                "puzzle_id": pid,
                "date": date,
                "words": _make_words_list(all_words, rng=rng, shuffle=config.shuffle_words),
                "answers": answers,
                "metadata": {"source": "kaggle", "seed": config.seed, "format": "B", "game_id": game_id},
            }
        )

    return puzzles


def puzzle_hash(words16: Sequence[str]) -> str:
    norm = sorted([_normalize_ws(w).casefold() for w in words16])
    return hashlib.sha256(("|".join(norm)).encode("utf-8")).hexdigest()


def save_jsonl(records: Sequence[Dict], out_path: Union[str, Path]) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def quick_validate(records: Sequence[Dict]) -> tuple[int, int]:
    suspect = 0
    for r in records:
        if len(r.get("words", [])) != 16:
            suspect += 1
            continue
        ans = r.get("answers", [])
        if len(ans) != 4 or any(len(a.get("words", [])) != 4 for a in ans):
            suspect += 1
    return len(records), suspect
