"""
SWOW-EN word association -> SFT JSONL (small) for teacher warmup.

We only use:
- cue
- R1/R2/R3 (corrected) or R1Raw/R2Raw/R3Raw (fallback)

Participant metadata is ignored.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

def _norm_ws(s: str) -> str:
    return " ".join(str(s).strip().split())

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Given a cue word, output ONE likely associated word. "
    "Return ONLY the associated word."
)

@dataclass(frozen=True)
class SWOWBuildConfig:
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
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
    cols = {c.lower().strip(): c for c in df.columns}
    if "cue" not in cols:
        raise KeyError(f"Missing 'cue' column. Found: {list(df.columns)}")
    cue_c = cols["cue"]

    corrected = [k for k in ["r1","r2","r3"] if k in cols]
    raw = [k for k in ["r1raw","r2raw","r3raw"] if k in cols]
    use_cols = corrected if corrected else raw

    if use_cols:
        work = df[[cue_c] + [cols[k] for k in use_cols]].copy()
        work = work.rename(columns={cue_c: "cue"})
        rows = []
        for _, row in work.iterrows():
            cue = _norm_ws(row["cue"])
            for k in use_cols:
                resp = row[cols[k]]
                if pd.isna(resp):
                    continue
                resp = _norm_ws(resp)
                if resp:
                    rows.append({"cue": cue, "response": resp, "count": 1})
        return pd.DataFrame(rows)

    if "response" in cols:
        resp_c = cols["response"]
        cnt_c = cols.get("count") or cols.get("frequency") or cols.get("freq")
        if cnt_c is None:
            work = df[[cue_c, resp_c]].copy()
            work.columns = ["cue", "response"]
            work["count"] = 1
        else:
            work = df[[cue_c, resp_c, cnt_c]].copy()
            work.columns = ["cue", "response", "count"]
        work["cue"] = work["cue"].astype(str).map(_norm_ws)
        work["response"] = work["response"].astype(str).map(_norm_ws)
        work["count"] = pd.to_numeric(work["count"], errors="coerce").fillna(1).astype(int)
        return work

    raise ValueError("Unsupported SWOW format: need R1/R2/R3 (or Raw) or response(+count).")

def expand_pairs(pairs: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    pairs = pairs[pairs["cue"].notna() & pairs["response"].notna()].copy()
    pairs = pairs[(pairs["cue"].str.len() > 0) & (pairs["response"].str.len() > 0)]
    pairs["count"] = pairs["count"].clip(lower=1, upper=1000)
    pairs = pairs.loc[pairs.index.repeat(pairs["count"])][["cue", "response"]].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    return pairs.iloc[rng.permutation(len(pairs))].reset_index(drop=True)

def split_pairs(pairs: pd.DataFrame, cfg: SWOWBuildConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(pairs)
    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(n)
    n_test = max(1, int(round(n * cfg.test_frac)))
    n_val = max(1, int(round(n * cfg.val_frac)))
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError("Not enough data for split; reduce val/test frac.")
    train = pairs.iloc[idx[:n_train]].reset_index(drop=True)
    val = pairs.iloc[idx[n_train:n_train+n_val]].reset_index(drop=True)
    test = pairs.iloc[idx[n_train+n_val:]].reset_index(drop=True)
    return train, val, test

def build_records(df_pairs: pd.DataFrame, *, split: str, cfg: SWOWBuildConfig, max_examples: int) -> List[Dict]:
    out: List[Dict] = []
    df_pairs = df_pairs.iloc[:max_examples].copy()
    for i, row in df_pairs.iterrows():
        cue = row["cue"]
        resp = row["response"]
        if len(cue) < cfg.min_len or len(resp) < cfg.min_len:
            continue
        user = f"Cue: {cue}\nAssociated word:"
        out.append(
            {
                "id": f"swow_{split}_{i}",
                "split": split,
                "source": "swow",
                "cue": cue,
                "messages": [
                    {"role": "system", "content": cfg.system_prompt},
                    {"role": "user", "content": user},
                ],
                "assistant": resp,
            }
        )
    return out
