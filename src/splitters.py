from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitConfig:
    train_size: float = 0.8
    val_size: float = 0.1
    test_size: float = 0.1
    seed: int = 42
    max_tries: int = 500


def _check_sizes(cfg: SplitConfig) -> None:
    s = cfg.train_size + cfg.val_size + cfg.test_size
    if not np.isclose(s, 1.0):
        raise ValueError(f"train+val+test deve essere 1.0, trovato {s}")


def _group_sample_split(
    df: pd.DataFrame,
    group_col: str,
    label_col: str,
    test_size: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = df[group_col].unique()
    n_test = max(1, int(round(len(groups) * test_size)))
    test_groups = set(rng.choice(groups, size=n_test, replace=False).tolist())
    is_test = df[group_col].isin(test_groups)
    return df[~is_test].copy(), df[is_test].copy()


def _score_split_balance(df_full: pd.DataFrame, df_a: pd.DataFrame, df_b: pd.DataFrame, label_col: str) -> float:
    """
    Quanto le medie delle label di A e B sono vicine a quella globale.
    Più basso = meglio.
    """
    global_mean = df_full[label_col].mean()
    return abs(df_a[label_col].mean() - global_mean) + abs(df_b[label_col].mean() - global_mean)


def _has_both_classes(df: pd.DataFrame, label_col: str) -> bool:
    vals = set(df[label_col].unique().tolist())
    return (0 in vals) and (1 in vals)


def split_by_movie_id(
    df: pd.DataFrame,
    cfg: SplitConfig,
    group_col: str = "movie_id",
    label_col: str = "label",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _check_sizes(cfg)
    rng0 = np.random.default_rng(cfg.seed)

    # 1) split off test
    best = None
    best_score = float("inf")

    for t in range(cfg.max_tries):
        rng = np.random.default_rng(rng0.integers(0, 2**32 - 1))
        df_rest, df_test = _group_sample_split(df, group_col, label_col, cfg.test_size, rng)

        if not _has_both_classes(df_test, label_col):
            continue
        if not _has_both_classes(df_rest, label_col):
            continue

        score = _score_split_balance(df, df_rest, df_test, label_col)
        if score < best_score:
            best_score = score
            best = (df_rest, df_test)

    if best is None:
        raise RuntimeError("Impossibile creare uno split test bilanciato. Aumenta max_tries o controlla i dati.")

    df_rest, df_test = best

    # 2) split rest into train/val (val_size relative to full -> convert relative to rest)
    val_rel = cfg.val_size / (cfg.train_size + cfg.val_size)

    best = None
    best_score = float("inf")
    for t in range(cfg.max_tries):
        rng = np.random.default_rng(rng0.integers(0, 2**32 - 1))
        df_train, df_val = _group_sample_split(df_rest, group_col, label_col, val_rel, rng)

        if not _has_both_classes(df_train, label_col):
            continue
        if not _has_both_classes(df_val, label_col):
            continue

        score = _score_split_balance(df_rest, df_train, df_val, label_col)
        if score < best_score:
            best_score = score
            best = (df_train, df_val)

    if best is None:
        raise RuntimeError("Impossibile creare train/val bilanciati. Aumenta max_tries o controlla i dati.")

    df_train, df_val = best

    # order + reset index
    for d in (df_train, df_val, df_test):
        d.sort_values(["movie_id", "review_id"], inplace=True)
        d.reset_index(drop=True, inplace=True)

    return df_train, df_val, df_test


def save_splits(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    processed_dir: Path,
) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "train.parquet").write_bytes(df_train.to_parquet(index=False))
    (processed_dir / "val.parquet").write_bytes(df_val.to_parquet(index=False))
    (processed_dir / "test.parquet").write_bytes(df_test.to_parquet(index=False))
