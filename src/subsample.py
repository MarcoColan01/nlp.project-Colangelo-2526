from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SubsampleConfig:
    total_reviews: int = 80_000
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1
    seed: int = 42
    label_col: str = "label"


def _targets_from_total(total: int, train_frac: float, val_frac: float, test_frac: float) -> tuple[int, int, int]:
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac deve fare 1.0")

    n_train = int(round(total * train_frac))
    n_val = int(round(total * val_frac))
    n_test = total - n_train - n_val  # assicura somma esatta

    # evita 0
    n_train = max(1, n_train)
    n_val = max(1, n_val)
    n_test = max(1, n_test)
    return n_train, n_val, n_test


def stratified_sample_by_label(df: pd.DataFrame, n: int, label_col: str = "label", seed: int = 42) -> pd.DataFrame:
    """
    Campiona n righe preservando (per quanto possibile) la proporzione della label.
    """
    if n >= len(df):
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if label_col not in df.columns:
        raise KeyError(f"label_col '{label_col}' non trovato in df")

    df0 = df[df[label_col] == 0]
    df1 = df[df[label_col] == 1]

    n0 = len(df0)
    n1 = len(df1)
    if n0 == 0 or n1 == 0:
        # caso estremo: una classe assente
        return df.sample(n=n, random_state=seed).reset_index(drop=True)

    pos_rate = n1 / (n0 + n1)
    n1_tgt = int(round(n * pos_rate))
    n0_tgt = n - n1_tgt

    # clamp se una classe non ha abbastanza esempi
    n1_tgt = min(n1_tgt, n1)
    n0_tgt = min(n0_tgt, n0)

    # se abbiamo perso esempi per clamp, riempi con l'altra classe
    missing = n - (n0_tgt + n1_tgt)
    if missing > 0:
        # prova a riempire prima con 0, poi con 1 (o viceversa, indifferente)
        extra0 = min(missing, n0 - n0_tgt)
        n0_tgt += extra0
        missing -= extra0
        if missing > 0:
            extra1 = min(missing, n1 - n1_tgt)
            n1_tgt += extra1
            missing -= extra1

    out = pd.concat([
        df0.sample(n=n0_tgt, random_state=seed),
        df1.sample(n=n1_tgt, random_state=seed),
    ], ignore_index=True)

    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def subsample_splits_to_total(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: SubsampleConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_train, n_val, n_test = _targets_from_total(cfg.total_reviews, cfg.train_frac, cfg.val_frac, cfg.test_frac)

    train_sub = stratified_sample_by_label(train_df, n_train, label_col=cfg.label_col, seed=cfg.seed)
    val_sub = stratified_sample_by_label(val_df, n_val, label_col=cfg.label_col, seed=cfg.seed + 1)
    test_sub = stratified_sample_by_label(test_df, n_test, label_col=cfg.label_col, seed=cfg.seed + 2)

    return train_sub, val_sub, test_sub


def save_subsampled_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    processed_dir: Path,
    prefix: str = "80k",
) -> tuple[Path, Path, Path]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    p_train = processed_dir / f"train_{prefix}.parquet"
    p_val = processed_dir / f"val_{prefix}.parquet"
    p_test = processed_dir / f"test_{prefix}.parquet"

    train_df.to_parquet(p_train, index=False)
    val_df.to_parquet(p_val, index=False)
    test_df.to_parquet(p_test, index=False)

    return p_train, p_val, p_test
