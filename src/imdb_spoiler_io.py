from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd


@dataclass(frozen=True)
class ImdbSpoilerSchema:
    text_col: str
    label_col: str
    movie_id_col: str


def _guess_schema(df: pd.DataFrame) -> ImdbSpoilerSchema:
    cols = set(df.columns)

    text_candidates = ["review_text", "review", "text", "comment", "content"]
    label_candidates = ["is_spoiler", "spoiler", "label", "isSpoiler"]
    movie_candidates = ["movie_id", "movieId", "imdb_id", "imdbId", "movie"]

    def pick(cands: Sequence[str]) -> str:
        for c in cands:
            if c in cols:
                return c
        raise KeyError(f"Impossibile determinare colonna tra: {cands}. Colonne disponibili: {sorted(cols)[:50]}...")

    return ImdbSpoilerSchema(
        text_col=pick(text_candidates),
        label_col=pick(label_candidates),
        movie_id_col=pick(movie_candidates),
    )


def _read_json_any(path: Path) -> pd.DataFrame:
    """
    Prova a leggere JSONL (lines=True), se fallisce prova JSON standard.
    """
    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        return pd.read_json(path, lines=False)


def load_raw_imdb_spoiler_json(raw_dir: Path) -> pd.DataFrame:
    """
    Cerca in data/raw il primo file json/jsonl e lo carica.
    Se ce ne sono più di uno, concatena.
    """
    raw_dir = raw_dir.resolve()
    files = list(raw_dir.glob("*.json")) + list(raw_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"Nessun .json/.jsonl trovato in {raw_dir}")

    dfs = []
    for f in sorted(files):
        df = _read_json_any(f)
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    return out


def prepare_reviews_dataframe(
    df_raw: pd.DataFrame,
    schema: Optional[ImdbSpoilerSchema] = None,
) -> tuple[pd.DataFrame, ImdbSpoilerSchema]:
    """
    Standardizza in colonne:
      - review_id (int)
      - movie_id  (str/int)
      - text      (str)
      - label     (int 0/1)
    """
    schema = schema or _guess_schema(df_raw)

    df = df_raw.copy()

    # rename to standard names
    df = df.rename(
        columns={
            schema.text_col: "text",
            schema.label_col: "label",
            schema.movie_id_col: "movie_id",
        }
    )

    # basic cleaning
    df = df[["movie_id", "text", "label"]].dropna()
    df["text"] = df["text"].astype(str)

    # normalize label
    # accetta bool, int, str "0"/"1"/"true"/"false"
    if df["label"].dtype == bool:
        df["label"] = df["label"].astype(int)
    else:
        df["label"] = df["label"].astype(str).str.lower().map(
            {"1": 1, "0": 0, "true": 1, "false": 0, "yes": 1, "no": 0}
        )
        if df["label"].isna().any():
            # fallback: prova cast numerico
            df["label"] = pd.to_numeric(df["label"], errors="coerce")
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)

    df["review_id"] = range(len(df))
    df = df[["review_id", "movie_id", "text", "label"]]

    # aggiorna schema standard
    schema_std = ImdbSpoilerSchema(text_col="text", label_col="label", movie_id_col="movie_id")
    return df, schema_std


def save_processed_reviews(
    df: pd.DataFrame,
    processed_dir: Path,
    save_csv: bool = False,
) -> Path:
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = processed_dir / "reviews.parquet"
    df.to_parquet(out_parquet, index=False)

    if save_csv:
        out_csv = processed_dir / "reviews.csv"
        df.to_csv(out_csv, index=False)

    return out_parquet
