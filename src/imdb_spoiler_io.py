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
    # Mapping flessibile per gestire vari format
    text_candidates = ["review_text", "review", "text", "comment", "content"]
    label_candidates = ["is_spoiler", "spoiler", "label", "isSpoiler"]
    movie_candidates = ["movie_id", "movieId", "imdb_id", "imdbId", "movie"]

    def pick(cands: Sequence[str]) -> str:
        for c in cands:
            if c in cols: return c
        raise KeyError(f"Schema error: colonne attese non trovate in {sorted(cols)[:10]}...")

    return ImdbSpoilerSchema(
        text_col=pick(text_candidates),
        label_col=pick(label_candidates),
        movie_id_col=pick(movie_candidates),
    )

def _read_json_any(path: Path) -> pd.DataFrame:
    """Prova a leggere JSONL, se fallisce prova JSON standard."""
    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        return pd.read_json(path, lines=False)

def load_raw_imdb_spoiler_json(raw_dir: Path) -> pd.DataFrame:
    """Carica il dataset JSON. Priorità a IMDB_reviews.json."""
    raw_dir = raw_dir.resolve()
    
    # Priorità al file specifico nominato
    specific_file = raw_dir / "IMDB_reviews.json"
    if specific_file.exists():
        print(f"Loading specific file: {specific_file}")
        return _read_json_any(specific_file)

    # Fallback su qualsiasi json
    files = list(raw_dir.glob("*.json")) + list(raw_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"Nessun file .json trovato in {raw_dir}")
    
    # Se multipli, concatena (utile per dataset sharded)
    print(f"Found {len(files)} JSON files. Concatenating...")
    dfs = []
    for f in sorted(files):
        dfs.append(_read_json_any(f))
    return pd.concat(dfs, ignore_index=True)

def prepare_reviews_dataframe(
    df_raw: pd.DataFrame,
    schema: Optional[ImdbSpoilerSchema] = None,
) -> tuple[pd.DataFrame, ImdbSpoilerSchema]:
    
    schema = schema or _guess_schema(df_raw)
    df = df_raw.copy()

    df = df.rename(columns={
        schema.text_col: "text",
        schema.label_col: "label",
        schema.movie_id_col: "movie_id",
    })

    df = df[["movie_id", "text", "label"]].dropna()
    df["text"] = df["text"].astype(str)

    # Normalizzazione label flessibile
    if df["label"].dtype == object:
         df["label"] = df["label"].astype(str).str.lower().map(
            {"1": 1, "0": 0, "true": 1, "false": 0, "yes": 1, "no": 0}
        )
    
    # Coerce to numeric e fillna
    df["label"] = pd.to_numeric(df["label"], errors='coerce').fillna(0).astype(int)
    
    df["review_id"] = range(len(df))
    return df[["review_id", "movie_id", "text", "label"]], ImdbSpoilerSchema("text", "label", "movie_id")

def save_processed_reviews(
    df: pd.DataFrame,
    processed_dir: Path,
    save_csv: bool = False,
) -> Path:
    """Salva il dataframe processato in formato Parquet (e opzionalmente CSV)."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = processed_dir / "reviews.parquet"
    df.to_parquet(out_parquet, index=False)

    if save_csv:
        out_csv = processed_dir / "reviews.csv"
        df.to_csv(out_csv, index=False)

    return out_parquet