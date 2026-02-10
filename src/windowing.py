from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


@dataclass(frozen=True)
class WindowingConfig:
    tokenizer_name: str
    max_length: int = 256
    stride: int = 128
    review_batch_size: int = 512  # quante review tokenizzare per batch in streaming


def build_tokenizer(tokenizer_name: str):
    return AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)


def create_chunks_dataframe(
    df_reviews: pd.DataFrame,
    tokenizer,
    text_col: str = "text",
    label_col: str = "label",
    review_id_col: str = "review_id",
    movie_id_col: str = "movie_id",
    max_length: int = 256,
    stride: int = 128,
) -> pd.DataFrame:
    """
    Chunkizza in memoria un batch di review. Usata internamente dallo streaming.
    """
    required = {text_col, label_col, review_id_col, movie_id_col}
    missing = required - set(df_reviews.columns)
    if missing:
        raise KeyError(f"Colonne mancanti: {missing}")

    df_reviews = df_reviews.reset_index(drop=True)
    texts = df_reviews[text_col].astype(str).tolist()

    enc = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_attention_mask=True,
        padding=False,
    )

    overflow = enc.get("overflow_to_sample_mapping")
    if overflow is None:
        raise RuntimeError("Tokenizer non ha restituito overflow_to_sample_mapping (serve tokenizer fast).")

    overflow = np.asarray(overflow, dtype=np.int64)

    # chunk_index per review (conteggio per sample nel batch)
    counters = {}
    chunk_idx = np.empty(len(overflow), dtype=np.int64)
    for i, sample_id in enumerate(overflow.tolist()):
        c = counters.get(sample_id, 0)
        chunk_idx[i] = c
        counters[sample_id] = c + 1

    review_ids = df_reviews[review_id_col].to_numpy()
    movie_ids = df_reviews[movie_id_col].to_numpy()
    labels = df_reviews[label_col].to_numpy()

    df_chunks = pd.DataFrame({
        "review_id": review_ids[overflow],
        "movie_id": movie_ids[overflow],
        "label": labels[overflow].astype(int),
        "chunk_index": chunk_idx,
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
    })

    if "token_type_ids" in enc:
        df_chunks["token_type_ids"] = enc["token_type_ids"]

    df_chunks.sort_values(["review_id", "chunk_index"], inplace=True)
    df_chunks.reset_index(drop=True, inplace=True)
    return df_chunks


def chunk_parquet_streaming(
    in_reviews_parquet: Path,
    out_chunks_parquet: Path,
    cfg: WindowingConfig,
    text_col: str = "text",
    label_col: str = "label",
    review_id_col: str = "review_id",
    movie_id_col: str = "movie_id",
    compression: str = "snappy",
) -> Path:
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    out_chunks_parquet.parent.mkdir(parents=True, exist_ok=True)

    tok = build_tokenizer(cfg.tokenizer_name)

    dataset = ds.dataset(str(in_reviews_parquet), format="parquet")

    # ✅ compatibile: scanner() invece di scan()
    scanner = dataset.scanner(
        columns=[review_id_col, movie_id_col, text_col, label_col],
        batch_size=cfg.review_batch_size,
    )

    reader = scanner.to_reader()

    writer = None
    rows_written = 0

    for record_batch in reader:
        df_batch = record_batch.to_pandas()
        if df_batch.empty:
            continue

        df_chunks = create_chunks_dataframe(
            df_reviews=df_batch,
            tokenizer=tok,
            text_col=text_col,
            label_col=label_col,
            review_id_col=review_id_col,
            movie_id_col=movie_id_col,
            max_length=cfg.max_length,
            stride=cfg.stride,
        )

        table = pa.Table.from_pandas(df_chunks, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(str(out_chunks_parquet), table.schema, compression=compression)

        writer.write_table(table)
        rows_written += len(df_chunks)

    if writer is not None:
        writer.close()

    if rows_written == 0:
        raise RuntimeError("Nessun chunk scritto: controlla input parquet e colonne.")

    return out_chunks_parquet
