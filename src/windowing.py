from __future__ import annotations 
from dataclasses import dataclass
from pathlib import Path 
from typing import Optional 
import pandas as pd
from transformers import AutoTokenizer 

@dataclass(frozen=True)
class WindowConfig:
    tokenizer_name: str
    max_len: int = 256
    stride: int = 128 

def build_tokenizer(tokenizer_name: str):
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    return tok 
    
def create_chunks_dataframe(
            df_reviews: pd.DataFrame,
            tokenizer,
            text_col: str = "text",
            label_col: str = "label",
            review_id_col: str = "review_id",
            movie_id_col: str = "movie_id",
            max_len: int = 256,
            stride: int = 128,
    ) -> pd.DataFrame:
    required = {text_col, label_col, review_id_col, movie_id_col}
    missing = required - set(df_reviews.columns)
    if missing:
        raise KeyError(f"Missing columns in df_reviews: {missing}.\nDisponibili: {list(df_reviews.columns)}")
    texts = df_reviews[text_col].astype(str).tolist()

    enc = tokenizer(
            texts,
            truncation=True,
            max_length=max_len,
            stride = stride,
            return_overflowing_tokens=True,
            return_attention_mask = True,
            padding=False
        )

    overflow_map = enc.get("overflow_to_sample_mapping")
    if overflow_map is None:
        raise RuntimeError("Tokenizer has not returned overflow_to_sample_mapping. Verify use_fast=True and params.")
        
    counters ={}
    chunk_index=[]
    for sample_id in overflow_map:
            c = counters.get(sample_id, 0)
            chunk_index.append(c)
            counters[sample_id] = c+1

    df_chunks = pd.DataFrame({
            "review_id": [df_reviews.iloc[i][review_id_col] for i in overflow_map],
            "movie_id":  [df_reviews.iloc[i][movie_id_col] for i in overflow_map],
            "label":     [int(df_reviews.iloc[i][label_col]) for i in overflow_map],
            "chunk_index": chunk_index,
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        })

    if "token_type_ids" in enc:
        df_chunks["token_type_ids"] = enc["token_type_ids"]

    df_chunks.sort_values(["review_id", "chunk_index"], inplace=True)
    df_chunks.reset_index(drop=True, inplace=True)
    return df_chunks
    
def save_chunks(df_chunks: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_chunks.to_parquet(out_path, index=False)
    return out_path
    
def load_chunks(path: Path) -> pd.Dataframe:
    return pd.read_parquet(path)
    
def build_and_save_split_chunks(
    in_split_path: Path,
    out_chunks_path: Path,
    cfg: WindowingConfig,
    text_col: str = "text",
) -> Path:
    df = pd.read_parquet(in_split_path)
    tok = build_tokenizer(cfg.tokenizer_name)
    df_chunks = create_chunks_dataframe(
        df_reviews=df,
        tokenizer=tok,
        text_col=text_col,
        max_len=cfg.max_len,
        stride=cfg.stride,
    )
    return save_chunks(df_chunks, out_chunks_path)