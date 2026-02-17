"""Utility script: crea split movie-aware e salva versioni tokenizzate head+tail.

Pensato per:
  - full dataset (≈570k) senza tokenizzare tutto in RAM con pandas
  - caching e map batched via HuggingFace Datasets

Uso tipico (da root progetto):
  python src/prepare_tokenized_splits.py --model bert-base-uncased
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset

from project_paths import get_paths
from imdb_spoiler_io import load_raw_imdb_spoiler_json, prepare_reviews_dataframe
from splitters import SplitConfig, split_by_movie_id
from subsample import SubsampleConfig, subsample_splits_to_total
from head_tail import head_tail_tokenize_batch
from teacher_finetune_headtail import build_teacher_tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="bert-base-uncased")
    ap.add_argument("--train_frac", type=float, default=0.84)
    ap.add_argument("--val_frac", type=float, default=0.08)
    ap.add_argument("--test_frac", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subsample_total", type=int, default=0, help="0 = usa tutto")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--head_len", type=int, default=128)
    ap.add_argument("--num_proc", type=int, default=4)
    args = ap.parse_args()

    ROOT = Path.cwd().resolve().parent
    paths = get_paths(ROOT)

    # 1) Load + standardize
    df_raw = load_raw_imdb_spoiler_json(paths.data_raw)
    df_all, _ = prepare_reviews_dataframe(df_raw)
    print(f"Total reviews: {len(df_all)}")

    # 2) Grouped split by movie_id
    cfg = SplitConfig(train_size=args.train_frac, val_size=args.val_frac, test_size=args.test_frac, seed=args.seed)
    train_df, val_df, test_df = split_by_movie_id(df_all, cfg=cfg)
    print(f"Split sizes -> train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    # 3) Optional subsample (per debug/iterazioni)
    if args.subsample_total and args.subsample_total > 0:
        scfg = SubsampleConfig(
            total_reviews=int(args.subsample_total),
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.seed,
            label_col="label",
        )
        train_df, val_df, test_df = subsample_splits_to_total(train_df, val_df, test_df, scfg)
        print(f"After subsample -> train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    # 4) Save raw splits (text+label) (opzionale ma utile)
    raw_dir = paths.data_processed / "raw_splits"
    raw_dir.mkdir(parents=True, exist_ok=True)
    p_train_raw = raw_dir / "train_raw.parquet"
    p_val_raw = raw_dir / "val_raw.parquet"
    p_test_raw = raw_dir / "test_raw.parquet"
    train_df.to_parquet(p_train_raw, index=False)
    val_df.to_parquet(p_val_raw, index=False)
    test_df.to_parquet(p_test_raw, index=False)

    # 5) Tokenize head+tail via HF datasets
    tok = build_teacher_tokenizer(args.model)

    def tokenize_file(p: Path, out_path: Path) -> None:
        ds = load_dataset("parquet", data_files=str(p))["train"]
        # map head+tail
        ds = ds.map(
            lambda ex: head_tail_tokenize_batch(
                ex,
                tokenizer=tok,
                max_length=args.max_length,
                head_len=args.head_len,
                text_col="text",
                add_token_type_ids=False,
            ),
            batched=True,
            num_proc=args.num_proc,
            desc=f"Head+Tail tokenization: {p.name}",
        )
        # labels -> labels (float)
        if "label" in ds.column_names and "labels" not in ds.column_names:
            ds = ds.rename_column("label", "labels")
        ds = ds.map(lambda ex: {"labels": float(ex["labels"])})
        # drop raw text/movie_id/review_id per alleggerire
        keep = [c for c in ["input_ids", "attention_mask", "token_type_ids", "labels"] if c in ds.column_names]
        ds = ds.select_columns(keep)
        ds.to_parquet(str(out_path))

    out_dir = paths.data_processed / "tokenized"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenize_file(p_train_raw, out_dir / "train_headtail.parquet")
    tokenize_file(p_val_raw, out_dir / "val_headtail.parquet")
    tokenize_file(p_test_raw, out_dir / "test_headtail.parquet")

    print("Done. Tokenized files written to:", out_dir)


if __name__ == "__main__":
    main()
