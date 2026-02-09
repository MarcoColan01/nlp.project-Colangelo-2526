from __future__ import annotations
import pandas as pd

def aggregate_max_pooling(
        df_chunks: pd.DataFrame,
        prob_col: str = "prob",
        review_id_col: str = "review_id",
) -> pd.DataFrame:
    required  ={review_id_col, prob_col}
    missing = required-set(df_chunks.columns)
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    
    out = df_chunks.groupby(review_id_col, as_index=False)[prob_col].max().rename(columns={prob_col: "prob_review"})
    return out 

def attach_review_labels(
        df_review_scores: pd.DataFrame,
        df_reviews: pd.DataFrame,
        review_id_col: str = "review_id",
        label_col: str = "label",
) -> pd.DataFrame:
    if review_id_col not in df_reviews.columns or label_col not in df_reviews.columns:
        raise KeyError("df_reviews must contain review_id and label columns.")
    
    merged = df_review_scores.merge(
        df_reviews[[review_id_col, label_col]],
        on=review_id_col,
        how="left",
        validate="one_to_one"
    )

    return merged