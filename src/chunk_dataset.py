from __future__ import annotations

from typing import Dict, Any, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

class ChunkDataset(Dataset):
    def __init__(
            self,
            df_chunks: pd.DataFrame,
            label_col: str = "label",
            include_ids: bool = True,
    ):
        self.df = df_chunks.reset_index(drop= True)
        self.label_col = label_col
        self.include_ids =include_ids

        required = {"input_ids", "attention_mask", label_col}
        missing  = required-set(self.df.columns)
        if missing:
            raise KeyError(f"Missing columns in df_chunks: {missing}")
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        item = {
            "input_ids": row["input_ids"],
            "attention_mask": row["attention_mask"],
            "labels": int(row[self.label_col]),
        }
        if "token_type_ids" in self.df.columns:
            item["token_type_ids"] = row["token_type_ids"]

        if self.include_ids:
            item["review_id"]
            item["chunk_index"]  =int(row["chunk_index"])
        
        return item  