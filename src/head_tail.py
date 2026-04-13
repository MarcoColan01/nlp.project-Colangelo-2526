from __future__ import annotations
import pandas as pd
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from tqdm.auto import tqdm 

def apply_head_tail_truncation(
        df: pd.DataFrame,
        tokenizer_name: str,
        max_length: int = 512,
        head_len: int = 128,
        text_col: str = "text"
) -> pd.DataFrame:

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    
    tail_len = max_length - head_len - 2

    if tail_len <= 0:
        raise ValueError(f"Head len ({head_len}) is too big for the chosen max_length ({max_length}).")
    
    print(f"Strategy: [CLS] + Head({head_len}) + Tail({tail_len}) + [SEP] = {max_length}")

    texts = df[text_col].astype(str).tolist()
    
    encodings = tokenizer(
        texts,
        add_special_tokens=False, 
        truncation=False,
        padding=False,
        verbose=False
    )

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    
    if cls_id is None or sep_id is None:
        raise ValueError("Il tokenizer deve avere cls_token_id e sep_token_id definiti.")

    input_ids_list = []
    attention_mask_list = []

    for ids in tqdm(encodings["input_ids"], desc="Applying Head+Tail"):
        curr_len = len(ids)
        allowed_body_len = max_length - 2

        if curr_len <= allowed_body_len:
            final_body = ids
        else:
            head_part = ids[:head_len]
            tail_part = ids[-tail_len:]
            final_body = head_part + tail_part

        final_ids = [cls_id] + final_body + [sep_id]
        
        mask = [1] * len(final_ids)

        input_ids_list.append(final_ids)
        attention_mask_list.append(mask)
    
    out_df = df.copy()
    out_df["input_ids"] = input_ids_list
    out_df["attention_mask"] = attention_mask_list
    
    
    out_df["token_type_ids"] = [[0] * len(x) for x in input_ids_list]

    return out_df