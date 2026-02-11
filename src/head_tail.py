from __future__ import annotations
import pandas as pd
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from tqdm.auto import tqdm 

'''
Tokenizza il testo. Se supera max_length, costruisce un input concatenando:
    [CLS] + Head (primi N token) + Tail (ultimi M token) + [SEP]
'''
def apply_head_tail_truncation(
        df: pd.DataFrame,
        tokenizer_name: str,
        max_length: int = 512,
        head_len: int = 128,
        text_col: str = "text"
) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    tail_len = max_length-head_len-2

    if tail_len <= 0:
        raise ValueError("Head len is too big for the choosen max_length.")
    
    print(f"Head-tail strategy: Head={head_len}, Tail={tail_len}, total= {max_length}")

    input_ids_list=[]
    attention_mask_list = []

    texts = df[text_col].astype(str).tolist()

    encodings = tokenizer(
        texts,
        add_special_tokens = True,
        truncation=False,
        padding=False,
        verbose=True
    )

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    for ids in tqdm(encodings["input_ids"], desc="Applying Head+Tail"):
        curr_len = len(ids)

        #Case 1: review is in max_length -> Keep all
        if curr_len <= max_length:
            final_ids = ids
        #Case 2: review is too long -> Head+Tail
        else:
            #Head: [CLS] + next tokens
            head_part = ids[:head_len+1]
            #Tail: last tail_len tokens before final SEP + [SEP]
            tail_part = ids[-(tail_len+1):]

            final_ids = head_part + tail_part

            if final_ids[0] != cls_id: final_ids[0] = cls_id
            if final_ids[-1] != sep_id: final_ids[-1] = sep_id
        
        final_ids = final_ids[:max_length]
        mask = [1] * len(final_ids)

        input_ids_list.append(final_ids)
        attention_mask_list.append(mask)
    
    out_df = df.copy()
    out_df["input_ids"] = input_ids_list
    out_df["attention_mask"] = attention_mask_list
    out_df["token_type_ids"] = [[0]*len(x) for x in input_ids_list]

    return out_df