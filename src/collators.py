from __future__ import annotations
from transformers import DataCollatorWithPadding

def build_data_collator(tokenizer):
    return DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")