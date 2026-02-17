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
    """
    Applica la strategia Head+Tail a un DataFrame pandas.
    
    Logica:
    1. Tokenizza il testo SENZA token speciali ([CLS], [SEP]).
    2. Se len > (max_length - 2): Prende Head + Tail.
    3. Aggiunge manualmente [CLS] all'inizio e [SEP] alla fine.
    
    Returns:
        DataFrame con colonne aggiunte 'input_ids', 'attention_mask'.
    """
    
    # 1. Configurazione Tokenizer
    # Importante: use_fast=True per velocità
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    
    # Calcolo dimensione Tail (sottraendo 2 per CLS e SEP)
    tail_len = max_length - head_len - 2

    if tail_len <= 0:
        raise ValueError(f"Head len ({head_len}) is too big for the chosen max_length ({max_length}).")
    
    print(f"Strategy: [CLS] + Head({head_len}) + Tail({tail_len}) + [SEP] = {max_length}")

    # 2. Tokenizzazione "pura" (senza CLS/SEP)
    texts = df[text_col].astype(str).tolist()
    
    # verbose=False per ridurre il rumore, usiamo tqdm dopo
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

    # 3. Applicazione Logica Head+Tail
    for ids in tqdm(encodings["input_ids"], desc="Applying Head+Tail"):
        curr_len = len(ids)
        allowed_body_len = max_length - 2

        if curr_len <= allowed_body_len:
            # Caso 1: Il testo ci sta tutto
            final_body = ids
        else:
            # Caso 2: Head + Tail
            head_part = ids[:head_len]
            tail_part = ids[-tail_len:]
            final_body = head_part + tail_part

        # Costruzione finale
        final_ids = [cls_id] + final_body + [sep_id]
        
        # Maschera (1 per i token reali)
        mask = [1] * len(final_ids)

        input_ids_list.append(final_ids)
        attention_mask_list.append(mask)
    
    # 4. Creazione Output
    out_df = df.copy()
    out_df["input_ids"] = input_ids_list
    out_df["attention_mask"] = attention_mask_list
    
    # Per BERT standard, token_type_ids sono tutti 0 per singola frase
    # Lo aggiungiamo per compatibilità completa
    out_df["token_type_ids"] = [[0] * len(x) for x in input_ids_list]

    return out_df