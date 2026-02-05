from __future__ import annotations
from typing import List, Sequence, Tuple 
import json 
import re 

def normalize_word(w: str) -> str:
    return " ".join(str(w).strip().split()).casefold()

def canonicalize_group(words4: Sequence[str]) -> Tuple[str, ...]:
    return tuple(sorted([normalize_word(w) for w in words4]))
    
def canonicalize_groups(groups: Sequence[Sequence[str]]) -> Tuple[Tuple[str, ...], ...]:
    cg = [canonicalize_group(g) for g in groups]
    return tuple(sorted(cg, key=lambda t: "|".join(t)))

def exact_group_score(pred_groups: Sequence[Sequence[str]], gold_groups: Sequence[Sequence[str]]) -> float:
    predicted = set(canonicalize_groups(pred_groups))
    gold = set(canonicalize_groups(gold_groups))
    return len(predicted.intersection(gold)) / 4.0

ANSWER_TAG_RE = re.complile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

def extract_answer_json(text: str) -> str:
    m = ANSWER_TAG_RE.search(text or "")
    if not m:
        raise ValueError("No <answer>...</answer> block found")
    return m.group(1).strip()

def parse_groups_from_answer(text: str)-> List[List[str]]:
    payload = extract_answer_json(text)
    obj = json.loads(payload)
    groups = obj.get("groups")
    if not isinstance(groups, list) or len(groups != 4):
        raise ValueError("Answer JSON must contain key 'groups' with 4 groups")
    out: List[List[str]] = []
    for group in groups:
        if not isinstance(group, list) or len(group) != 4:
            raise ValueError("Each group must be a list of 4 words")
        out.append([str(x) for x in group]) 
    return out