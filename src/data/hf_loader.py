from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Any, Dict, List, Sequence, Union
from pathlib import Path
import json 
import numpy as np 

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None  # type: ignore

@dataclass(frozen = True)
class HFLoadConfig:
    dataset_id: str
    split: str = 'train'
    words_field: Optional[str] = None
    answers_field: Optional[str] = None
    date_field: Optional[str] = None
    id_field: Optional[str] = None
    num_permutations: int = 1
    seed: int = 1234

def _infer_field(example: Dict[str, Any], candidates: Sequence[str]) -> Optional[str]:
    keys = {k.lower(): k for k in example.keys()}
    for c in candidates:
        if c.lower() in keys:
            return keys[c.lower()]     
    return None

def _ensure_list_str(x: Any) -> List[str]: 
    if  x is None:
        return []
    if isinstance(x, (list, tuple, np.ndarray)):
        return [str(i) for i in x]
    if isinstance(x, str) and x.strip().startswith("["):
        try:
            arr = json.loads(x)
            if isinstance(arr, list):
                return [str(i) for i in arr]
        except Exception:
            pass
    return [str(x)]

def _normalize_record(rec: Dict[str, Any], *, cfg: HFLoaderConfig, idx: int) -> Dict[str, Any]:
    words_f = cfg.words_field or _infer_field(rec, ["words", "word_list", "puzzle_words"])
    answers_f = cfg.answers_field or _infer_field(rec, ["answers", "solution", "groups"])
    date_f = cfg.date_field or _infer_field(rec, ["date", "puzzle_date"])
    id_f = cfg.id_field or _infer_field(rec, ["puzzle_id", "id", "game_id"])

    if not words_f:
        raise KeyError(f"Could not infer words field from keys: {list(rec.keys())}")
    
    words16 = _ensure_list_str(rec[words_f])
    if len(words16) != 16:
        raise ValueError(f"Expected 16 words, got {len(words16)} at index {idx}")
    
    answers_out: List[Dict[str, Any]] = []
    if answers_f and rec.get(answers_f) is not None: 
        ans = rec[answers_f]
        if isinstance(ans, str):
            try:
                ans = json.loads(ans)
            except Exception:
                ans = None 
        if isinstance(ans, list):
            if all(isinstance(a, dict) for a in ans):
                for a in ans:
                    w = _ensure_list_str(a.get("words", a.get("items")))
                    desc = a.get("answerDescription", a.get("category", a.get("name", "")))
                    answers_out.append({"answerDescription": (str(desc) if desc is not None else ""), "words": w})
            elif all(isinstance(a, (list, tuple)) for a in ans):
                for a in ans:
                    answers_out.append({"answerDescription": "", "words": [str(x) for x in a]})
        elif isinstance(ans, dict):
            groups = ans.get("answers") or ans.get("groups")
            if isinstance(groups, list):
                for g in groups:
                    if isinstance(g, dict):
                        w = _ensure_list_str(g.get("words"))
                        desc = g.get("answerDescription", g.get("category", g.get("name", "")))
                        answers_out.append({"answerDescription": str(desc), "words": w})
                    elif isinstance(g, (list,tuple)):
                        answers_out.append({"answerDescription": "", "words": [str(x) for x in g]})
    if answers_out:
        if len(answers_out) != 4 or any(len(a.get("words", [])) != 4 for a in answers_out):
            raise ValueError(f"Answers not in 4x4 shape at index {idx}")
    
    date_val = str(rec[date_f]) if date_f and rec.get(date_f) is not None else None 
    rid = str(rec[id_f]) if id_f and rec.get(id_f) is not None else None 
    puzzle_id = rid if rid else f"hf_{idx}"

    return{
        "puzzle_id": puzzle_id,
        "date": date_val,
        "words": words16,
        "answers": answers_out,
        "metadata": {"source": "hf", "split": cfg.split},
    }

def load_hf_puzzles(cfg: HFLoaderConfig) -> List[Dict[str, Any]]:
    if load_dataset is None:
        raise ImportError("datasets library is not installed. Please install it to use HFLoader.")
    
    ds = load_dataset(cfg.dataset_id, split=cfg.split)
    puzzles_base = [_normalize_record(rec, cfg=cfg, idx=i) for i, rec in enumerate(ds)]

    if cfg.num_permutations <= 1:
        return puzzles_base
    
    rng = np.random.default_rng(cfg.seed)
    puzzles_augmented: List[Dict[str, Any]] = []
    for puzzle in puzzles_base:
        words = puzzle["words"]
        for k in range(cfg.num_permutations):
            idx = rng.permutation(16) 
            p2 = dict(puzzle) 
            p2["puzzle_id"] = f"{puzzle['puzzle_id']}_perm{k}" 
            p2["words"] = [words[i] for i in idx]
            p2["metadata"] = dict(puzzle.get("metadata", {})) 
            p2["metadata"].update({"perm_index": k, "seed": cfg.seed})
            puzzles_augmented.append(p2)
    return puzzles_augmented

def save_jsonl(records: Sequence[Dict[str, Any]], out_path: Union[str, Path]) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        try:
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
        except Exception:
            pass 
        if hasattr(o, "item"):
            try:
                return o.item()
            except Exception:
                pass
        return str(o) 
    
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, default=_default) + "\n")
            
