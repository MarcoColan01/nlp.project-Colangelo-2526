from __future__ import annotations
from typing import Dict, List, Union 
from pathlib import Path 
import json 

def read_jsonl(path: Union[str, Path]) -> List[Dict]:
    path = Path(path)
    out: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue 
            out.append(json.loads(line))
    return out

def write_jsonl(records: List[Dict], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")