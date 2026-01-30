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