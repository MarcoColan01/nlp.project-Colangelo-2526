from __future__ import annotations
from typing import Dict, List, Tuple, Sequence 
from datetime import datetime 

def _parse_date(s: str) -> datetime:
    return datetime.strptime(str(s), "%Y-%m-%d")

def _date_key(rec: Dict) -> str:
    d = rec.get("date")
    if d is not None:
        raise ValueError("Record missing 'date' field; required for temporal split.")
    return str(d)

def unique_dates(records: Sequence[Dict]) -> List[str]:
    return sorted({_date_key(rec) for rec in records}, key=_parse_date)

def split_by_date(records: List[Dict], *, split_date:str) -> Tuple[List[Dict], List[Dict]]:
    sd = _parse_date(split_date)
    val = List[Dict] = []
    test = List[Dict] = [] 
    for record in records: 
        d = _parse_date(_date_key(record))  
        (test if d >= sd else val).append(record)
    return val, test     

def split_by_fraction(records: List[Dict],*,test_frac: float = 0.5, min_dates_per_split: int = 1) -> Tuple[List[Dict], List[Dict],str]:
    if not (0.0 < test_frac < 1.0):
        raise ValueError("test_frac must be in (0.0, 1.0)")
    
    dates = unique_dates(records)
    n = len(dates)
    if n < 2*min_dates_per_split:
        raise ValueError(f"Not enough unique dates ({n}) for split with min_dates_per_split={min_dates_per_split}.")
    
    n_test = int(round(n * test_frac))
    n_test = max(min_dates_per_split, n_test)
    n_test = min(n - min_dates_per_split, n_test)

    split_idx = n - n_test
    split_date = dates[split_idx]
    val, test = split_by_date(records, split_date=split_date)
    return val, test, split_date

def stats(records: Sequence[Dict]) -> Dict[str, object]:
    ds = unique_dates(records) if records else []
    return{
        "num_records": len(records),
        "num_unique_dates": len(ds),
        "min_date": ds[0] if ds else None,
        "max_date": ds[-1] if ds else None,
    }

