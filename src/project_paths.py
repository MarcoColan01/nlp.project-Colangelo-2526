from __future__ import annotations 
from dataclasses import dataclass 
from pathlib import Path

@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    src: Path 
    notebooks: Path 
    data_processed: Path 
    data_raw: Path
    checkpoints: Path 

def get_project_root(start: Path | None = None) -> Path:
    p = (start or Path.cwd()).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "src").exists() and (parent / "data").exists():
            return parent
    return p

def get_paths(start: Path | None = None, create: bool = True) -> ProjectPaths:
    root = get_project_root(start)
    paths = ProjectPaths(
        root=root,
        src=root / "src",
        notebooks=root / "notebooks",
        data_raw=root / "data" / "raw",
        data_processed=root / "data" / "processed",
        checkpoints=root / "checkpoints",
    )
    if create:
        paths.data_raw.mkdir(parents=True, exist_ok=True)
        paths.data_processed.mkdir(parents=True, exist_ok=True)
        paths.checkpoints.mkdir(parents=True, exist_ok=True)
    return paths