from __future__ import annotations 
import os 
import subprocess 
import zipfile 
from pathlib import Path 
from typing import Optional 

def _find_kaggle_json(project_root: Path, raw_dir: Path) -> Optional[Path]:
    candidates =  [
        project_root / "kaggle.json",
        raw_dir / "kaggle.json",
        Path.home() / ".kaggle" / "kaggle.json",
    ]
    for candidate in candidates:
        if candidate.exists(): 
            return candidate
    return None 

def _prepare_kaggle_env(kaggle_json_path: Optional[Path]) -> dict:
    env = os.environ.copy()
    if kaggle_json_path is None:
        return env 
    config_dir = str(kaggle_json_path.parent.resolve())
    env["KAGGLE_CONFIG_DIR"] = config_dir

    try:
        os.chmod(kaggle_json_path, 0o600)
    except Exception:
        pass 
    
    return env

def _unzip_all(raw_dir: Path) -> None:
    for z in raw_dir.glob("*.zip"):
        with zipfile.ZipFile(z, "r") as zip_ref:
            zip_ref.extractall(raw_dir)

def download_kaggle_dataset_once(
    dataset_slug: str,
    raw_dir: Path,
    project_root: Path,
    force: bool = False,
) -> None: 
    raw_dir.mkdir(parents=True, exist_ok=True)

    already = list(raw_dir.glob("*.json")) + list(raw_dir.glob("*.jsonl")) + list(raw_dir.glob("*.jsonl.gz"))
    if already and not force:
        return

    kaggle_json = _find_kaggle_json(project_root=project_root, raw_dir=raw_dir)
    env = _prepare_kaggle_env(kaggle_json)

    cmd = ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(raw_dir), "--unzip"]

    try:
        subprocess.run(cmd, check=True, env=env)
    except FileNotFoundError as e:
        raise RuntimeError("'kaggle' comand not found. Install Kaggle CLI: pip install kaggle") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "Download Kaggle fallito. Controlla dataset_slug, kaggle.json e permessi."
        ) from e

    _unzip_all(raw_dir)