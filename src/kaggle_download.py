from __future__ import annotations
import argparse
import os
import subprocess
import zipfile
from pathlib import Path
from typing import Optional

from project_paths import get_paths


def _find_kaggle_json(
    project_root: Path,
    raw_dir: Path,
    explicit_path: Optional[Path] = None,
) -> Optional[Path]:
    candidates = []
    if explicit_path is not None:
        candidates.append(explicit_path)

    candidates.extend([
        project_root / "kaggle.json",
        raw_dir / "kaggle.json",
        Path.home() / ".kaggle" / "kaggle.json",
    ])

    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    return None


def _prepare_kaggle_env(kaggle_json_path: Optional[Path]) -> dict:
    env = os.environ.copy()
    if kaggle_json_path is None:
        return env

    env["KAGGLE_CONFIG_DIR"] = str(kaggle_json_path.parent.resolve())

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
    kaggle_json_path: Optional[Path] = None,
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)

    already = (
        list(raw_dir.glob("*.json"))
        + list(raw_dir.glob("*.jsonl"))
        + list(raw_dir.glob("*.jsonl.gz"))
    )
    if already and not force:
        print(f"Dataset already found in {raw_dir}. Use --force to re-download.")
        return

    kaggle_json = _find_kaggle_json(
        project_root=project_root,
        raw_dir=raw_dir,
        explicit_path=kaggle_json_path,
    )

    if kaggle_json is None:
        raise FileNotFoundError(
            "No kaggle.json found. Either place it in ~/.kaggle/kaggle.json "
            "or pass --kaggle-json /path/to/kaggle.json."
        )

    env = _prepare_kaggle_env(kaggle_json)

    cmd = [
        "kaggle", "datasets", "download",
        "-d", dataset_slug,
        "-p", str(raw_dir),
        "--unzip",
    ]

    try:
        subprocess.run(cmd, check=True, env=env)
    except FileNotFoundError as e:
        raise RuntimeError(
            "'kaggle' command not found. Install Kaggle CLI with: pip install kaggle"
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "Kaggle download failed. Check dataset slug, kaggle.json, and permissions."
        ) from e

    _unzip_all(raw_dir)
    print(f"Download completed. Files available in: {raw_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download the IMDb Spoiler Dataset from Kaggle into data/raw."
    )
    parser.add_argument(
        "--dataset-slug",
        required=True,
        help="Kaggle dataset slug in the form owner/dataset-name",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Optional custom raw data directory. Defaults to data/raw.",
    )
    parser.add_argument(
        "--kaggle-json",
        type=Path,
        default=None,
        help="Optional path to kaggle.json. If omitted, ~/.kaggle/kaggle.json is used if available.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if JSON files already exist.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    paths = get_paths(Path(__file__).resolve())
    raw_dir = args.raw_dir or paths.data_raw

    download_kaggle_dataset_once(
        dataset_slug=args.dataset_slug,
        raw_dir=raw_dir,
        project_root=paths.root,
        force=args.force,
        kaggle_json_path=args.kaggle_json,
    )


if __name__ == "__main__":
    main()