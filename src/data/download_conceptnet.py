# src/data/download_conceptnet.py
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.request import urlopen

from utils import project_paths, ensure_dir, get_logger

# Official ConceptNet 5.7 assertions dump (gzipped, tab-separated)
# Source: ConceptNet downloads wiki + multiple downstream libs reference this exact URL. :contentReference[oaicite:1]{index=1}
CONCEPTNET_ASSERTIONS_URL = "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz"
DEFAULT_FILENAME = "conceptnet-assertions-5.7.0.csv.gz"


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download_conceptnet_assertions_once(
    out_dir: Path | None = None,
    filename: str = DEFAULT_FILENAME,
    url: str = CONCEPTNET_ASSERTIONS_URL,
) -> Path:
    """
    Downloads ConceptNet assertions dump into src/data/conceptnet/ (by default).

    "Once" policy:
      - If target file already exists, we DO NOT download again.
      - We raise a RuntimeError to force you to explicitly delete it if you want re-download.
    """
    logger = get_logger("conceptnet-download")
    paths = project_paths(create=True)

    out_dir = out_dir or (paths.src / "data" / "conceptnet")
    ensure_dir(out_dir)

    target = out_dir / filename
    if target.exists():
        raise RuntimeError(
            f"Dump already exists at: {target}\n"
            f"Delete it manually if you want to re-download."
        )

    logger.info(f"Downloading ConceptNet assertions to: {target}")
    logger.info(f"URL: {url}")

    # Stream download to disk
    with urlopen(url) as r:
        total = r.headers.get("Content-Length")
        total_bytes = int(total) if total is not None else None

        tmp = target.with_suffix(target.suffix + ".part")
        written = 0
        chunk = 1024 * 1024  # 1MB

        with tmp.open("wb") as f:
            while True:
                data = r.read(chunk)
                if not data:
                    break
                f.write(data)
                written += len(data)
                if total_bytes:
                    pct = (written / total_bytes) * 100
                    logger.info(f"Downloaded: {written/1e6:.1f}MB / {total_bytes/1e6:.1f}MB ({pct:.1f}%)")
                else:
                    logger.info(f"Downloaded: {written/1e6:.1f}MB")

        tmp.rename(target)

    logger.info("Download complete.")
    logger.info(f"File size: {target.stat().st_size/1e6:.1f}MB")
    logger.info(f"SHA256: {_sha256_file(target)}")

    return target


if __name__ == "__main__":
    # Running this directly will download the file once.
    download_conceptnet_assertions_once()
