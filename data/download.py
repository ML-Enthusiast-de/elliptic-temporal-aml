#!/usr/bin/env python3
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import requests


BASE_URL = "https://data.pyg.org/datasets/elliptic"
FILES = [
    "elliptic_txs_features.csv",
    "elliptic_txs_edgelist.csv",
    "elliptic_txs_classes.csv",
]


def repo_root() -> Path:
    # src/data/download.py -> repo root is 2 parents up from src/
    return Path(__file__).resolve().parents[2]


def download_and_extract(url: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        zf.extractall(out_dir)


def main() -> None:
    raw_dir = repo_root() / "data" / "raw" / "elliptic"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for fname in FILES:
        url = f"{BASE_URL}/{fname}.zip"
        print(f"Downloading {fname} ...")
        download_and_extract(url, raw_dir)

    missing = [f for f in FILES if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing after download: {missing}")

    print("✅ Done. Files in:", raw_dir)


if __name__ == "__main__":
    main()
