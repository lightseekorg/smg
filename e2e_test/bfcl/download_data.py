#!/usr/bin/env python3
"""Download BFCL v3 open-source test data from HuggingFace.

Usage:
    python e2e_test/bfcl/download_data.py
"""

from __future__ import annotations

import os
import shutil
import urllib.request
from pathlib import Path

HF_BASE = (
    "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main"
)

DATA_DIR = Path(__file__).parent / "data"
TIMEOUT_SECONDS = 60

FILES = [
    "BFCL_v3_simple.json",
    "BFCL_v3_multiple.json",
    "BFCL_v3_parallel.json",
    "BFCL_v3_parallel_multiple.json",
    "BFCL_v3_irrelevance.json",
]

ANSWER_FILES = [
    ("possible_answer/BFCL_v3_simple.json", "BFCL_v3_simple_answer.json"),
    ("possible_answer/BFCL_v3_multiple.json", "BFCL_v3_multiple_answer.json"),
    ("possible_answer/BFCL_v3_parallel.json", "BFCL_v3_parallel_answer.json"),
    (
        "possible_answer/BFCL_v3_parallel_multiple.json",
        "BFCL_v3_parallel_multiple_answer.json",
    ),
]


def _count_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _download_one(remote_path: str, local_name: str, *, force: bool = False) -> None:
    url = f"{HF_BASE}/{remote_path}"
    dest = DATA_DIR / local_name
    if dest.exists() and not force:
        print(f"Skipping {local_name} (already downloaded)")
        print(f"  -> {dest.name} ({_count_lines(dest)} entries)")
        return

    print(f"Downloading {local_name}...")
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as resp, tmp.open("wb") as out:
            shutil.copyfileobj(resp, out)
            out.flush()
            os.fsync(out.fileno())
        tmp.replace(dest)
    finally:
        if tmp.exists():
            tmp.unlink()

    lines = _count_lines(dest)
    print(f"  → {dest.name} ({lines} entries)")


def download(*, force_redownload: bool = False) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for fname in FILES:
        _download_one(fname, fname, force=force_redownload)

    for remote_path, local_name in ANSWER_FILES:
        _download_one(remote_path, local_name, force=force_redownload)

    print("\nDone. All BFCL v3 data downloaded.")


if __name__ == "__main__":
    download()
