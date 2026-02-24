#!/usr/bin/env python3
"""Download BFCL v3 open-source test data from HuggingFace.

Usage:
    python e2e_test/bfcl/download_data.py
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

HF_BASE = (
    "https://huggingface.co/datasets/gorilla-llm/"
    "Berkeley-Function-Calling-Leaderboard/resolve/main"
)

DATA_DIR = Path(__file__).parent / "data"

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


def download() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for fname in FILES:
        url = f"{HF_BASE}/{fname}"
        dest = DATA_DIR / fname
        print(f"Downloading {fname}...")
        urllib.request.urlretrieve(url, dest)
        lines = sum(1 for _ in open(dest))
        print(f"  → {dest.name} ({lines} entries)")

    for remote_path, local_name in ANSWER_FILES:
        url = f"{HF_BASE}/{remote_path}"
        dest = DATA_DIR / local_name
        print(f"Downloading {local_name}...")
        urllib.request.urlretrieve(url, dest)
        lines = sum(1 for _ in open(dest))
        print(f"  → {dest.name} ({lines} entries)")

    print("\nDone. All BFCL v3 data downloaded.")


if __name__ == "__main__":
    download()
