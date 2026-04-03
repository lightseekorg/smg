"""BFCL v3 data loader.

Loads open-source BFCL v3 test data from the Gorilla project
(gorilla-llm/Berkeley-Function-Calling-Leaderboard on HuggingFace)
and converts it to OpenAI-compatible tool calling format.

"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent / "data"

BFCL_CATEGORIES = [
    "simple",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
]

_FILE_MAP = {
    "simple": "BFCL_v3_simple.json",
    "multiple": "BFCL_v3_multiple.json",
    "parallel": "BFCL_v3_parallel.json",
    "parallel_multiple": "BFCL_v3_parallel_multiple.json",
    "irrelevance": "BFCL_v3_irrelevance.json",
}

_ANSWER_FILE_MAP = {
    "simple": "BFCL_v3_simple_answer.json",
    "multiple": "BFCL_v3_multiple_answer.json",
    "parallel": "BFCL_v3_parallel_answer.json",
    "parallel_multiple": "BFCL_v3_parallel_multiple_answer.json",
}

DOWNLOAD_HINT = "Run: python e2e_test/bfcl/download_data.py"


class MissingBFCLAnswerFileError(FileNotFoundError):
    """Raised when BFCL question data exists but the mapped answer file is missing."""


@dataclass
class BFCLCase:
    """A single BFCL test case."""

    id: str
    question: list[dict[str, Any]]
    function: list[dict[str, Any]]
    ground_truth: list[dict[str, Any]] = field(default_factory=list)
    category: str = ""


def load_bfcl_category(
    category: str,
    *,
    limit: int | None = None,
) -> list[BFCLCase]:
    """Load BFCL test cases for a category, merged with ground truth answers."""
    if category not in _FILE_MAP:
        raise ValueError(f"Unknown category: {category}. Choose from {BFCL_CATEGORIES}")

    question_path = DATA_DIR / _FILE_MAP[category]
    if not question_path.exists():
        raise FileNotFoundError(f"BFCL data not found: {question_path}. {DOWNLOAD_HINT}")

    questions_by_id: dict[str, dict] = {}
    with question_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            questions_by_id[entry["id"]] = entry

    answer_filename = _ANSWER_FILE_MAP.get(category)
    answers_by_id: dict[str, list] = {}
    if answer_filename:
        answer_path = DATA_DIR / answer_filename
        if not answer_path.exists():
            raise MissingBFCLAnswerFileError(
                f"BFCL answer file not found: {answer_path}. {DOWNLOAD_HINT}"
            )
        with answer_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                answers_by_id[entry["id"]] = entry.get("ground_truth", [])

    results: list[BFCLCase] = []
    for test_id, entry in questions_by_id.items():
        if answer_filename and test_id not in answers_by_id:
            raise MissingBFCLAnswerFileError(
                f"BFCL ground truth missing for category={category!r}, "
                f"id={test_id!r} in {answer_filename}"
            )
        raw_question = entry.get("question", [])
        messages = (
            raw_question[0] if raw_question and isinstance(raw_question[0], list) else raw_question
        )

        ground_truth = answers_by_id.get(test_id, [])

        results.append(
            BFCLCase(
                id=test_id,
                question=messages,
                function=entry.get("function", []),
                ground_truth=ground_truth,
                category=category,
            )
        )

    if limit is not None:
        results = results[:limit]

    return results
