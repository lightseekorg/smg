"""BFCL v3 data loader.

Loads open-source BFCL v3 test data from the Gorilla project
(gorilla-llm/Berkeley-Function-Calling-Leaderboard on HuggingFace)
and converts it to OpenAI-compatible tool calling format.

"""

from __future__ import annotations

import json
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


class MissingBFCLAnswerFileError(FileNotFoundError):
    """Raised when BFCL question data exists but the mapped answer file is missing."""


def load_bfcl_category(
    category: str,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load BFCL test cases for a category, merged with ground truth answers.

    Returns a list of dicts, each with:
      - id: str
      - question: list[dict]         (unwrapped messages)
      - function: list[dict]         (original BFCL function defs)
      - ground_truth: list[dict]     (structured: [{func: {param: [vals]}}])
      - category: str
    """
    if category not in _FILE_MAP:
        raise ValueError(f"Unknown category: {category}. Choose from {BFCL_CATEGORIES}")

    question_path = DATA_DIR / _FILE_MAP[category]
    if not question_path.exists():
        raise FileNotFoundError(
            f"BFCL data not found: {question_path}. "
            "Run the download script or see e2e_test/bfcl/data/README.md"
        )

    questions_by_id: dict[str, dict] = {}
    with open(question_path, encoding="utf-8") as f:
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
                f"BFCL answer file not found: {answer_path}. "
                "Run: python e2e_test/bfcl/download_data.py"
            )
        with open(answer_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                answers_by_id[entry["id"]] = entry.get("ground_truth", [])

    results: list[dict[str, Any]] = []
    for test_id, entry in questions_by_id.items():
        raw_question = entry.get("question", [])
        messages = (
            raw_question[0] if raw_question and isinstance(raw_question[0], list) else raw_question
        )

        ground_truth = answers_by_id.get(test_id, [])

        results.append(
            {
                "id": test_id,
                "question": messages,
                "function": entry.get("function", []),
                "ground_truth": ground_truth,
                "category": category,
            }
        )

    if limit is not None:
        results = results[:limit]

    return results


def _fix_parameter_type(params: dict) -> dict:
    """Convert BFCL's non-standard types to valid JSON Schema types recursively."""
    result = dict(params)
    ptype = result.get("type")
    if ptype == "dict":
        result["type"] = "object"
    elif ptype == "float":
        result["type"] = "number"
    props = result.get("properties")
    if isinstance(props, dict):
        result["properties"] = {
            k: _fix_parameter_type(v) if isinstance(v, dict) else v for k, v in props.items()
        }
    items = result.get("items")
    if isinstance(items, dict):
        result["items"] = _fix_parameter_type(items)
    return result


def bfcl_to_openai_tools(bfcl_functions: list[dict]) -> list[dict]:
    """Convert BFCL function definitions to OpenAI tools format.

    Handles the BFCL-specific quirks:
      - parameters.type "dict" → "object"
      - parameters.type "float" → "number"
      - Wraps in {"type": "function", "function": ...}
    """
    tools = []
    for fn in bfcl_functions:
        fixed_fn = dict(fn)
        if "parameters" in fixed_fn:
            fixed_fn["parameters"] = _fix_parameter_type(fixed_fn["parameters"])
        tools.append({"type": "function", "function": fixed_fn})
    return tools
