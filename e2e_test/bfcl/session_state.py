"""Thread-safe session-level result collector for BFCL tests.

Isolated here so both test_bfcl.py and fixtures/hooks.py import the same
module object — giving them a shared view of accumulated results, and routing
session lifecycle.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from .evaluator import BFCLEvaluator

logger = logging.getLogger(__name__)

_results_lock = threading.Lock()
_all_results: list[dict[str, Any]] = []
_evaluator = BFCLEvaluator()


def get_evaluator() -> BFCLEvaluator:
    """Return the shared session evaluator instance."""
    return _evaluator


def append_result(result: dict[str, Any]) -> None:
    """Record a single BFCL case result for session summary generation."""
    with _results_lock:
        _all_results.append(result)


def get_or_create_run_dir() -> Path:
    """Return the session's log directory, creating it exactly once."""
    with _results_lock:
        return _evaluator.get_run_dir()


def write_summary_if_needed() -> None:
    """Write summary.json — called from fixtures/hooks.py at session end."""
    with _results_lock:
        results = list(_all_results)
        run_dir = _evaluator.get_existing_run_dir()
    if not results or run_dir is None:
        return
    summary = _evaluator.save_summary(run_dir, results)
    logger.info(
        "BFCL summary: %d/%d passed (%.1f%%) — %s/summary.json",
        summary["passed"],
        summary["total"],
        summary["accuracy_pct"],
        run_dir,
    )
