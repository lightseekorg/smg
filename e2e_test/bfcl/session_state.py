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

logger = logging.getLogger(__name__)

_results_lock = threading.Lock()
_all_results: list[dict[str, Any]] = []
_run_dir: Path | None = None


def append_result(result: dict[str, Any]) -> None:
    with _results_lock:
        _all_results.append(result)


def get_or_create_run_dir() -> Path:
    """Return the session's log directory, creating it exactly once."""
    global _run_dir
    with _results_lock:
        if _run_dir is None:
            from .evaluator import get_run_dir

            _run_dir = get_run_dir()
        return _run_dir


def write_summary_if_needed() -> None:
    """Write summary.json — called from fixtures/hooks.py at session end."""
    from .evaluator import save_summary

    with _results_lock:
        results = list(_all_results)
        run_dir = _run_dir
    if not results or run_dir is None:
        return
    summary = save_summary(run_dir, results)
    logger.info(
        "BFCL summary: %d/%d passed (%.1f%%) — %s/summary.json",
        summary["passed"],
        summary["total"],
        summary["accuracy_pct"],
        run_dir,
    )
