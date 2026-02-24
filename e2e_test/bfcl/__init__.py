"""BFCL (Berkeley Function Calling Leaderboard) test infrastructure.

Provides data loading, evaluation, and per-test logging for open-source
BFCL v3 test cases run against the SMG gateway.
"""

from .evaluator import evaluate_tool_calls, get_run_dir, save_summary, save_test_log
from .loader import (
    BFCL_CATEGORIES,
    bfcl_to_openai_tools,
    load_bfcl_category,
)

__all__ = [
    "BFCL_CATEGORIES",
    "bfcl_to_openai_tools",
    "evaluate_tool_calls",
    "get_run_dir",
    "load_bfcl_category",
    "save_summary",
    "save_test_log",
]
