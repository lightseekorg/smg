"""BFCL (Berkeley Function Calling Leaderboard) test infrastructure.

Provides data loading, evaluation, and per-test logging for open-source
BFCL v3 test cases run against the SMG gateway.
"""

from .converter import bfcl_to_openai_tools
from .evaluator import BFCLEvaluator, extract_tool_calls, log_file_for_summary
from .loader import (
    BFCL_CATEGORIES,
    BFCLCase,
    MissingBFCLAnswerFileError,
    load_bfcl_category,
)

__all__ = [
    "BFCL_CATEGORIES",
    "BFCLCase",
    "BFCLEvaluator",
    "MissingBFCLAnswerFileError",
    "bfcl_to_openai_tools",
    "extract_tool_calls",
    "load_bfcl_category",
    "log_file_for_summary",
]
