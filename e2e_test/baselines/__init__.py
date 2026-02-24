"""Baseline comparison infrastructure for benchmark regression detection.

Compares fresh benchmark summaries against stored baselines checked into
git, flagging accuracy regressions that exceed configurable tolerances.

Usage:
    from baselines.compare import baseline_path, load_baseline, save_baseline, compare_summary
"""

from .compare import (
    ComparisonResult,
    Regression,
    baseline_path,
    compare_summary,
    load_baseline,
    save_baseline,
)

__all__ = [
    "ComparisonResult",
    "Regression",
    "baseline_path",
    "compare_summary",
    "load_baseline",
    "save_baseline",
]
