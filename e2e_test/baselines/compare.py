"""Baseline comparison engine for benchmark regression detection.

Compares a fresh benchmark summary against a stored baseline and flags
accuracy regressions that exceed configurable tolerances.

Design:
  - Accuracy is the primary regression criterion (correctness benchmarks).
  - Latency is hardware-dependent and NOT checked (use nightly perf for that).
  - Missing baselines are handled gracefully (warning, not failure).
  - Baseline files are deterministically formatted for clean git diffs.
  - Tolerances are configurable via env vars or function args.

Baseline file layout:
    e2e_test/baselines/
    └── <benchmark>/
        └── <model>_<backend>.json

Environment variables:
    BASELINE_ACCURACY_TOLERANCE  — max allowed accuracy drop in percentage points (default: 3.0)
    BFCL_UPDATE_BASELINE         — if set, save current results as the new baseline
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BASELINES_DIR = Path(__file__).resolve().parent

_DEFAULT_ACCURACY_TOLERANCE_PP = 3.0


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Regression:
    """A single metric that regressed beyond tolerance."""

    metric: str
    category: str
    baseline_value: float
    actual_value: float
    delta_pp: float
    tolerance_pp: float

    @property
    def description(self) -> str:
        return (
            f"{self.category} {self.metric}: "
            f"{self.baseline_value:.2f}% → {self.actual_value:.2f}% "
            f"(Δ{self.delta_pp:+.2f}pp, tolerance: ±{self.tolerance_pp:.1f}pp)"
        )


@dataclass
class ComparisonResult:
    """Full result of comparing a run against a baseline."""

    baseline_path: Path | None
    tolerance_pp: float
    regressions: list[Regression] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    skipped_categories: list[str] = field(default_factory=list)
    details: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return len(self.regressions) == 0

    def format_report(self) -> str:
        """Human-readable multi-line report for logs."""
        lines = [
            "=== Baseline Comparison ===",
            f"Baseline: {self.baseline_path}",
            f"Tolerance: ±{self.tolerance_pp:.1f} percentage points",
            "",
        ]

        for cat in sorted(self.details):
            d = self.details[cat]
            status = d["status"]
            marker = "✓" if status == "ok" else ("↑" if status == "improved" else "✗")
            label = cat.ljust(20)
            lines.append(
                f"  {marker} {label} "
                f"{d['baseline']:.2f}% → {d['actual']:.2f}% "
                f"(Δ{d['delta_pp']:+.2f}pp)"
            )

        lines.append("")
        if self.passed:
            lines.append("Result: PASSED")
            if self.improvements:
                lines.append("Improvements:")
                for imp in self.improvements:
                    lines.append(f"  + {imp}")
        else:
            lines.append(
                f"Result: FAILED ({len(self.regressions)} regression(s))"
            )
            for reg in self.regressions:
                lines.append(f"  ✗ {reg.description}")
            if self.improvements:
                lines.append("Improvements (does not offset regressions):")
                for imp in self.improvements:
                    lines.append(f"  + {imp}")

        if self.skipped_categories:
            lines.append(
                f"Skipped (not in current run): "
                f"{', '.join(self.skipped_categories)}"
            )

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serializable dict for comparison.json."""
        return {
            "baseline_path": str(self.baseline_path) if self.baseline_path else None,
            "tolerance_pp": self.tolerance_pp,
            "passed": self.passed,
            "regressions": [
                {
                    "metric": r.metric,
                    "category": r.category,
                    "baseline": r.baseline_value,
                    "actual": r.actual_value,
                    "delta_pp": round(r.delta_pp, 2),
                }
                for r in self.regressions
            ],
            "improvements": self.improvements,
            "skipped_categories": self.skipped_categories,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _normalize_model_name(model: str) -> str:
    """Qwen/Qwen2.5-7B-Instruct → qwen2.5-7b-instruct"""
    return model.rsplit("/", 1)[-1].lower()


def baseline_path(benchmark: str, model: str, backend: str) -> Path:
    """Deterministic path to a baseline file.

    Example:
        baseline_path("bfcl", "Qwen/Qwen2.5-7B-Instruct", "grpc")
        → <baselines_dir>/bfcl/qwen2.5-7b-instruct_grpc.json
    """
    name = f"{_normalize_model_name(model)}_{backend.lower()}"
    return BASELINES_DIR / benchmark / f"{name}.json"


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------


def load_baseline(path: Path) -> dict[str, Any] | None:
    """Load a baseline file. Returns None if missing or corrupt."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load baseline %s: %s", path, exc)
        return None


def save_baseline(
    path: Path,
    summary: dict[str, Any],
    *,
    model: str = "",
    backend: str = "",
) -> Path:
    """Write a benchmark summary as a new baseline file.

    Extracts only the stable metrics needed for comparison (no timestamps,
    no failure details, no latency). The file is deterministically formatted
    (sorted keys, 2-space indent, trailing newline) for clean git diffs.
    """
    by_category: dict[str, dict[str, Any]] = {}
    for cat, stats in sorted(summary.get("by_category", {}).items()):
        total = stats.get("total", 0)
        passed = stats.get("passed", 0)
        by_category[cat] = {
            "accuracy_pct": round(passed / total * 100, 2) if total else 0.0,
            "failed": stats.get("failed", 0),
            "passed": passed,
            "total": total,
        }

    baseline = {
        "accuracy_pct": summary.get("accuracy_pct", 0.0),
        "by_category": by_category,
        "failed": summary.get("failed", 0),
        "metadata": {
            "backend": backend,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
        },
        "passed": summary.get("passed", 0),
        "total": summary.get("total", 0),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
    logger.info("Baseline saved to %s", path)
    return path


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------


def _accuracy_from_stats(stats: dict[str, Any]) -> float | None:
    """Compute accuracy % from a stats dict (handles both explicit and derived)."""
    if "accuracy_pct" in stats:
        return float(stats["accuracy_pct"])
    total = stats.get("total", 0)
    passed = stats.get("passed", 0)
    if total > 0:
        return round(passed / total * 100, 2)
    return None


def compare_summary(
    summary: dict[str, Any],
    baseline: dict[str, Any],
    baseline_file: Path,
    *,
    accuracy_tolerance_pp: float | None = None,
) -> ComparisonResult:
    """Compare a benchmark summary against a stored baseline.

    Checks overall and per-category accuracy. A regression is flagged when
    accuracy drops by more than ``accuracy_tolerance_pp`` percentage points.

    Args:
        summary: Fresh benchmark summary (the dict returned by save_summary).
        baseline: Stored baseline (loaded by load_baseline).
        baseline_file: Path to the baseline file (for reporting).
        accuracy_tolerance_pp: Max allowed accuracy drop in percentage points.
            Falls back to BASELINE_ACCURACY_TOLERANCE env var, then 3.0.

    Returns:
        ComparisonResult with any regressions and improvements.
    """
    if accuracy_tolerance_pp is None:
        accuracy_tolerance_pp = float(
            os.environ.get(
                "BASELINE_ACCURACY_TOLERANCE",
                str(_DEFAULT_ACCURACY_TOLERANCE_PP),
            )
        )

    result = ComparisonResult(
        baseline_path=baseline_file,
        tolerance_pp=accuracy_tolerance_pp,
    )

    # --- overall accuracy ---
    actual_acc = _accuracy_from_stats(summary)
    baseline_acc = _accuracy_from_stats(baseline)

    if actual_acc is not None and baseline_acc is not None:
        delta = actual_acc - baseline_acc
        status = "ok"
        if delta < -accuracy_tolerance_pp:
            status = "regression"
            result.regressions.append(
                Regression(
                    metric="accuracy",
                    category="overall",
                    baseline_value=baseline_acc,
                    actual_value=actual_acc,
                    delta_pp=round(delta, 2),
                    tolerance_pp=accuracy_tolerance_pp,
                )
            )
        elif delta > accuracy_tolerance_pp:
            status = "improved"
            result.improvements.append(
                f"overall accuracy: {baseline_acc:.2f}% → {actual_acc:.2f}% "
                f"(+{delta:.2f}pp)"
            )
        result.details["overall"] = {
            "baseline": baseline_acc,
            "actual": actual_acc,
            "delta_pp": round(delta, 2),
            "status": status,
        }

    # --- per-category accuracy ---
    baseline_cats = baseline.get("by_category", {})
    summary_cats = summary.get("by_category", {})

    for cat in sorted(baseline_cats):
        if cat not in summary_cats:
            result.skipped_categories.append(cat)
            continue

        b_acc = _accuracy_from_stats(baseline_cats[cat])
        s_acc = _accuracy_from_stats(summary_cats[cat])
        if b_acc is None or s_acc is None:
            continue

        cat_delta = s_acc - b_acc
        status = "ok"

        if cat_delta < -accuracy_tolerance_pp:
            status = "regression"
            result.regressions.append(
                Regression(
                    metric="accuracy",
                    category=cat,
                    baseline_value=b_acc,
                    actual_value=s_acc,
                    delta_pp=round(cat_delta, 2),
                    tolerance_pp=accuracy_tolerance_pp,
                )
            )
        elif cat_delta > accuracy_tolerance_pp:
            status = "improved"
            result.improvements.append(
                f"{cat} accuracy: {b_acc:.2f}% → {s_acc:.2f}% "
                f"(+{cat_delta:.2f}pp)"
            )

        result.details[cat] = {
            "baseline": b_acc,
            "actual": s_acc,
            "delta_pp": round(cat_delta, 2),
            "status": status,
        }

    # Warn about total mismatch (e.g. running with BFCL_LIMIT)
    b_total = baseline.get("total", 0)
    s_total = summary.get("total", 0)
    if b_total > 0 and s_total > 0 and b_total != s_total:
        logger.warning(
            "Run total (%d) differs from baseline total (%d) — "
            "accuracy comparison may be less meaningful",
            s_total,
            b_total,
        )

    return result
