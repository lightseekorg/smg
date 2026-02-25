"""BFCL test evaluator and per-test log writer.

Validates model tool call output against BFCL structured ground truth
and writes detailed JSON logs for every test case (pass or fail)

Log directory layout:
    bfcl_logs/
    └── <YYYY-MM-DDTHH-MM-SS>/
        ├── summary.json
        ├── simple/
        │   ├── simple_0_PASS.json
        │   └── simple_1_FAIL.json
        ├── multiple/
        ├── parallel/
        └── irrelevance/
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BFCL_LOGS_DIR = Path(__file__).parent.parent / "bfcl_logs"


def get_run_dir() -> Path:
    """Create and return a timestamped run directory for this test session."""
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = BFCL_LOGS_DIR / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _call_matches(actual: dict[str, Any], gt_entry: dict[str, Any]) -> bool:
    """Check whether a single actual tool call satisfies a ground truth entry."""
    expected_name = next(iter(gt_entry.keys()), None)
    if actual.get("name") != expected_name:
        return False
    expected_args = gt_entry.get(expected_name, {})
    for param_name, possible_values in expected_args.items():
        actual_val = actual.get("arguments", {}).get(param_name)
        if not isinstance(possible_values, list):
            possible_values = [possible_values]
        if not any(_values_match(actual_val, pv) for pv in possible_values):
            return False
    return True


def _match_tool_calls(
    actual_tool_calls: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
) -> list[int | None]:
    """Return a maximum matching from actual calls to ground-truth entries."""
    candidate_gt_indices = [
        [gt_idx for gt_idx, gt_entry in enumerate(ground_truth) if _call_matches(actual, gt_entry)]
        for actual in actual_tool_calls
    ]
    actual_for_gt: list[int | None] = [None] * len(ground_truth)

    def _assign(actual_idx: int, seen_gt_indices: set[int]) -> bool:
        for gt_idx in candidate_gt_indices[actual_idx]:
            if gt_idx in seen_gt_indices:
                continue
            seen_gt_indices.add(gt_idx)
            prev_actual_idx = actual_for_gt[gt_idx]
            if prev_actual_idx is None or _assign(prev_actual_idx, seen_gt_indices):
                actual_for_gt[gt_idx] = actual_idx
                return True
        return False

    for actual_idx in sorted(
        range(len(actual_tool_calls)),
        key=lambda idx: len(candidate_gt_indices[idx]),
    ):
        _assign(actual_idx, set())

    gt_for_actual: list[int | None] = [None] * len(actual_tool_calls)
    for gt_idx, actual_idx in enumerate(actual_for_gt):
        if actual_idx is not None:
            gt_for_actual[actual_idx] = gt_idx
    return gt_for_actual


def evaluate_tool_calls(
    actual_tool_calls: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    *,
    category: str = "",
) -> tuple[bool, list[str]]:
    """Compare actual tool calls against BFCL structured ground truth.

    For irrelevance tests, expects zero tool calls.
    For all others, validates function names and arguments against
    the ground truth's list-of-possible-values format.

    Returns (passed, list_of_error_messages).
    """
    if category == "irrelevance":
        if actual_tool_calls:
            return False, [f"Irrelevance test: expected 0 tool calls, got {len(actual_tool_calls)}"]
        return True, []

    if not ground_truth:
        if not actual_tool_calls:
            return True, []
        return False, ["No ground truth available but model produced tool calls"]

    errors: list[str] = []

    if len(actual_tool_calls) != len(ground_truth):
        errors.append(f"Expected {len(ground_truth)} tool call(s), got {len(actual_tool_calls)}")

    gt_for_actual = _match_tool_calls(actual_tool_calls, ground_truth)
    matched_gt_indices = {gt_idx for gt_idx in gt_for_actual if gt_idx is not None}

    for actual, gt_idx in zip(actual_tool_calls, gt_for_actual):
        if gt_idx is None:
            errors.append(f"Unexpected tool call: '{actual.get('name', '?')}'")

    for gt_idx, gt_entry in enumerate(ground_truth):
        if gt_idx not in matched_gt_indices:
            gt_name = next(iter(gt_entry.keys()), "?")
            errors.append(f"Unmatched expected call: '{gt_name}'")

    return len(errors) == 0, errors


def save_test_log(
    run_dir: Path,
    *,
    test_id: str,
    category: str,
    model: str,
    parser: str,
    backend: str,
    request_payload: dict[str, Any],
    response_payload: dict[str, Any] | None,
    ground_truth: list[Any],
    actual_tool_calls: list[dict[str, Any]],
    passed: bool,
    errors: list[str],
    latency_ms: float,
) -> Path:
    """Write a detailed JSON log file for a single BFCL test case."""
    cat_dir = run_dir / category
    cat_dir.mkdir(parents=True, exist_ok=True)

    safe_id = test_id.replace("/", "_").replace(" ", "_")
    status = "PASS" if passed else "FAIL"
    filename = f"{safe_id}_{status}.json"

    log_entry = {
        "test_id": test_id,
        "category": category,
        "model": model,
        "parser": parser,
        "backend": backend,
        "timestamp": datetime.now(UTC).isoformat(),
        "passed": passed,
        "request": request_payload,
        "response": response_payload,
        "ground_truth": ground_truth,
        "actual_tool_calls": actual_tool_calls,
        "errors": errors,
        "latency_ms": round(latency_ms, 2),
    }

    out_path = cat_dir / filename
    out_path.write_text(json.dumps(log_entry, indent=2, default=str))
    return out_path


def save_summary(run_dir: Path, results: list[dict[str, Any]]) -> dict[str, Any]:
    """Write a rich summary.json with per-category stats, failure diagnostics,
    and latency distribution — everything you need at a glance."""
    by_cat: dict[str, dict[str, Any]] = {}
    all_latencies: list[float] = []
    failures: list[dict[str, Any]] = []

    for r in results:
        cat = r["category"]
        if cat not in by_cat:
            by_cat[cat] = {"total": 0, "passed": 0, "failed": 0, "latencies_ms": []}
        by_cat[cat]["total"] += 1
        lat = r.get("latency_ms", 0.0)
        by_cat[cat]["latencies_ms"].append(lat)
        all_latencies.append(lat)

        if r["passed"]:
            by_cat[cat]["passed"] += 1
        else:
            by_cat[cat]["failed"] += 1
            failures.append(
                {
                    "test_id": r.get("test_id", "?"),
                    "category": cat,
                    "errors": r.get("errors", []),
                    "latency_ms": round(lat, 1),
                    "finish_reason": r.get("finish_reason"),
                    "completion_tokens": r.get("completion_tokens"),
                    "had_reasoning": r.get("had_reasoning", False),
                    "log_file": r.get("log_file", ""),
                }
            )

    for cat_stats in by_cat.values():
        lats = cat_stats.pop("latencies_ms")
        if lats:
            lats_sorted = sorted(lats)
            n = len(lats_sorted)
            cat_stats["latency_ms"] = {
                "min": round(lats_sorted[0], 1),
                "median": round(lats_sorted[n // 2], 1),
                "p95": round(lats_sorted[min(int(n * 0.95), n - 1)], 1),
                "max": round(lats_sorted[-1], 1),
                "mean": round(sum(lats) / n, 1),
            }

    total = sum(c["total"] for c in by_cat.values())
    passed = sum(c["passed"] for c in by_cat.values())

    latency_summary = {}
    if all_latencies:
        s = sorted(all_latencies)
        n = len(s)
        latency_summary = {
            "min": round(s[0], 1),
            "median": round(s[n // 2], 1),
            "p95": round(s[min(int(n * 0.95), n - 1)], 1),
            "max": round(s[-1], 1),
            "mean": round(sum(s) / n, 1),
        }

    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "accuracy_pct": round(passed / total * 100, 2) if total else 0.0,
        "latency_ms": latency_summary,
        "by_category": by_cat,
        "failures": failures,
    }

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _values_match(actual: Any, expected: Any) -> bool:
    """Flexible comparison handling type coercion and empty-string-means-absent."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        if len(expected) != len(actual):
            return False
        return all(k in actual and _values_match(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return False
        return all(_values_match(a, e) for a, e in zip(actual, expected))
    if actual == expected:
        if isinstance(actual, bool) != isinstance(expected, bool):
            return False
        return True
    if expected == "" and actual is None:
        return True
    try:
        if not isinstance(actual, bool) and not isinstance(expected, bool):
            if float(actual) == float(expected):
                return True
    except (TypeError, ValueError):
        pass
    if isinstance(expected, str) and isinstance(actual, str):
        if actual.strip().lower() == expected.strip().lower():
            return True
    return False
