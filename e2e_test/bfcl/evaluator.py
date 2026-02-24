"""BFCL test evaluator and per-test log writer.

Validates model tool call output against BFCL structured ground truth
and writes detailed JSON logs for every test case (pass or fail),
mirroring the fc-dash .test-logs/ pattern.

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BFCL_LOGS_DIR = Path(__file__).parent.parent / "bfcl_logs"


def get_run_dir() -> Path:
    """Create and return a timestamped run directory for this test session."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = BFCL_LOGS_DIR / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


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
        if len(actual_tool_calls) > 0:
            return False, [
                f"Irrelevance test: expected 0 tool calls, got {len(actual_tool_calls)}"
            ]
        return True, []

    if not ground_truth:
        if len(actual_tool_calls) == 0:
            return True, []
        return False, ["No ground truth available but model produced tool calls"]

    errors: list[str] = []

    if len(actual_tool_calls) != len(ground_truth):
        errors.append(
            f"Expected {len(ground_truth)} tool call(s), got {len(actual_tool_calls)}"
        )

    check_count = min(len(actual_tool_calls), len(ground_truth))

    for i in range(check_count):
        gt_entry = ground_truth[i]
        actual = actual_tool_calls[i]

        expected_name = next(iter(gt_entry.keys()), None)
        expected_args = gt_entry.get(expected_name, {}) if expected_name else {}

        if actual["name"] != expected_name:
            errors.append(
                f"[{i}] Expected function '{expected_name}', got '{actual['name']}'"
            )
            continue

        for param_name, possible_values in expected_args.items():
            actual_val = actual["arguments"].get(param_name)

            if not isinstance(possible_values, list):
                possible_values = [possible_values]

            all_accepted = list(possible_values)
            if "" in all_accepted:
                all_accepted.append(None)

            matched = any(_values_match(actual_val, pv) for pv in all_accepted)
            if not matched:
                errors.append(
                    f"[{i}] '{expected_name}' arg '{param_name}': "
                    f"expected one of {possible_values!r}, got {actual_val!r}"
                )

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
        "timestamp": datetime.now(timezone.utc).isoformat(),
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
            failures.append({
                "test_id": r.get("test_id", "?"),
                "category": cat,
                "errors": r.get("errors", []),
                "latency_ms": round(lat, 1),
                "finish_reason": r.get("finish_reason"),
                "completion_tokens": r.get("completion_tokens"),
                "had_reasoning": r.get("had_reasoning", False),
                "log_file": r.get("log_file", ""),
            })

    for cat_stats in by_cat.values():
        lats = cat_stats.pop("latencies_ms")
        if lats:
            lats_sorted = sorted(lats)
            cat_stats["latency_ms"] = {
                "min": round(lats_sorted[0], 1),
                "median": round(lats_sorted[len(lats_sorted) // 2], 1),
                "p95": round(lats_sorted[int(len(lats_sorted) * 0.95)], 1),
                "max": round(lats_sorted[-1], 1),
                "mean": round(sum(lats) / len(lats), 1),
            }

    total = sum(c["total"] for c in by_cat.values())
    passed = sum(c["passed"] for c in by_cat.values())

    latency_summary = {}
    if all_latencies:
        s = sorted(all_latencies)
        latency_summary = {
            "min": round(s[0], 1),
            "median": round(s[len(s) // 2], 1),
            "p95": round(s[int(len(s) * 0.95)], 1),
            "max": round(s[-1], 1),
            "mean": round(sum(s) / len(s), 1),
        }

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
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
    if actual == expected:
        return True
    if expected == "" and actual is None:
        return True
    if expected is None and actual is None:
        return True
    try:
        if float(actual) == float(expected):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(expected, str) and isinstance(actual, str):
        if actual.strip().lower() == expected.strip().lower():
            return True
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) == len(actual):
            return all(_values_match(a, e) for a, e in zip(actual, expected))
    return False
