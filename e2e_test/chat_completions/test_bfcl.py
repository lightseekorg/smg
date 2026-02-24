"""BFCL (Berkeley Function Calling Leaderboard) E2E Tests.

Runs open-source BFCL v3 test cases against the SMG gateway with
per-test JSON logging for full observability.

Logs are written to e2e_test/bfcl_logs/<timestamp>/<category>/<id>_PASS|FAIL.json
with a summary.json at the run root.

Test categories:
  - simple: single function call (400 cases)
  - multiple: pick correct function from several (200 cases)
  - parallel: make parallel function calls (200 cases)
  - parallel_multiple: parallel + multiple (200 cases)
  - irrelevance: model should NOT call any function (240 cases)

Usage:
    # Run all BFCL tests (requires GPU worker + gateway via setup_backend fixture)
    pytest e2e_test/chat_completions/test_bfcl.py -v

    # Run only simple category
    pytest e2e_test/chat_completions/test_bfcl.py -k "simple_" -v

    # Run with a subset (first 20 per category)
    BFCL_LIMIT=20 pytest e2e_test/chat_completions/test_bfcl.py -v

"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter
from typing import Any

import openai
import pytest

try:
    from baselines.compare import (
        baseline_path,
        compare_summary,
        load_baseline,
        save_baseline,
    )

    _HAS_BASELINES = True
except ImportError:
    _HAS_BASELINES = False

from bfcl import (
    bfcl_to_openai_tools,
    evaluate_tool_calls,
    get_run_dir,
    load_bfcl_category,
    save_summary,
    save_test_log,
)

logger = logging.getLogger(__name__)

BFCL_LIMIT = int(os.environ.get("BFCL_LIMIT", "0")) or None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_tool_calls(response: Any) -> list[dict[str, Any]]:
    """Pull structured tool calls out of an OpenAI ChatCompletion response."""
    tool_calls = response.choices[0].message.tool_calls or []
    result = []
    for tc in tool_calls:
        try:
            args = json.loads(tc.function.arguments)
        except (json.JSONDecodeError, TypeError):
            args = {}
        result.append({"name": tc.function.name, "arguments": args})
    return result


def _load(category: str) -> list[dict]:
    try:
        return load_bfcl_category(category, limit=BFCL_LIMIT)
    except FileNotFoundError:
        return []


def _make_ids(cases: list[dict]) -> list[str]:
    return [c["id"] for c in cases]


_all_cases = (
    _load("simple")
    + _load("multiple")
    + _load("parallel")
    + _load("parallel_multiple")
    + _load("irrelevance")
)


# ---------------------------------------------------------------------------
# Session-scoped fixtures for the log directory and result collector
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def bfcl_run_dir():
    """Single timestamped directory for all BFCL logs in this test session."""
    return get_run_dir()


_all_results: list[dict[str, Any]] = []


@pytest.fixture(autouse=True, scope="session")
def _write_summary_on_exit(bfcl_run_dir):
    """Write summary.json and compare against baseline when the session ends."""
    yield
    if not _all_results:
        return

    summary = save_summary(bfcl_run_dir, _all_results)
    logger.info(
        "BFCL summary: %d/%d passed (%.1f%%) — %s/summary.json",
        summary["passed"],
        summary["total"],
        summary["accuracy_pct"],
        bfcl_run_dir,
    )

    if not _HAS_BASELINES:
        return

    model_counts = Counter(r.get("model", "unknown") for r in _all_results)
    backend_counts = Counter(r.get("backend", "unknown") for r in _all_results)
    model = model_counts.most_common(1)[0][0]
    backend = backend_counts.most_common(1)[0][0]

    bl_path = baseline_path("bfcl", model, backend)

    if os.environ.get("BFCL_UPDATE_BASELINE"):
        save_baseline(bl_path, summary, model=model, backend=backend)
        logger.info("Baseline updated — commit %s to lock it in", bl_path)
        return

    baseline = load_baseline(bl_path)
    if baseline is None:
        logger.info(
            "No baseline found for bfcl / %s / %s — skipping comparison. "
            "Run with BFCL_UPDATE_BASELINE=1 to create one.",
            model,
            backend,
        )
        return

    comparison = compare_summary(summary, baseline, bl_path)

    comparison_file = bfcl_run_dir / "comparison.json"
    comparison_file.write_text(
        json.dumps(comparison.to_dict(), indent=2) + "\n"
    )

    report = comparison.format_report()
    if comparison.passed:
        logger.info("\n%s", report)
    else:
        logger.warning("\n%s", report)


# ---------------------------------------------------------------------------
# Core runner shared by all test classes
# ---------------------------------------------------------------------------


def _run_bfcl_case(
    *,
    case: dict,
    model: str,
    parser: str,
    backend: str,
    client: openai.OpenAI,
    run_dir: Any,
) -> None:
    """Execute a single BFCL test case, log the result, assert on failure."""
    category = case["category"]
    test_id = case["id"]
    messages = case["question"]
    tools = bfcl_to_openai_tools(case["function"])

    request_payload = {
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0.01,
        "max_tokens": 1024,
    }

    start = time.monotonic()
    response_payload = None
    actual: list[dict[str, Any]] = []

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.01,
            max_tokens=1024,
        )
        response_payload = response.model_dump()
        actual = _extract_tool_calls(response)
    except Exception as exc:
        latency = (time.monotonic() - start) * 1000
        save_test_log(
            run_dir,
            test_id=test_id,
            category=category,
            model=model,
            parser=parser,
            backend=backend,
            request_payload=request_payload,
            response_payload=None,
            ground_truth=case.get("ground_truth", []),
            actual_tool_calls=[],
            passed=False,
            errors=[f"API error: {exc}"],
            latency_ms=latency,
        )
        _all_results.append({
            "test_id": test_id,
            "category": category,
            "passed": False,
            "errors": [f"API error: {exc}"],
            "latency_ms": latency,
            "finish_reason": None,
            "completion_tokens": None,
            "had_reasoning": False,
            "log_file": f"{category}/{test_id.replace('/', '_')}_FAIL.json",
            "model": model,
            "backend": backend,
        })
        pytest.fail(f"BFCL {test_id}: API call failed — {exc}")

    latency = (time.monotonic() - start) * 1000
    passed, errors = evaluate_tool_calls(
        actual,
        case.get("ground_truth", []),
        category=category,
    )

    log_path = save_test_log(
        run_dir,
        test_id=test_id,
        category=category,
        model=model,
        parser=parser,
        backend=backend,
        request_payload=request_payload,
        response_payload=response_payload,
        ground_truth=case.get("ground_truth", []),
        actual_tool_calls=actual,
        passed=passed,
        errors=errors,
        latency_ms=latency,
    )

    finish_reason = None
    completion_tokens = None
    had_reasoning = False
    if response_payload:
        choices = response_payload.get("choices") or []
        if choices:
            finish_reason = choices[0].get("finish_reason")
            msg = choices[0].get("message") or {}
            had_reasoning = bool(msg.get("reasoning_content"))
        usage = response_payload.get("usage") or {}
        completion_tokens = usage.get("completion_tokens")

    _all_results.append({
        "test_id": test_id,
        "category": category,
        "passed": passed,
        "errors": errors,
        "latency_ms": latency,
        "finish_reason": finish_reason,
        "completion_tokens": completion_tokens,
        "had_reasoning": had_reasoning,
        "log_file": log_path.name,
        "model": model,
        "backend": backend,
    })

    status = "PASS" if passed else "FAIL"
    logger.info(
        "BFCL %s [%s] %.0fms → %s",
        test_id,
        status,
        latency,
        log_path.name,
    )

    if not passed:
        pytest.fail(f"BFCL {test_id}: {'; '.join(errors)}")


# ============================================================================
# Fixture-based tests (use setup_backend → launches GPU worker + gateway)
# ============================================================================


@pytest.mark.skip_for_runtime(
    "trtllm", reason="TRT-LLM does not support guided decoding (json_schema)"
)
@pytest.mark.model("Qwen/Qwen2.5-7B-Instruct")
@pytest.mark.gateway(extra_args=["--tool-call-parser", "qwen", "--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestBFCLQwen:
    """BFCL v3 accuracy — Qwen 2.5 7B with qwen parser (all categories)."""

    @pytest.mark.parametrize("case", _all_cases, ids=_make_ids(_all_cases))
    def test_case(self, setup_backend, bfcl_run_dir, case):
        backend_name, model, client, _ = setup_backend
        _run_bfcl_case(
            case=case, model=model, parser="qwen", backend=backend_name,
            client=client, run_dir=bfcl_run_dir,
        )


# ============================================================================
# Standalone mode: run against an already-running gateway
# ============================================================================


@pytest.mark.skipif(
    not os.environ.get("BFCL_BASE_URL"),
    reason="Set BFCL_BASE_URL to run standalone BFCL tests",
)
class TestBFCLStandalone:
    """Run BFCL tests against an externally managed gateway.

    Set environment variables:
        BFCL_BASE_URL=http://localhost:30000
        BFCL_MODEL=Qwen/Qwen2.5-7B-Instruct
        BFCL_PARSER=qwen  (for logging only)
        BFCL_LIMIT=20     (optional: limit cases per category)
    """

    @pytest.fixture(autouse=True)
    def _setup_client(self, bfcl_run_dir):
        base_url = os.environ["BFCL_BASE_URL"]
        self.model = os.environ.get("BFCL_MODEL", "default")
        self.parser = os.environ.get("BFCL_PARSER", "unknown")
        self.client = openai.OpenAI(
            base_url=f"{base_url.rstrip('/')}/v1",
            api_key=os.environ.get("BFCL_API_KEY", "not-used"),
        )
        self.run_dir = bfcl_run_dir

    @pytest.mark.parametrize("case", _all_cases, ids=_make_ids(_all_cases))
    def test_case(self, case):
        _run_bfcl_case(
            case=case, model=self.model, parser=self.parser, backend="standalone",
            client=self.client, run_dir=self.run_dir,
        )
