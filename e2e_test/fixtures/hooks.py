"""Pytest hooks for E2E test collection and marker registration.

This module handles:
- Marker registration: Defining custom pytest markers
- Test filtering: Env-var-based filtering by engine, vendor, and GPU tier
- Test ordering: Cluster items by backend config so the session-scoped
  worker pool in ``setup_backend`` can amortize cold starts across classes.
- Session cleanup: Stop any pooled workers at session end.
"""

from __future__ import annotations

import logging
import os

import pytest
from infra import get_runtime

from .markers import resolve_class_marker
from .setup_backend import shutdown_worker_pool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Marker registration
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "engine(*names): engines this test runs on (sglang, vllm, trtllm)",
    )
    config.addinivalue_line(
        "markers",
        "vendor(*names): cloud vendors this test runs on (openai, anthropic, xai, gemini)",
    )
    config.addinivalue_line(
        "markers",
        "gpu(count): number of GPUs required (0, 1, 2, 4)",
    )
    config.addinivalue_line(
        "markers",
        "model(name): mark test to use a specific model from MODEL_SPECS",
    )
    config.addinivalue_line(
        "markers",
        "skip_for_runtime(*runtimes, reason=None): skip test for specific runtimes "
        "(e.g., @pytest.mark.skip_for_runtime('trtllm', reason='no guided decoding'))",
    )
    config.addinivalue_line(
        "markers",
        "gateway(policy=..., timeout=..., extra_args=...): gateway/router configuration",
    )
    config.addinivalue_line(
        "markers",
        "workers(count=1, prefill=None, decode=None): worker topology configuration",
    )
    config.addinivalue_line(
        "markers",
        "storage(backend): storage backend for cloud tests (memory, oracle-custom)",
    )
    config.addinivalue_line(
        "markers",
        "external: mark test as depending on external services",
    )
    config.addinivalue_line(
        "markers",
        "e2e: mark test as an end-to-end test requiring GPU workers",
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow-running",
    )
    config.addinivalue_line(
        "markers",
        "slowtest: mark test as slow-running (alias)",
    )
    config.addinivalue_line(
        "markers",
        "nightly: mark test as a nightly comprehensive benchmark",
    )


# ---------------------------------------------------------------------------
# Runtime-specific skip handling
# ---------------------------------------------------------------------------


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip tests marked with ``@pytest.mark.skip_for_runtime``."""
    marker = item.get_closest_marker("skip_for_runtime")
    if marker:
        current_runtime = get_runtime()
        skip_runtimes = marker.args
        if current_runtime in skip_runtimes:
            reason = marker.kwargs.get("reason", f"Not supported on {current_runtime}")
            pytest.skip(f"Skipping for {current_runtime}: {reason}")


# ---------------------------------------------------------------------------
# Environment-variable-based test filtering
# ---------------------------------------------------------------------------


def _get_marker(item: pytest.Item, name: str):
    """Get the most specific marker, preferring child class over parent.

    Delegates to resolve_class_marker() which walks the class MRO (child-first)
    so that a child class marker overrides a parent class marker.
    """
    return resolve_class_marker(item, name)


def _parametrize_argnames_to_set(argnames: object) -> set[str]:
    if isinstance(argnames, str):
        return {n.strip() for n in argnames.split(",") if n.strip()}
    if isinstance(argnames, (tuple, list)):
        return {str(n) for n in argnames}
    return set()


def _class_level_backend_sort_token(item: pytest.Item) -> str | None:
    """Stable token from class ``pytestmark`` ``parametrize`` for backend fixtures.

    When ``setup_backend`` / ``backend_router`` are parametrized on the class,
    using only ``callspec.params`` splits items by concrete value and scatters
    them in the global sort. Aggregating class-level ``parametrize`` keeps all
    variants of that class adjacent (stable sort preserves intra-class order).
    """
    cls = getattr(item, "cls", None)
    if cls is None:
        return None
    tokens: list[str] = []
    for base in cls.__mro__:
        if base is object:
            continue
        marks = getattr(base, "pytestmark", None)
        if marks is None:
            continue
        if not isinstance(marks, list):
            marks = [marks]
        for mark in marks:
            if getattr(mark, "name", None) != "parametrize":
                continue
            mark_args = tuple(getattr(mark, "args", ()) or ())
            if len(mark_args) < 2:
                continue
            argnames_obj = mark_args[0]
            argvalues_obj = mark_args[1]
            names = _parametrize_argnames_to_set(argnames_obj)
            if not (names & {"setup_backend", "backend_router"}):
                continue
            tokens.append(repr((argnames_obj, argvalues_obj)))
    if not tokens:
        return None
    return "|".join(sorted(set(tokens)))


def _backend_sort_key(item: pytest.Item) -> tuple:
    """Stable ordering key used to cluster test classes by backend config.

    Items that share ``(model_id, workers_config, backend_bucket)`` end up
    adjacent so the session-scoped worker pool in ``setup_backend`` can
    reuse warm workers between consecutive same-config classes. The backend
    bucket prefers an aggregate token from class-level ``parametrize`` marks
    for ``setup_backend`` / ``backend_router`` so mixed parametrizations do
    not scatter tests; otherwise falls back to ``callspec.params``. Items
    without these markers/params fall into a single neutral bucket and
    keep their original collection order via Python's stable sort.
    """
    model = resolve_class_marker(item, "model")
    workers = resolve_class_marker(item, "workers")

    model_id = model.args[0] if model and model.args else ""
    if workers is not None:
        wcount = workers.kwargs.get("count") or 0
        wprefill = workers.kwargs.get("prefill") or 0
        wdecode = workers.kwargs.get("decode") or 0
    else:
        wcount = wprefill = wdecode = 0

    backend = _class_level_backend_sort_token(item)
    if backend is None:
        callspec = getattr(item, "callspec", None)
        if callspec is not None:
            params = getattr(callspec, "params", {})
            backend = str(params.get("setup_backend") or params.get("backend_router") or "")
        else:
            backend = ""

    return (model_id, wcount, wprefill, wdecode, backend)


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Filter collected tests based on env vars, then cluster by backend config."""
    engine = os.environ.get("E2E_ENGINE") or None
    vendor = os.environ.get("E2E_VENDOR") or None
    gpu_tier = os.environ.get("E2E_GPU_TIER") or None

    if any([engine, vendor, gpu_tier]):
        selected: list[pytest.Item] = []
        for item in items:
            if engine:
                engine_marker = _get_marker(item, "engine")
                if not engine_marker or engine not in engine_marker.args:
                    continue
            if vendor:
                vendor_marker = _get_marker(item, "vendor")
                if not vendor_marker or vendor not in vendor_marker.args:
                    continue
            if gpu_tier is not None:
                gpu_marker = _get_marker(item, "gpu")
                gpu_count = gpu_marker.args[0] if gpu_marker else 1
                if str(gpu_count) != gpu_tier:
                    continue
            selected.append(item)

        items[:] = selected

    # Stable sort: equal keys preserve the original (post-filter) order, so
    # tests within a single class stay together and only the class-level
    # grouping changes.
    if os.environ.get("E2E_DISABLE_TEST_SORT", "").lower() not in ("1", "true", "yes"):
        items.sort(key=_backend_sort_key)


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Stop any workers held by the ``setup_backend`` pool at session end."""
    shutdown_worker_pool()
