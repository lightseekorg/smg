"""Pytest hooks for E2E test collection and marker registration.

This module handles:
- Marker registration: Defining custom pytest markers
- Test filtering: Env-var-based filtering by engine, vendor, and GPU tier
"""

from __future__ import annotations

import os

import pytest
from infra import get_runtime

from .markers import resolve_class_marker

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
    """Skip tests marked with ``@pytest.mark.skip_for_runtime``.

    A single test item can carry multiple ``skip_for_runtime`` marks — e.g. a
    method-level ``@pytest.mark.skip_for_runtime("trtllm", ...)`` plus a
    parametrize-attached ``pytest.param(5, marks=skip_for_runtime("tokenspeed",
    ...))``. ``get_closest_marker`` only returns one of them, which silently
    drops the others. Iterate every mark so a runtime that's named in any of
    them gets skipped, regardless of which is "closest".
    """
    current_runtime = get_runtime()
    for marker in item.iter_markers(name="skip_for_runtime"):
        if current_runtime in marker.args:
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


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Filter collected tests based on E2E_ENGINE, E2E_VENDOR, and E2E_GPU_TIER env vars."""
    engine = os.environ.get("E2E_ENGINE") or None
    vendor = os.environ.get("E2E_VENDOR") or None
    gpu_tier = os.environ.get("E2E_GPU_TIER") or None

    if not any([engine, vendor, gpu_tier]):
        return

    selected: list[pytest.Item] = []
    for item in items:
        # Filter by engine
        if engine:
            engine_marker = _get_marker(item, "engine")
            if not engine_marker or engine not in engine_marker.args:
                continue
        # Filter by vendor
        if vendor:
            vendor_marker = _get_marker(item, "vendor")
            if not vendor_marker or vendor not in vendor_marker.args:
                continue
        # Filter by GPU tier
        if gpu_tier is not None:
            gpu_marker = _get_marker(item, "gpu")
            gpu_count = gpu_marker.args[0] if gpu_marker else 1
            if str(gpu_count) != gpu_tier:
                continue
        selected.append(item)

    items[:] = selected
