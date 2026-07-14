"""Unit tests for EPD command-builder logic (no GPU, no wheel required)."""

from __future__ import annotations

from infra.constants import WorkerType


def test_worker_type_encode_exists():
    assert WorkerType.ENCODE == "encode"
    assert WorkerType.ENCODE.value == "encode"
