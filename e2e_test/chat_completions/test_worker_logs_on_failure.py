"""Temporary test to verify worker logs are dumped on failure.

This test deliberately fails to confirm the pytest_runtest_makereport
hook dumps worker logs inline. DELETE THIS FILE after verification.
"""

from __future__ import annotations

import pytest


@pytest.mark.engine("sglang", "vllm")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestWorkerLogDump:
    """Deliberately failing test to verify worker log dump on failure."""

    def test_deliberate_failure(self, model, setup_backend):
        """This test always fails to trigger the worker log dump hook."""
        _, _, client, *_ = setup_backend

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say hi"}],
            max_tokens=10,
        )

        # Deliberately fail
        assert False, "DELIBERATE FAILURE: check that worker logs appear below this line"
