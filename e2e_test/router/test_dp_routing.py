"""DP rank-pinning verification for vLLM gRPC workers (``--dp-aware``).

Output correctness alone cannot catch broken pinning: when no rank reaches the
engine, vLLM load-balances across DP ranks internally and still produces
correct completions. This test asserts the transmitted rank instead, via the
servicer's per-request ``dp_rank=<n>`` log line:

1. Both ``dp_rank=0`` and ``dp_rank=1`` appear in the worker log — the router
   discovered dp_size, expanded the single worker URL into one worker per
   rank, and pinned requests to both.
2. ``dp_rank=None`` never appears — every request carried a pinned rank
   instead of falling back to vLLM-internal balancing.

Requirements:
    - vLLM gRPC worker launched with --data-parallel-size 2
    - 2 GPUs (one per DP rank)

Usage:
    E2E_RUNTIME=vllm pytest e2e_test/router/test_dp_routing.py -v
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import pytest
from infra.pd_logs import (
    assert_worker_logs_captured,
    unique_prompt,
    wait_for_marker,
    worker_log_dir,
)

logger = logging.getLogger(__name__)

# Router logs land here via the gateway marker (rolling files named smg.YYYY-MM-DD)
_LOG_DIR = Path(tempfile.gettempdir()) / f"smg-e2e-dp-routing-{os.getpid()}"

# Scope to this test's worker — PD workers in the same CI job log dp_rank=None
_WORKER_LOG_GLOB = "worker-meta-llama__Llama-3.2-1B-Instruct_vllm_grpc_*.log"

RANK_MARKERS = ("dp_rank=0", "dp_rank=1")
UNPINNED_MARKER = "dp_rank=None"


@pytest.mark.engine("vllm")
@pytest.mark.gpu(2)
@pytest.mark.model("meta-llama/Llama-3.2-1B-Instruct")
@pytest.mark.e2e
@pytest.mark.gateway(log_level="debug", log_dir=str(_LOG_DIR), extra_args=["--dp-aware"])
@pytest.mark.workers(count=1, gpus=2, extra_engine_args=["--data-parallel-size", "2"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestDpAwareRankPinning:
    """Verify the router pins every request to a DP rank and covers both ranks."""

    def test_requests_pinned_to_both_ranks(self, setup_backend):
        backend, model, client, *_ = setup_backend

        for _ in range(8):
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": unique_prompt()}],
                max_tokens=16,
                temperature=0.0,
            )
            assert response.choices[0].message.content, "Empty completion from DP backend"

        worker_dir = worker_log_dir(_LOG_DIR)
        wait_for_marker(worker_dir, _WORKER_LOG_GLOB, RANK_MARKERS[0])
        worker_logs = wait_for_marker(worker_dir, _WORKER_LOG_GLOB, RANK_MARKERS[1])
        assert_worker_logs_captured(worker_logs, "DP rank pinning")

        missing = [m for m in RANK_MARKERS if m not in worker_logs]
        assert not missing, (
            f"Router never pinned {missing} — 8 round-robin requests across "
            f"dp_size=2 must hit both ranks; checked {worker_dir}/{_WORKER_LOG_GLOB}"
        )
        assert UNPINNED_MARKER not in worker_logs, (
            "Worker received a request without a pinned DP rank — the router "
            f"fell back to vLLM-internal balancing; checked {worker_dir}/{_WORKER_LOG_GLOB}"
        )
