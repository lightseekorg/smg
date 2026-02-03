"""MMLU evaluation tests for vLLM PD (Prefill-Decode) disaggregated routing.

vLLM PD disaggregation uses NIXL for transparent KV cache transfer
between prefill and decode workers. The router sends the request to
prefill with max_tokens=1 (to compute KV cache), then sends the original
request to decode (which auto-discovers KV cache via NIXL).

Requirements:
    - vLLM with NIXL support
    - GPUs: num_prefill + num_decode (default: 2 GPUs for 1+1)
    - RDMA/InfiniBand connectivity between GPUs

Usage:
    pytest e2e_test/router/test_vllm_pd_mmlu.py -v
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from infra import run_eval

logger = logging.getLogger(__name__)


@pytest.mark.e2e
@pytest.mark.parametrize("setup_backend", ["vllm_pd"], indirect=True)
class TestVllmPDMMLU:
    """MMLU evaluation tests using vLLM PD disaggregated routing."""

    def test_vllm_pd_mmlu_basic(self, setup_backend):
        """Basic MMLU evaluation with vLLM PD disaggregation.

        Runs MMLU with 1 prefill + 1 decode worker and validates
        accuracy meets threshold (>= 0.65).
        """
        backend, model, client, *_ = setup_backend

        args = SimpleNamespace(
            base_url=str(client.base_url),
            model=model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
            temperature=0.1,
        )
        metrics = run_eval(args)

        assert (
            metrics["score"] >= 0.65
        ), f"vLLM PD MMLU score {metrics['score']:.2f} below threshold 0.65"
        logger.info("vLLM PD MMLU score: %.2f (threshold: 0.65)", metrics["score"])
