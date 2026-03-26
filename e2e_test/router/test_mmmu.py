"""MMMU evaluation tests for multimodal router functionality.

Tests the router's ability to handle MMMU (Massive Multi-discipline
Multimodal Understanding) benchmark evaluations using Qwen3-VL.

Uses the Art and Design category (120 samples) for a manageable
CI-friendly evaluation that still provides meaningful signal.

Usage:
    # Run MMMU tests
    pytest e2e_test/router/test_mmmu.py -v

    # Run with specific backend
    pytest e2e_test/router/test_mmmu.py -v -k "grpc"
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from infra import run_eval

logger = logging.getLogger(__name__)

# Baseline accuracy from Qwen3-VL-8B-Instruct on Art and Design category
# vLLM gRPC: ~0.60-0.61
MMMU_THRESHOLD = 0.57


@pytest.mark.engine("vllm")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.model("Qwen/Qwen3-VL-8B-Instruct")
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestMMMUGrpc:
    """MMMU evaluation tests using gRPC workers."""

    def test_mmmu_art_and_design(self, setup_backend):
        """MMMU Art and Design category evaluation.

        Runs 120 multimodal questions (Art, Art_Theory, Design, Music)
        and validates accuracy meets minimum threshold.
        """
        backend, model, client, *_ = setup_backend

        args = SimpleNamespace(
            base_url=str(client.base_url),
            model=model,
            eval_name="mmmu_art_and_design",
            num_examples=None,  # Use all 120 samples
            num_threads=32,
            temperature=0.0,
        )
        metrics = run_eval(args)

        assert metrics["score"] >= MMMU_THRESHOLD, (
            f"MMMU score {metrics['score']:.3f} below threshold {MMMU_THRESHOLD}"
        )
        logger.info(
            "MMMU Art and Design gRPC score: %.3f (threshold: %.2f)",
            metrics["score"],
            MMMU_THRESHOLD,
        )
