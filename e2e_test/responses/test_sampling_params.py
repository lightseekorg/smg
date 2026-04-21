"""Sampling parameter pass-through tests for Response API.

Verifies that sampling parameters (temperature, top_p, top_k, min_p,
frequency_penalty, presence_penalty, repetition_penalty) are accepted
by the gateway and produce valid responses through gRPC backends.
"""

from __future__ import annotations

import logging

import pytest

logger = logging.getLogger(__name__)


class _SamplingParamsBase:
    """Shared sampling parameter tests — subclass to set model/engine marks."""

    def test_all_sampling_params_accepted(self, model, api_client):
        """Non-default sampling params produce a valid response with correct echo-back."""
        resp = api_client.responses.create(
            model=model,
            input="What is 2+2?",
            temperature=0.5,
            top_p=0.9,
            max_output_tokens=256,
            extra_body={
                "top_k": 40,
                "min_p": 0.05,
                "frequency_penalty": 0.5,
                "presence_penalty": 0.3,
                "repetition_penalty": 1.1,
            },
        )
        assert resp.status == "completed"
        assert resp.error is None
        assert len(resp.output) > 0

        assert resp.temperature == pytest.approx(0.5)
        assert resp.top_p == pytest.approx(0.9, abs=1e-4)
        assert resp.max_output_tokens == 256

        assert resp.usage is not None
        tokens = resp.usage.output_tokens or resp.usage.completion_tokens
        assert tokens is not None
        assert tokens > 0

    def test_temperature_zero_is_deterministic(self, model, api_client):
        """temperature=0 reaches the backend — same prompt yields identical output."""
        kwargs = {
            "model": model,
            "input": "What is the capital of France? Answer in one word.",
            "temperature": 0,
            "max_output_tokens": 16,
        }
        resp1 = api_client.responses.create(**kwargs)
        resp2 = api_client.responses.create(**kwargs)

        assert resp1.status == "completed"
        assert resp2.status == "completed"
        assert resp1.output_text == resp2.output_text

    def test_sampling_params_streaming(self, model, api_client):
        """Sampling params work in streaming mode."""
        stream = api_client.responses.create(
            model=model,
            input="Say hello.",
            stream=True,
            temperature=0.5,
            top_p=0.9,
            max_output_tokens=256,
            extra_body={
                "top_k": 40,
                "min_p": 0.05,
                "frequency_penalty": 0.5,
                "presence_penalty": 0.3,
                "repetition_penalty": 1.1,
            },
        )
        events = list(stream)
        completed = [e for e in events if e.type == "response.completed"]
        assert len(completed) == 1

        resp = completed[0].response
        assert resp.status == "completed"
        assert len(resp.output) > 0
        assert resp.temperature == pytest.approx(0.5)
        assert resp.top_p == pytest.approx(0.9, abs=1e-4)
        assert resp.usage is not None
        tokens = resp.usage.output_tokens or resp.usage.completion_tokens
        assert tokens is not None
        assert tokens > 0


@pytest.mark.engine("sglang")
@pytest.mark.gpu(2)
@pytest.mark.e2e
@pytest.mark.model("Qwen/Qwen2.5-14B-Instruct")
@pytest.mark.gateway(extra_args=["--tool-call-parser", "qwen", "--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestSamplingParamsLocal(_SamplingParamsBase):
    """Regular model (Qwen via SGLang)."""


@pytest.mark.engine("sglang", "vllm", "trtllm")
@pytest.mark.gpu(2)
@pytest.mark.e2e
@pytest.mark.model("openai/gpt-oss-20b")
@pytest.mark.gateway(extra_args=["--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestSamplingParamsHarmony(_SamplingParamsBase):
    """Harmony model (gpt-oss-20b via all gRPC backends)."""
