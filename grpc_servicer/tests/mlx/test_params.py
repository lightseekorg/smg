"""Tests for sampling params conversion from proto to mlx-lm."""

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from smg_grpc_proto import vllm_engine_pb2

from smg_grpc_servicer.mlx.servicer import MlxEngineServicer


class TestBuildSampler:
    def test_default_params_returns_callable(self):
        """Default (empty) sampling params should produce a working sampler."""
        params = vllm_engine_pb2.SamplingParams()
        sampler = MlxEngineServicer._build_sampler(params)
        logits = mx.zeros((10,))
        token = sampler(logits)
        assert isinstance(token, mx.array)

    def test_temperature_and_top_p(self):
        """Temperature and top_p should be forwarded to make_sampler."""
        params = vllm_engine_pb2.SamplingParams(temperature=0.8, top_p=0.95)
        sampler = MlxEngineServicer._build_sampler(params)
        logits = mx.ones((100,))
        token = sampler(logits)
        assert 0 <= token.item() < 100

    def test_top_k(self):
        """top_k should limit sampling to top k tokens."""
        params = vllm_engine_pb2.SamplingParams(temperature=1.0, top_k=5)
        sampler = MlxEngineServicer._build_sampler(params)
        logits = mx.ones((100,))
        token = sampler(logits)
        assert 0 <= token.item() < 100

    def test_zero_temperature_is_argmax(self):
        """temp=0 (default) should produce deterministic argmax."""
        params = vllm_engine_pb2.SamplingParams()
        sampler = MlxEngineServicer._build_sampler(params)
        logits = mx.array([0.1, 0.5, 0.3, 0.9, 0.2])
        token = sampler(logits)
        assert token.item() == 3  # argmax


class TestBuildLogitsProcessors:
    def test_empty_params_returns_empty_list(self):
        """No penalties/bias should return empty processor list."""
        params = vllm_engine_pb2.SamplingParams()
        processors = MlxEngineServicer._build_logits_processors(params)
        assert processors == []

    def test_repetition_penalty(self):
        """repetition_penalty should produce a logits processor."""
        params = vllm_engine_pb2.SamplingParams(repetition_penalty=1.2)
        processors = MlxEngineServicer._build_logits_processors(params)
        assert len(processors) >= 1
        tokens = mx.array([1, 2, 3])
        logits = mx.ones((1, 100))
        result = processors[0](tokens, logits)
        assert result.shape == logits.shape

    def test_logit_bias(self):
        """logit_bias dict should produce a logits processor."""
        params = vllm_engine_pb2.SamplingParams(logit_bias={5: 10.0, 10: -10.0})
        processors = MlxEngineServicer._build_logits_processors(params)
        assert len(processors) >= 1

    def test_multiple_penalties(self):
        """Multiple penalty types should each produce a processor."""
        params = vllm_engine_pb2.SamplingParams(
            repetition_penalty=1.1,
            frequency_penalty=0.5,
            presence_penalty=0.3,
            logit_bias={0: 1.0},
        )
        processors = MlxEngineServicer._build_logits_processors(params)
        assert len(processors) == 4


class TestBuildStateMachine:
    def test_no_stop_tokens_returns_default(self):
        """No stop tokens should return a default state machine."""
        params = vllm_engine_pb2.SamplingParams()
        eos_token_ids = []
        sm = MlxEngineServicer._build_state_machine(params, eos_token_ids)
        assert sm is not None

    def test_stop_token_ids(self):
        """stop_token_ids should create stop transitions."""
        params = vllm_engine_pb2.SamplingParams(stop_token_ids=[128001, 128009])
        eos_token_ids = []
        sm = MlxEngineServicer._build_state_machine(params, eos_token_ids)
        assert sm is not None

    def test_eos_tokens_included_by_default(self):
        """EOS tokens should be in the state machine unless ignore_eos."""
        params = vllm_engine_pb2.SamplingParams()
        eos_token_ids = [2]
        sm = MlxEngineServicer._build_state_machine(params, eos_token_ids)
        assert sm is not None

    def test_ignore_eos_excludes_eos(self):
        """ignore_eos=True should exclude EOS tokens from state machine."""
        params = vllm_engine_pb2.SamplingParams(ignore_eos=True, stop_token_ids=[999])
        eos_token_ids = [2]
        sm = MlxEngineServicer._build_state_machine(params, eos_token_ids)
        assert sm is not None
