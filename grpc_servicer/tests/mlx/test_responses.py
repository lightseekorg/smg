"""Tests for gRPC response builders (logprobs, chunk, complete)."""

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from smg_grpc_proto import vllm_engine_pb2

from smg_grpc_servicer.mlx.servicer import MlxEngineServicer


class TestBuildOutputLogprobs:
    def test_returns_none_when_num_logprobs_is_none(self):
        result = MlxEngineServicer._build_output_logprobs(
            token_id=5, logprobs_array=mx.zeros((100,)), num_logprobs=None
        )
        assert result is None

    def test_returns_logprobs_for_token(self):
        logprobs = mx.array([-2.0, -1.0, -0.5, -3.0, -0.1])
        result = MlxEngineServicer._build_output_logprobs(
            token_id=4, logprobs_array=logprobs, num_logprobs=3
        )
        assert result is not None
        assert len(result.token_ids) == 1
        assert result.token_ids[0] == 4
        assert abs(result.token_logprobs[0] - (-0.1)) < 0.01
        assert len(result.top_logprobs) == 1
        assert len(result.top_logprobs[0].token_ids) == 3

    def test_top_logprobs_sorted_descending(self):
        logprobs = mx.array([-5.0, -1.0, -3.0, -0.5, -2.0])
        result = MlxEngineServicer._build_output_logprobs(
            token_id=3, logprobs_array=logprobs, num_logprobs=3
        )
        top = result.top_logprobs[0]
        assert list(top.token_ids) == [3, 1, 4]


class TestChunkResponse:
    def test_basic_chunk(self):
        resp = MlxEngineServicer._chunk_response(
            token_ids=[42], prompt_tokens=10, completion_tokens=1,
            cached_tokens=0, index=0, output_logprobs=None,
        )
        assert resp.HasField("chunk")
        chunk = resp.chunk
        assert list(chunk.token_ids) == [42]
        assert chunk.prompt_tokens == 10
        assert chunk.completion_tokens == 1
        assert chunk.index == 0

    def test_chunk_with_logprobs(self):
        logprobs_proto = vllm_engine_pb2.OutputLogProbs(
            token_ids=[42], token_logprobs=[-0.5]
        )
        resp = MlxEngineServicer._chunk_response(
            token_ids=[42], prompt_tokens=10, completion_tokens=1,
            cached_tokens=0, index=0, output_logprobs=logprobs_proto,
        )
        assert resp.chunk.HasField("output_logprobs")


class TestCompleteResponse:
    def test_basic_complete_stop(self):
        resp = MlxEngineServicer._complete_response(
            output_ids=[10, 20, 30], finish_reason="stop",
            prompt_tokens=5, completion_tokens=3, cached_tokens=0, index=0,
        )
        assert resp.HasField("complete")
        complete = resp.complete
        assert list(complete.output_ids) == [10, 20, 30]
        assert complete.finish_reason == "stop"
        assert complete.prompt_tokens == 5
        assert complete.completion_tokens == 3

    def test_complete_length(self):
        resp = MlxEngineServicer._complete_response(
            output_ids=[1, 2], finish_reason="length",
            prompt_tokens=10, completion_tokens=2, cached_tokens=0, index=0,
        )
        assert resp.complete.finish_reason == "length"

    def test_complete_with_matched_token_id(self):
        resp = MlxEngineServicer._complete_response(
            output_ids=[1, 2], finish_reason="stop",
            prompt_tokens=10, completion_tokens=2, cached_tokens=0, index=0,
            matched_token_id=128001,
        )
        assert resp.complete.matched_token_id == 128001
