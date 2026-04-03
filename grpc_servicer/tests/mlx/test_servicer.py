"""Tests for MlxEngineServicer: __init__, generation loop, Generate, Abort."""

import asyncio
import time
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock

import pytest

from smg_grpc_proto import vllm_engine_pb2

from smg_grpc_servicer.mlx.servicer import MlxEngineServicer


@dataclass
class FakeGenResponse:
    uid: int
    token: int
    logprobs: object
    finish_reason: Optional[str] = None
    current_state: Optional[str] = None
    match_sequence: Optional[list] = None
    prompt_cache: Optional[list] = None
    all_tokens: Optional[list] = None


class FakeLogprobs:
    def __init__(self, vocab_size=100, token_logprob=-0.5):
        self._vocab_size = vocab_size
        self._token_logprob = token_logprob

    def __getitem__(self, idx):
        return FakeScalar(self._token_logprob)

    @property
    def shape(self):
        return (self._vocab_size,)


class FakeScalar:
    def __init__(self, val):
        self._val = val

    def item(self):
        return self._val


class FakeBatchGenerator:
    """Fake batch generator that only yields responses when a queue is registered."""

    def __init__(self, responses_by_uid=None):
        self._responses = responses_by_uid or {}
        self._insert_count = 0
        self._inserted_uids = set()
        self._removed_uids = []
        self._servicer = None  # set after construction to gate on queue readiness

    def insert(self, prompts, max_tokens=None, samplers=None,
               logits_processors=None, state_machines=None):
        uids = []
        for _ in prompts:
            uid = self._insert_count
            self._insert_count += 1
            self._inserted_uids.add(uid)
            uids.append(uid)
        return uids

    def next(self):
        gen_responses = []
        for uid, responses in list(self._responses.items()):
            if uid not in self._inserted_uids or uid in self._removed_uids:
                continue
            # Only yield if the servicer has registered a queue for this uid
            if self._servicer and uid not in self._servicer._uid_queues:
                continue
            if responses:
                gen_responses.append(responses.pop(0))
        return [], gen_responses

    def remove(self, uids):
        self._removed_uids.extend(uids)

    def close(self):
        pass


def _make_servicer(fake_bg):
    servicer = object.__new__(MlxEngineServicer)
    servicer.batch_generator = fake_bg
    servicer.model_path = "test-model"
    servicer.model_dir = "/tmp/test-model"
    servicer.model_config = {
        "vocab_size": 100,
        "max_position_embeddings": 4096,
        "eos_token_id": 2,
    }
    servicer._eos_token_ids = [2]
    servicer._active_requests = 0
    servicer.start_time = time.time()
    servicer._request_uid_map = {}
    servicer._uid_queues = {}
    servicer._shutdown_event = asyncio.Event()
    servicer._loop = None
    servicer._gen_thread = None
    fake_bg._servicer = servicer
    return servicer


class TestGenerateStreaming:
    @pytest.mark.asyncio
    async def test_streaming_yields_chunks_then_complete(self):
        logprobs = FakeLogprobs()
        fake_bg = FakeBatchGenerator(responses_by_uid={
            0: [
                FakeGenResponse(uid=0, token=10, logprobs=logprobs),
                FakeGenResponse(uid=0, token=20, logprobs=logprobs),
                FakeGenResponse(uid=0, token=2, logprobs=logprobs, finish_reason="stop", match_sequence=[2]),
            ],
        })
        servicer = _make_servicer(fake_bg)
        loop = asyncio.get_running_loop()
        servicer._loop = loop

        servicer.start_generation_loop()

        request = vllm_engine_pb2.GenerateRequest(
            request_id="req-1",
            tokenized=vllm_engine_pb2.TokenizedInput(input_ids=[1, 5, 7]),
            sampling_params=vllm_engine_pb2.SamplingParams(),
            stream=True,
        )
        context = MagicMock()
        context.cancelled = MagicMock(return_value=False)

        responses = []
        async for resp in servicer.Generate(request, context):
            responses.append(resp)

        servicer.stop_generation_loop()

        # In streaming: each token yields a chunk. On finish, yield chunk + complete.
        # So: chunk(10), chunk(20), chunk(2), complete = 4 responses
        chunks = [r for r in responses if r.HasField("chunk")]
        completes = [r for r in responses if r.HasField("complete")]
        assert len(chunks) >= 2  # At least the first 2 tokens
        assert len(completes) == 1
        assert completes[0].complete.finish_reason == "stop"


class TestGenerateNonStreaming:
    @pytest.mark.asyncio
    async def test_non_streaming_returns_all_tokens_in_complete(self):
        logprobs = FakeLogprobs()
        fake_bg = FakeBatchGenerator(responses_by_uid={
            0: [
                FakeGenResponse(uid=0, token=10, logprobs=logprobs),
                FakeGenResponse(uid=0, token=20, logprobs=logprobs),
                FakeGenResponse(uid=0, token=2, logprobs=logprobs, finish_reason="stop"),
            ],
        })
        servicer = _make_servicer(fake_bg)
        loop = asyncio.get_running_loop()
        servicer._loop = loop
        servicer.start_generation_loop()

        request = vllm_engine_pb2.GenerateRequest(
            request_id="req-2",
            tokenized=vllm_engine_pb2.TokenizedInput(input_ids=[1, 5]),
            sampling_params=vllm_engine_pb2.SamplingParams(),
            stream=False,
        )
        context = MagicMock()
        context.cancelled = MagicMock(return_value=False)

        responses = []
        async for resp in servicer.Generate(request, context):
            responses.append(resp)

        servicer.stop_generation_loop()

        assert len(responses) == 1
        assert responses[0].HasField("complete")
        assert list(responses[0].complete.output_ids) == [10, 20, 2]


class TestAbort:
    @pytest.mark.asyncio
    async def test_abort_removes_request(self):
        fake_bg = FakeBatchGenerator()
        servicer = _make_servicer(fake_bg)
        servicer._request_uid_map["req-1"] = 0
        servicer._uid_queues[0] = asyncio.Queue()

        request = vllm_engine_pb2.AbortRequest(request_ids=["req-1"])
        context = MagicMock()
        resp = await servicer.Abort(request, context)

        assert isinstance(resp, vllm_engine_pb2.AbortResponse)
        assert "req-1" not in servicer._request_uid_map
        assert 0 in fake_bg._removed_uids


class TestStubRPCs:
    @pytest.mark.asyncio
    async def test_embed_returns_unimplemented(self):
        fake_bg = FakeBatchGenerator()
        servicer = _make_servicer(fake_bg)
        request = vllm_engine_pb2.EmbedRequest(request_id="e-1")
        context = MagicMock()
        context.abort = MagicMock(side_effect=Exception("UNIMPLEMENTED"))
        with pytest.raises(Exception, match="UNIMPLEMENTED"):
            await servicer.Embed(request, context)

    @pytest.mark.asyncio
    async def test_subscribe_kv_events_returns_unimplemented(self):
        from smg_grpc_proto.generated import common_pb2

        fake_bg = FakeBatchGenerator()
        servicer = _make_servicer(fake_bg)
        request = common_pb2.SubscribeKvEventsRequest()
        context = MagicMock()
        context.abort = MagicMock(side_effect=Exception("UNIMPLEMENTED"))
        with pytest.raises(Exception, match="UNIMPLEMENTED"):
            async for _ in servicer.SubscribeKvEvents(request, context):
                pass
