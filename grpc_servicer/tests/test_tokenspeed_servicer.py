"""Unit tests for ``smg_grpc_servicer.tokenspeed.servicer``.

Runs against a minimal ``FakeAsyncLLM`` that implements only the AsyncLLM
surface the servicer actually touches. We *do* require TokenSpeed to be
importable (the servicer takes real request classes from ``tokenspeed.*``),
so the whole module is skipped when TokenSpeed is not installed.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import grpc
import pytest

pytest.importorskip(
    "smg_grpc_proto",
    reason="smg-grpc-proto must be installed to test the servicer",
)

from smg_grpc_proto import sglang_scheduler_pb2  # noqa: E402
from smg_grpc_servicer.tokenspeed import servicer as _servicer_module  # noqa: E402
from smg_grpc_servicer.tokenspeed.servicer import (  # noqa: E402
    TokenSpeedSchedulerServicer,
    _abort_status_code,
    _finish_reason_to_dict,
    _make_json_serializable,
)

# ---------------------------------------------------------------------------
# Stub request classes. The servicer lazily imports ``GenerateReqInput`` and
# ``EmbeddingReqInput`` so tests can substitute minimal local stand-ins
# without pulling in TokenSpeed's full scheduler graph.
# ---------------------------------------------------------------------------


class _StubReq:
    """Minimal stand-in with the attributes the servicer sets on req objects."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        # Allow later attribute assignment for rid / text / bootstrap_*.
        self.rid = None
        self.text = None


class StubGenerateReqInput(_StubReq):
    pass


class StubEmbeddingReqInput(_StubReq):
    pass


@pytest.fixture(autouse=True)
def _stub_request_inputs(monkeypatch):
    """Redirect the servicer's lazy imports to the local stubs."""
    monkeypatch.setattr(_servicer_module, "_lazy_generate_req_input", lambda: StubGenerateReqInput)
    monkeypatch.setattr(
        _servicer_module, "_lazy_embedding_req_input", lambda: StubEmbeddingReqInput
    )
    yield


# ---------------------------------------------------------------------------
# Local fake finish-reason classes. The servicer duck-types on ``.to_json()``
# so tests don't need to import TokenSpeed's request_types module (which
# pulls in the full scheduler graph and breaks in minimal test envs).
# ---------------------------------------------------------------------------


class FINISH_MATCHED_TOKEN:
    def __init__(self, matched):
        self.matched = matched

    def to_json(self):
        return {"type": "stop", "matched": self.matched}


class FINISH_MATCHED_STR:
    def __init__(self, matched):
        self.matched = matched

    def to_json(self):
        return {"type": "stop", "matched": self.matched}


class FINISH_LENGTH:
    def __init__(self, length):
        self.length = length

    def to_json(self):
        return {"type": "length", "length": self.length}


class FINISH_ABORT:
    def __init__(self, message="Unknown error"):
        self.message = message

    def to_json(self):
        return {"type": "abort", "message": self.message}


# ---------------------------------------------------------------------------
# FakeAsyncLLM — minimal stand-in for TokenSpeed's AsyncLLM in unit tests.
# ---------------------------------------------------------------------------


@dataclass
class _FakeState:
    finished: bool = False


@dataclass
class FakeAsyncLLM:
    """Implements just enough AsyncLLM surface to drive the servicer."""

    outputs: list[dict] = field(default_factory=list)
    is_generation: bool = True
    context_len: int = 8192
    max_req_input_len: int | None = 4096
    # Captured state — the servicer mutates/inspects these.
    rid_to_state: dict[str, _FakeState] = field(default_factory=dict)
    gracefully_exit: bool = False
    last_receive_tstamp: float = 0.0
    handle_loop_started: bool = False
    aborted_rids: list[str] = field(default_factory=list)
    # Override hook: a callable producing outputs per request, used for
    # tests that need dynamic yields (e.g. cancel mid-stream).
    generate_fn: Callable[[Any], Any] | None = None

    server_args: Any = field(
        default_factory=lambda: SimpleNamespace(
            model_path="fake-model",
            tokenizer_path="fake-model",
            served_model_name="fake-model",
            preferred_sampling_params=None,
        )
    )
    model_config: Any = field(
        default_factory=lambda: SimpleNamespace(
            vocab_size=32000,
            is_multimodal=False,
            hf_config=SimpleNamespace(
                eos_token_id=2,
                pad_token_id=0,
                bos_token_id=1,
                model_type="llama",
                architectures=["LlamaForCausalLM"],
            ),
        )
    )

    def auto_create_handle_loop(self) -> None:
        self.handle_loop_started = True

    def abort_request(self, rid: str) -> None:
        self.aborted_rids.append(rid)
        self.rid_to_state.pop(rid, None)

    async def generate_request(self, obj):
        # Record the request so tests can assert on what was forwarded.
        rid = getattr(obj, "rid", None) or "no-rid"
        self.rid_to_state[rid] = _FakeState()
        if self.generate_fn is not None:
            async for out in self.generate_fn(obj):
                self.last_receive_tstamp = 9999.0  # anything > tic
                yield out
            return
        for out in self.outputs:
            self.last_receive_tstamp = 9999.0
            yield out
        self.rid_to_state[rid].finished = True


@pytest.fixture
def fake_engine() -> FakeAsyncLLM:
    return FakeAsyncLLM()


@pytest.fixture
def servicer(fake_engine: FakeAsyncLLM) -> TokenSpeedSchedulerServicer:
    return TokenSpeedSchedulerServicer(
        async_llm=fake_engine,
        server_args=fake_engine.server_args,
        scheduler_info={
            "max_total_num_tokens": 100000,
            "max_req_input_len": 4096,
        },
    )


class _FakeAbortError(grpc.aio.AbortError):
    """Stand-in for grpc.aio.AbortError raised by our mock context.abort()."""

    def __init__(self, code: grpc.StatusCode, details: str):
        super().__init__()
        self.code = code
        self.details = details

    def __str__(self) -> str:  # makes pytest.raises(match=...) useful
        return f"ABORT({self.code.name}, {self.details})"


def _make_context() -> MagicMock:
    """Build a grpc.aio.ServicerContext whose ``abort()`` raises AbortError.

    Real gRPC servicer contexts raise ``grpc.aio.AbortError`` from
    ``context.abort()``. The servicer has a dedicated ``except
    grpc.aio.AbortError: raise`` branch to let that propagate cleanly, so
    the mock reproduces that behaviour.
    """
    ctx = MagicMock(spec=grpc.aio.ServicerContext)

    async def _abort(code, details):
        raise _FakeAbortError(code, details)

    ctx.abort = AsyncMock(side_effect=_abort)
    ctx.set_code = MagicMock()
    ctx.set_details = MagicMock()
    return ctx


# ---------------------------------------------------------------------------
# Pure-helper tests
# ---------------------------------------------------------------------------


class TestFinishReasonToDict:
    def test_none(self):
        assert _finish_reason_to_dict(None) is None

    def test_length(self):
        assert _finish_reason_to_dict(FINISH_LENGTH(length=42)) == {
            "type": "length",
            "length": 42,
        }

    def test_matched_token(self):
        assert _finish_reason_to_dict(FINISH_MATCHED_TOKEN(matched=7)) == {
            "type": "stop",
            "matched": 7,
        }

    def test_matched_str(self):
        assert _finish_reason_to_dict(FINISH_MATCHED_STR(matched="</s>")) == {
            "type": "stop",
            "matched": "</s>",
        }

    def test_abort(self):
        out = _finish_reason_to_dict(FINISH_ABORT(message="boom"))
        assert out["type"] == "abort"
        assert out["message"] == "boom"

    def test_passthrough_dict(self):
        d = {"type": "stop", "matched": "foo"}
        assert _finish_reason_to_dict(d) is d

    def test_unknown_falls_back_to_stop(self):
        # Any non-None non-dict non-BaseFinishReason value is coerced to a
        # "stop" dict with its str() as the ``matched`` field — defensive
        # behaviour so an unknown finish type never tears the stream down.
        assert _finish_reason_to_dict("weird") == {"type": "stop", "matched": "weird"}
        assert _finish_reason_to_dict(42) == {"type": "stop", "matched": "42"}


class TestAbortStatusCode:
    @pytest.mark.parametrize(
        "status_code, expected",
        [
            (400, grpc.StatusCode.INVALID_ARGUMENT),
            (408, grpc.StatusCode.DEADLINE_EXCEEDED),
            (504, grpc.StatusCode.DEADLINE_EXCEEDED),
            (429, grpc.StatusCode.RESOURCE_EXHAUSTED),
            (500, grpc.StatusCode.INTERNAL),
            (None, grpc.StatusCode.INTERNAL),
        ],
    )
    def test_mapping(self, status_code, expected):
        assert _abort_status_code({"status_code": status_code}) == expected


class TestMakeJsonSerializable:
    def test_primitives(self):
        assert _make_json_serializable(1) == 1
        assert _make_json_serializable("x") == "x"
        assert _make_json_serializable(True) is True
        assert _make_json_serializable(None) is None

    def test_list_tuple_set(self):
        assert _make_json_serializable([1, "a"]) == [1, "a"]
        assert _make_json_serializable((1, 2)) == [1, 2]
        assert _make_json_serializable({1, 2, 3}) in (
            [1, 2, 3],
            [1, 3, 2],
            [2, 1, 3],
            [2, 3, 1],
            [3, 1, 2],
            [3, 2, 1],
        )

    def test_nested_dict(self):
        assert _make_json_serializable({"a": [1, {"b": 2}]}) == {"a": [1, {"b": 2}]}

    def test_exotic_types_coerced_to_str(self):
        class Foo:
            def __str__(self):
                return "foo-str"

        assert _make_json_serializable(Foo()) == "foo-str"


# ---------------------------------------------------------------------------
# Sampling params conversion
# ---------------------------------------------------------------------------


class TestSamplingParamsConversion:
    def test_defaults_not_forwarded(self):
        params = sglang_scheduler_pb2.SamplingParams()
        out = TokenSpeedSchedulerServicer._sampling_params_from_proto(params)
        # proto3 defaults (0 / False / "") should not end up as TokenSpeed
        # overrides — only the always-forwarded bool fields appear.
        assert "temperature" not in out
        assert "top_p" not in out
        assert "top_k" not in out
        assert "max_new_tokens" not in out
        # always-forwarded bools
        assert out["skip_special_tokens"] is False
        assert out["spaces_between_special_tokens"] is False
        assert out["no_stop_trim"] is False
        assert out["ignore_eos"] is False

    def test_numeric_fields_forwarded(self):
        params = sglang_scheduler_pb2.SamplingParams(
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            min_p=0.05,
            frequency_penalty=0.1,
            presence_penalty=0.2,
            repetition_penalty=1.1,
            max_new_tokens=128,
            min_new_tokens=4,
        )
        out = TokenSpeedSchedulerServicer._sampling_params_from_proto(params)
        assert out["temperature"] == pytest.approx(0.7)
        assert out["top_p"] == pytest.approx(0.9)
        assert out["top_k"] == 50
        assert out["min_p"] == pytest.approx(0.05)
        assert out["frequency_penalty"] == pytest.approx(0.1)
        assert out["presence_penalty"] == pytest.approx(0.2)
        assert out["repetition_penalty"] == pytest.approx(1.1)
        assert out["max_new_tokens"] == 128
        assert out["min_new_tokens"] == 4

    def test_stop_lists_and_logit_bias(self):
        params = sglang_scheduler_pb2.SamplingParams(
            stop=["\n\n", "</s>"],
            stop_token_ids=[2, 0],
            logit_bias={"100": -10.0, "200": 10.0},
        )
        out = TokenSpeedSchedulerServicer._sampling_params_from_proto(params)
        assert out["stop"] == ["\n\n", "</s>"]
        assert out["stop_token_ids"] == [2, 0]
        assert out["logit_bias"] == {"100": -10.0, "200": 10.0}

    @pytest.mark.parametrize(
        "setter, key, value",
        [
            (lambda p: setattr(p, "regex", "a.*"), "regex", "a.*"),
            (lambda p: setattr(p, "json_schema", "{}"), "json_schema", "{}"),
            (lambda p: setattr(p, "ebnf_grammar", "g"), "ebnf", "g"),
            (lambda p: setattr(p, "structural_tag", "tag"), "structural_tag", "tag"),
        ],
    )
    def test_constraints(self, setter, key, value):
        params = sglang_scheduler_pb2.SamplingParams()
        setter(params)
        out = TokenSpeedSchedulerServicer._sampling_params_from_proto(params)
        assert out[key] == value


# ---------------------------------------------------------------------------
# Generate RPC
# ---------------------------------------------------------------------------


def _make_generate_request(
    *,
    request_id: str = "rid-1",
    input_ids: list[int] | None = None,
    stream: bool = False,
    max_new_tokens: int = 16,
) -> sglang_scheduler_pb2.GenerateRequest:
    return sglang_scheduler_pb2.GenerateRequest(
        request_id=request_id,
        tokenized=sglang_scheduler_pb2.TokenizedInput(
            # Preserve explicit empty-list inputs (for "rejects empty ids" test);
            # only fall back to the default if the caller didn't supply any.
            input_ids=(input_ids if input_ids is not None else [1, 2, 3, 4]),
            original_text="hello",
        ),
        sampling_params=sglang_scheduler_pb2.SamplingParams(
            temperature=0.0,
            max_new_tokens=max_new_tokens,
        ),
        stream=stream,
    )


class TestGenerate:
    @pytest.mark.asyncio
    async def test_non_streaming_emits_complete(
        self, fake_engine: FakeAsyncLLM, servicer: TokenSpeedSchedulerServicer
    ):
        # TokenSpeed's AsyncLLM includes the trailing matched-stop token in
        # ``output_ids`` (and prepends chat-template header tokens — modeled in
        # ``test_strips_chat_template_prefix`` below). The servicer normalizes
        # these out before the proto goes to the smg gateway so the tool
        # parsers see the same tokens they would from the SGLang path. Here we
        # check the matched-stop trim: ``raw=[10,11,12]`` with ``matched=12``
        # should arrive as ``[10,11]`` on the wire, and the matched id still
        # rides in the ``matched_token_id`` field.
        fake_engine.outputs = [
            {
                "text": "hi",
                "output_ids": [10, 11, 12],
                "meta_info": {
                    "prompt_tokens": 4,
                    "completion_tokens": 3,
                    "cached_tokens": 0,
                    "finish_reason": FINISH_MATCHED_TOKEN(matched=12),
                },
            }
        ]
        ctx = _make_context()
        req = _make_generate_request(stream=False)

        frames = [frame async for frame in servicer.Generate(req, ctx)]
        assert len(frames) == 1
        frame = frames[0]
        assert frame.request_id == "rid-1"
        assert frame.HasField("complete")
        complete = frame.complete
        assert list(complete.output_ids) == [10, 11]
        assert complete.finish_reason == "stop"
        assert complete.matched_token_id == 12
        assert complete.prompt_tokens == 4
        # Meta's completion_tokens passes through unchanged — matches SGLang's
        # ``meta_info.get("completion_tokens")`` convention — even though the
        # on-the-wire ``output_ids`` drops the stop token.
        assert complete.completion_tokens == 3
        ctx.abort.assert_not_called()

    @pytest.mark.asyncio
    async def test_strips_chat_template_prefix(
        self, fake_engine: FakeAsyncLLM, servicer: TokenSpeedSchedulerServicer
    ):
        """Reproducer for the bug where ``assistant\\n\\n`` leaked into the
        decoded text and broke the ``llama`` tool-call parser.

        Real-world capture on Llama-3.2-1B-Instruct with a function-calling
        prompt — ``output_ids`` was 27 tokens: 5 chat-template header tokens
        (``<|eot_id|>, <|start_header_id|>, "assistant", <|end_header_id|>,
        "\\n\\n"``) + 21 generated JSON tokens + 1 ``<|eom_id|>`` stop. With
        ``skip_special_tokens=True`` only the 128xxx control tokens get
        stripped at detokenization time, so the word token ``"assistant"``
        (78191) and ``"\\n\\n"`` (271) leaked into the text and flipped
        ``serde_json::from_str`` from succeeding on clean JSON to failing on
        ``assistant\\n\\n{...}``.

        The servicer now slices to the last ``completion_tokens`` tokens so
        downstream detokenization only sees the actual generated content.
        """
        fake_engine.outputs = [
            {
                "text": '{"name": "add", "parameters": {"a": 3, "b": 5}}',
                # Shape observed in the wild: [<|eot|>, <|start|>, "assistant",
                # <|end|>, "\n\n", ...21 json tokens, <|eom|>] = 27 tokens.
                # ``completion_tokens`` in TokenSpeed's meta covers the content
                # *plus* the stop token, so 21 + 1 = 22.
                "output_ids": [
                    128009,
                    128006,
                    78191,
                    128007,
                    271,
                    *range(9000, 9021),
                    128008,
                ],
                "meta_info": {
                    "prompt_tokens": 200,
                    "completion_tokens": 22,
                    "cached_tokens": 0,
                    "finish_reason": FINISH_MATCHED_TOKEN(matched=128008),
                },
            }
        ]
        ctx = _make_context()
        req = _make_generate_request(stream=False)

        frames = [frame async for frame in servicer.Generate(req, ctx)]
        complete = frames[0].complete
        # Header tokens dropped via the ``raw[-completion_tokens:]`` slice;
        # trailing stop token dropped because ``matched == token_ids[-1]``.
        assert list(complete.output_ids) == list(range(9000, 9021))
        assert complete.matched_token_id == 128008
        # meta_info.completion_tokens passes through; only ``output_ids`` is
        # normalized. Keeps the tokenspeed servicer's wire contract aligned
        # with the SGLang reference.
        assert complete.completion_tokens == 22

    @pytest.mark.asyncio
    async def test_streaming_emits_chunks_then_complete(
        self, fake_engine: FakeAsyncLLM, servicer: TokenSpeedSchedulerServicer
    ):
        fake_engine.outputs = [
            {
                "text": "hi",
                "output_ids": [10],  # delta chunk 1
                "meta_info": {
                    "prompt_tokens": 4,
                    "completion_tokens": 1,
                    "cached_tokens": 0,
                    "finish_reason": None,
                },
            },
            {
                "text": "hi there",
                "output_ids": [11, 12],  # delta chunk 2 + finish
                "meta_info": {
                    "prompt_tokens": 4,
                    "completion_tokens": 3,
                    "cached_tokens": 0,
                    "finish_reason": FINISH_LENGTH(length=16),
                },
            },
        ]
        ctx = _make_context()
        req = _make_generate_request(stream=True)

        frames = [frame async for frame in servicer.Generate(req, ctx)]
        # Expect: 2 chunks + 1 complete (emitted alongside the final chunk).
        # ``completion_tokens`` here (3) exceeds this chunk's delta length (2),
        # so the slice falls back to the raw delta. Length-finish has no
        # matched stop to strip either, so token_ids pass through.
        assert len(frames) == 3
        assert frames[0].HasField("chunk")
        assert list(frames[0].chunk.token_ids) == [10]
        assert frames[1].HasField("chunk")
        assert list(frames[1].chunk.token_ids) == [11, 12]
        assert frames[2].HasField("complete")
        assert frames[2].complete.finish_reason == "length"
        assert list(frames[2].complete.output_ids) == [11, 12]

    @pytest.mark.asyncio
    async def test_empty_input_ids_rejected(
        self, fake_engine: FakeAsyncLLM, servicer: TokenSpeedSchedulerServicer
    ):
        ctx = _make_context()
        req = _make_generate_request(input_ids=[])

        with pytest.raises(_FakeAbortError) as exc:
            async for _ in servicer.Generate(req, ctx):
                pass
        assert exc.value.code == grpc.StatusCode.INVALID_ARGUMENT
        ctx.abort.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_abort_finish_reason_surfaces_as_grpc_error(
        self, fake_engine: FakeAsyncLLM, servicer: TokenSpeedSchedulerServicer
    ):
        fake_engine.outputs = [
            {
                "text": "",
                "output_ids": [],
                "meta_info": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cached_tokens": 0,
                    "finish_reason": {
                        "type": "abort",
                        "message": "client disconnected",
                        "status_code": 400,
                    },
                },
            }
        ]
        ctx = _make_context()
        req = _make_generate_request()

        with pytest.raises(_FakeAbortError) as exc:
            async for _ in servicer.Generate(req, ctx):
                pass
        assert exc.value.code == grpc.StatusCode.INVALID_ARGUMENT

    @pytest.mark.asyncio
    async def test_cancel_calls_abort_request(
        self, fake_engine: FakeAsyncLLM, servicer: TokenSpeedSchedulerServicer
    ):
        """Cancelling the Generate task should tell the scheduler to drop the rid."""

        started = asyncio.Event()

        async def never_finish(_obj):
            started.set()
            # Block forever so we can cancel from outside. ``yield`` is
            # unreachable but keeps this an async generator.
            await asyncio.sleep(30)
            yield {}  # pragma: no cover

        fake_engine.generate_fn = never_finish
        ctx = _make_context()
        req = _make_generate_request()

        gen = servicer.Generate(req, ctx)
        task = asyncio.create_task(_drain(gen))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert "rid-1" in fake_engine.aborted_rids

    @pytest.mark.asyncio
    async def test_cancel_aborts_all_n_children(
        self, fake_engine: FakeAsyncLLM, servicer: TokenSpeedSchedulerServicer
    ):
        """n>1 expands rid to a list of per-choice ids; cancel must sweep them all.

        _build_generate_req rewrites ``rid`` to ``[rid-n0, rid-n1, ...]`` so
        TokenSpeed's batch path sees unique rids per choice. If Generate's
        cancel handler aborts only the original rid, the child scheduler
        requests keep consuming GPU work. This test guards that edge.
        """
        started = asyncio.Event()

        async def never_finish(_obj):
            started.set()
            await asyncio.sleep(30)
            yield {}  # pragma: no cover

        fake_engine.generate_fn = never_finish
        ctx = _make_context()
        req = _make_generate_request()
        req.sampling_params.n = 3

        gen = servicer.Generate(req, ctx)
        task = asyncio.create_task(_drain(gen))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # Every per-choice rid must have had abort_request called.
        assert set(fake_engine.aborted_rids) >= {"rid-1-n0", "rid-1-n1", "rid-1-n2"}


async def _drain(async_gen):
    async for _ in async_gen:
        pass


# ---------------------------------------------------------------------------
# Embed RPC
# ---------------------------------------------------------------------------


class TestEmbed:
    @pytest.mark.asyncio
    async def test_embed_ok(self, fake_engine: FakeAsyncLLM, servicer: TokenSpeedSchedulerServicer):
        fake_engine.is_generation = False
        fake_engine.outputs = [
            {
                "embedding": [0.1, 0.2, 0.3],
                "meta_info": {"prompt_tokens": 5},
            }
        ]
        ctx = _make_context()
        request = sglang_scheduler_pb2.EmbedRequest(
            request_id="e-1",
            tokenized=sglang_scheduler_pb2.TokenizedInput(
                input_ids=[4, 5, 6, 7, 8],
                original_text="x",
            ),
        )
        resp = await servicer.Embed(request, ctx)
        assert resp is not None
        assert list(resp.embedding) == pytest.approx([0.1, 0.2, 0.3])
        assert resp.embedding_dim == 3
        assert resp.prompt_tokens == 5

    @pytest.mark.asyncio
    async def test_embed_missing_tokenized_aborts(
        self, fake_engine: FakeAsyncLLM, servicer: TokenSpeedSchedulerServicer
    ):
        ctx = _make_context()
        request = sglang_scheduler_pb2.EmbedRequest(request_id="e-1")
        with pytest.raises(_FakeAbortError) as exc:
            await servicer.Embed(request, ctx)
        assert exc.value.code == grpc.StatusCode.INVALID_ARGUMENT


# ---------------------------------------------------------------------------
# Abort / HealthCheck / GetModelInfo / GetServerInfo / GetLoads
# ---------------------------------------------------------------------------


class TestAbortRpc:
    @pytest.mark.asyncio
    async def test_abort_known(
        self, fake_engine: FakeAsyncLLM, servicer: TokenSpeedSchedulerServicer
    ):
        fake_engine.rid_to_state["rid-1"] = _FakeState()
        resp = await servicer.Abort(
            sglang_scheduler_pb2.AbortRequest(request_id="rid-1"),
            _make_context(),
        )
        assert resp.success is True
        assert "rid-1" in fake_engine.aborted_rids

    @pytest.mark.asyncio
    async def test_abort_unknown(
        self, fake_engine: FakeAsyncLLM, servicer: TokenSpeedSchedulerServicer
    ):
        resp = await servicer.Abort(
            sglang_scheduler_pb2.AbortRequest(request_id="missing"),
            _make_context(),
        )
        assert resp.success is False
        # Nothing to abort — no state for "missing" or any "missing-n*" child.
        assert fake_engine.aborted_rids == []

    @pytest.mark.asyncio
    async def test_abort_sweeps_n_children(
        self, fake_engine: FakeAsyncLLM, servicer: TokenSpeedSchedulerServicer
    ):
        """Abort("rid-1") must sweep the per-choice rids Generate mints
        when ``sampling_params.n > 1`` (``rid-1-n0``, ``rid-1-n1``, ...).
        """
        for child in ("rid-1-n0", "rid-1-n1", "rid-1-n2"):
            fake_engine.rid_to_state[child] = _FakeState()
        # An unrelated rid the sweep must NOT touch.
        fake_engine.rid_to_state["unrelated-rid"] = _FakeState()

        resp = await servicer.Abort(
            sglang_scheduler_pb2.AbortRequest(request_id="rid-1"),
            _make_context(),
        )
        assert resp.success is True
        assert sorted(fake_engine.aborted_rids) == [
            "rid-1-n0",
            "rid-1-n1",
            "rid-1-n2",
        ]


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_reports_shutdown(
        self, fake_engine: FakeAsyncLLM, servicer: TokenSpeedSchedulerServicer
    ):
        fake_engine.gracefully_exit = True
        resp = await servicer.HealthCheck(
            sglang_scheduler_pb2.HealthCheckRequest(), _make_context()
        )
        assert resp.healthy is False
        assert "shutting down" in resp.message.lower()

    @pytest.mark.asyncio
    async def test_reports_healthy_when_scheduler_pushes_output(
        self, fake_engine: FakeAsyncLLM, servicer: TokenSpeedSchedulerServicer
    ):
        # generate_request yields once and updates last_receive_tstamp, which
        # is what the health RPC watches for.
        fake_engine.outputs = [
            {
                "text": "",
                "output_ids": [99],
                "meta_info": {"finish_reason": FINISH_LENGTH(length=1)},
            }
        ]
        resp = await servicer.HealthCheck(
            sglang_scheduler_pb2.HealthCheckRequest(), _make_context()
        )
        assert resp.healthy is True


class TestGetModelInfo:
    @pytest.mark.asyncio
    async def test_basic_fields(
        self, fake_engine: FakeAsyncLLM, servicer: TokenSpeedSchedulerServicer
    ):
        resp = await servicer.GetModelInfo(
            sglang_scheduler_pb2.GetModelInfoRequest(), _make_context()
        )
        assert resp.model_path == "fake-model"
        assert resp.is_generation is True
        assert resp.vocab_size == 32000
        assert resp.max_context_length == 8192
        assert list(resp.eos_token_ids) == [2]
        assert resp.model_type == "llama"
        assert list(resp.architectures) == ["LlamaForCausalLM"]


class TestGetServerInfo:
    @pytest.mark.asyncio
    async def test_returns_scheduler_info(
        self, fake_engine: FakeAsyncLLM, servicer: TokenSpeedSchedulerServicer
    ):
        fake_engine.rid_to_state["a"] = _FakeState()
        fake_engine.rid_to_state["b"] = _FakeState()
        resp = await servicer.GetServerInfo(
            sglang_scheduler_pb2.GetServerInfoRequest(), _make_context()
        )
        assert resp.active_requests == 2
        assert resp.server_type == "grpc"
        assert resp.max_total_num_tokens == 100000

    @pytest.mark.asyncio
    async def test_uses_tokenspeed_service_bases(self, servicer: TokenSpeedSchedulerServicer):
        """TokenSpeed's servicer inherits the dedicated
        ``TokenSpeedSchedulerServicer`` stub — identity is carried by the
        proto package/service name, not by a field inside ``server_args``.
        Guard the inheritance so nobody reverts to ``SglangSchedulerServicer``
        under the impression that 'wire shape is the same'; the wire shape
        is the same, the *service path* is not, and the Rust router routes
        on the service path.
        """
        from smg_grpc_proto.generated import tokenspeed_scheduler_pb2_grpc

        assert isinstance(servicer, tokenspeed_scheduler_pb2_grpc.TokenSpeedSchedulerServicer)


class TestGetLoads:
    @pytest.mark.asyncio
    async def test_stub_returns_empty(
        self, fake_engine: FakeAsyncLLM, servicer: TokenSpeedSchedulerServicer
    ):
        resp = await servicer.GetLoads(sglang_scheduler_pb2.GetLoadsRequest(), _make_context())
        assert resp.dp_rank_count == 0
        assert resp.version == "tokenspeed"


# ---------------------------------------------------------------------------
# _build_generate_req semantics (pre-tokenized input, disagg params)
# ---------------------------------------------------------------------------


class TestBuildGenerateReq:
    def test_preserves_input_ids(self, servicer: TokenSpeedSchedulerServicer):
        req = _make_generate_request(input_ids=[11, 22, 33], stream=True)
        obj = servicer._build_generate_req(req)
        assert obj.input_ids == [11, 22, 33]
        assert obj.rid == "rid-1"
        assert obj.stream is True
        assert obj.sampling_params["max_new_tokens"] == 16

    def test_disaggregated_params(self, servicer: TokenSpeedSchedulerServicer):
        req = _make_generate_request()
        req.disaggregated_params.bootstrap_host = "10.0.0.1"
        req.disaggregated_params.bootstrap_port = 23456
        req.disaggregated_params.bootstrap_room = 0  # valid room id even at 0
        obj = servicer._build_generate_req(req)
        assert obj.bootstrap_host == "10.0.0.1"
        assert obj.bootstrap_port == 23456
        assert obj.bootstrap_room == 0

    def test_rejects_missing_tokenized(self, servicer: TokenSpeedSchedulerServicer):
        req = sglang_scheduler_pb2.GenerateRequest(request_id="x")
        with pytest.raises(ValueError, match="tokenized"):
            servicer._build_generate_req(req)
