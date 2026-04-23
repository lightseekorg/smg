# mypy: ignore-errors
"""TokenSpeed gRPC servicer.

Implements the SGLang ``SglangScheduler`` gRPC service on top of TokenSpeed's
:class:`tokenspeed.runtime.engine.async_llm.AsyncLLM` — the main-process async
frontend that replaced ``TokenizerManager`` in the AsyncLLM refactor.

Why the SGLang proto?
---------------------
The Rust ``smg_router`` auto-detects each worker's runtime by probing the
gRPC health check for every known proto shape (SGLang → vLLM → TRT-LLM → MLX).
As soon as one handshake succeeds, that's the wire format the router uses.
Implementing the SGLang service means TokenSpeed workers slot into the
existing Rust pipeline unchanged — no new client, no new proto to maintain,
no schema drift between the TokenSpeed and SGLang surfaces.

The two backends already share the shape the servicer cares about
(``TokenizedGenerateReqInput``-style tokenized inputs, ``max_new_tokens``
naming on sampling params, ``{"text", "output_ids", "meta_info"}`` output
dicts), so the mapping is thin. See ``docs/architecture.md`` (TODO) for
the detailed field correspondence.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import logging
import os
import time
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import grpc
from google.protobuf.struct_pb2 import Struct
from google.protobuf.timestamp_pb2 import Timestamp
from smg_grpc_proto import sglang_scheduler_pb2, sglang_scheduler_pb2_grpc
from smg_grpc_proto.generated import common_pb2

from smg_grpc_servicer.tokenizer_bundle import CHUNK_SIZE, build_tokenizer_zip
from smg_grpc_servicer.tokenspeed.health_servicer import TokenSpeedHealthServicer

if TYPE_CHECKING:
    # Type-only imports — not resolved at module load so the servicer is
    # importable in test environments that stub AsyncLLM / ServerArgs.
    from tokenspeed.runtime.engine.async_llm import AsyncLLM
    from tokenspeed.runtime.server_args import ServerArgs

logger = logging.getLogger(__name__)

HEALTH_CHECK_TIMEOUT = int(os.getenv("TOKENSPEED_HEALTH_CHECK_TIMEOUT", "20"))


def _lazy_generate_req_input():
    """Late import for ``tokenspeed.runtime.engine.io_struct.GenerateReqInput``.

    Kept lazy so the top of this module loads in test environments that stub
    the TokenSpeed engine surface (unit tests don't need a fully-working
    TokenSpeed install to exercise proto ↔ request-input conversion).
    """
    from tokenspeed.runtime.engine.io_struct import GenerateReqInput

    return GenerateReqInput


def _lazy_embedding_req_input():
    """Late import for ``tokenspeed.runtime.engine.io_struct.EmbeddingReqInput``."""
    from tokenspeed.runtime.engine.io_struct import EmbeddingReqInput

    return EmbeddingReqInput


def _finish_reason_to_dict(reason: Any) -> dict | None:
    """Normalise a TokenSpeed finish reason into the SGLang on-wire shape.

    TokenSpeed emits ``BaseFinishReason``-style objects (or an already-normalised
    dict) in ``meta_info["finish_reason"]``; downstream code expects a dict
    with at minimum ``{"type": ...}`` and optionally ``{"matched": int|str}``.
    ``None`` means "still running".

    We duck-type on ``to_json()`` rather than importing the concrete
    ``BaseFinishReason`` class so the servicer module loads without pulling
    in TokenSpeed's full request-processing graph.
    """
    if reason is None:
        return None
    if isinstance(reason, dict):
        return reason
    to_json = getattr(reason, "to_json", None)
    if callable(to_json):
        try:
            result = to_json()
            if isinstance(result, dict):
                return result
        except Exception:  # noqa: BLE001
            logger.exception("Finish reason to_json() raised; falling back")
    # Unknown shape — coerce to string-type stop to avoid crashing the stream.
    return {"type": "stop", "matched": str(reason)}


class TokenSpeedSchedulerServicer(sglang_scheduler_pb2_grpc.SglangSchedulerServicer):
    """gRPC servicer exposing TokenSpeed's AsyncLLM over the SGLang proto."""

    def __init__(
        self,
        async_llm: AsyncLLM,
        server_args: ServerArgs,
        scheduler_info: dict,
        health_servicer: TokenSpeedHealthServicer | None = None,
    ):
        self.async_llm = async_llm
        self.server_args = server_args
        self.scheduler_info = scheduler_info
        self.health_servicer = health_servicer
        self.start_time = time.time()

        # Drive AsyncLLM's output-dispatch loop. This is idempotent — the
        # first caller creates the handle loop; subsequent callers (including
        # the HealthCheck RPC) are no-ops thanks to ``no_create_loop``.
        self.async_llm.auto_create_handle_loop()

        logger.info("TokenSpeedSchedulerServicer initialized")

    # ------------------------------------------------------------------
    # Generate (server-streaming)
    # ------------------------------------------------------------------

    async def Generate(
        self,
        request: sglang_scheduler_pb2.GenerateRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[sglang_scheduler_pb2.GenerateResponse]:
        rid = request.request_id
        logger.info("Generate request %s (stream=%s)", rid, request.stream)

        try:
            req_obj = self._build_generate_req(request)
        except ValueError as e:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
            return

        # For n>1, tokenspeed's batch handler generates fresh UUIDs per
        # sub-request and tags each streamed dict with a sequential
        # ``index`` (see tokenizer_manager.py::_handle_batch_request).
        # Non-streaming n>1 yields a *list* of final dicts instead. We
        # handle both shapes below.
        expanded_rid = getattr(req_obj, "rid", None)

        aborted = False
        try:
            async for output in self.async_llm.generate_request(req_obj):
                # Non-streaming n>1 emits a list of final dicts in one yield.
                if isinstance(output, list):
                    for idx, item in enumerate(output):
                        item_reason = _finish_reason_to_dict(
                            item.get("meta_info", {}).get("finish_reason")
                        )
                        if item_reason and item_reason.get("type") == "abort":
                            code = _abort_status_code(item_reason)
                            await context.abort(code, item_reason.get("message") or "aborted")
                            return
                        ci = int(item.get("index", idx))
                        yield self._complete_response(rid, item, item_reason, ci)
                    continue

                meta = output.get("meta_info", {})
                reason_dict = _finish_reason_to_dict(meta.get("finish_reason"))
                is_finished = reason_dict is not None

                if is_finished and reason_dict.get("type") == "abort":
                    code = _abort_status_code(reason_dict)
                    await context.abort(code, reason_dict.get("message") or "aborted")
                    return

                choice_index = int(output.get("index", 0))

                if request.stream:
                    yield self._chunk_response(rid, output, reason_dict, choice_index)
                    if is_finished:
                        yield self._complete_response(rid, output, reason_dict, choice_index)
                elif is_finished:
                    yield self._complete_response(rid, output, reason_dict, choice_index)

        except ValueError as e:
            logger.warning("Generate invalid request %s: %s", rid, e)
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except asyncio.CancelledError:
            # Client disconnected — sweep every scheduler-side rid we minted
            # (including the per-choice ``{rid}-n{i}`` children n>1 creates)
            # so abandoned requests don't keep consuming GPU work.
            aborted = True
            if isinstance(expanded_rid, list):
                for r in expanded_rid:
                    self.async_llm.abort_request(r)
            else:
                self.async_llm.abort_request(rid)
            raise
        except grpc.aio.AbortError:
            raise
        except Exception as e:
            logger.exception("Generate failed for request %s", rid)
            await context.abort(grpc.StatusCode.INTERNAL, str(e))
        finally:
            # Defensive cleanup — the scheduler owns rid_to_state, but if the
            # stream was torn down before finish we need to notify it. When
            # n>1 we expanded rid to a list of per-choice ids, so walk them.
            if not aborted:
                rids_to_check = (
                    list(expanded_rid)
                    if isinstance(expanded_rid, list)
                    else ([expanded_rid] if isinstance(expanded_rid, str) else [])
                )
                for r in rids_to_check:
                    state = self.async_llm.rid_to_state.get(r)
                    if state is not None and not getattr(state, "finished", False):
                        self.async_llm.abort_request(r)

    # ------------------------------------------------------------------
    # Embed (unary)
    # ------------------------------------------------------------------

    async def Embed(
        self,
        request: sglang_scheduler_pb2.EmbedRequest,
        context: grpc.aio.ServicerContext,
    ) -> sglang_scheduler_pb2.EmbedResponse:
        rid = request.request_id
        logger.info("Embed request %s", rid)

        if not request.HasField("tokenized"):
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "EmbedRequest requires tokenized input",
            )
            return

        EmbeddingReqInput = _lazy_embedding_req_input()
        obj = EmbeddingReqInput(
            input_ids=list(request.tokenized.input_ids),
        )
        obj.rid = rid
        # Preserve any original_text the router sent along so logs are useful.
        if request.tokenized.original_text:
            obj.text = request.tokenized.original_text

        aborted = False
        try:
            embedding: list[float] | None = None
            prompt_tokens = 0
            async for output in self.async_llm.generate_request(obj):
                # EmbeddingReqInput is non-streaming: the loop yields exactly
                # one dict at finish, carrying the embedding vector.
                embedding = output.get("embedding")
                prompt_tokens = output.get("meta_info", {}).get("prompt_tokens", 0)

            if embedding is None:
                await context.abort(grpc.StatusCode.INTERNAL, "Empty embedding result")
                return

            return sglang_scheduler_pb2.EmbedResponse(
                embedding=list(embedding),
                prompt_tokens=prompt_tokens,
                embedding_dim=len(embedding),
            )
        except asyncio.CancelledError:
            # Client disconnected mid-embedding — drop the request so its
            # rid doesn't leak in rid_to_state.
            aborted = True
            self.async_llm.abort_request(rid)
            raise
        except grpc.aio.AbortError:
            raise
        except ValueError as e:
            logger.warning("Embed invalid request %s: %s", rid, e)
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except Exception as e:
            logger.exception("Embed failed for request %s", rid)
            await context.abort(grpc.StatusCode.INTERNAL, str(e))
        finally:
            if not aborted:
                state = self.async_llm.rid_to_state.get(rid)
                if state is not None and not getattr(state, "finished", False):
                    self.async_llm.abort_request(rid)

    # ------------------------------------------------------------------
    # HealthCheck (unary)
    # ------------------------------------------------------------------

    async def HealthCheck(
        self,
        request: sglang_scheduler_pb2.HealthCheckRequest,
        context: grpc.aio.ServicerContext,
    ) -> sglang_scheduler_pb2.HealthCheckResponse:
        """Deep health probe — sends a 1-token generation to the scheduler.

        Mirrors SGLang's contract exactly: if the scheduler pushes *any*
        output within ``HEALTH_CHECK_TIMEOUT`` seconds, we consider it alive.
        We bypass the normal AsyncLLM lock/metrics by crafting a dedicated
        request with ``log_metrics=False`` so health checks don't skew
        Prometheus counters.
        """
        rid = f"HEALTH_CHECK_{time.time()}"

        if self.async_llm.gracefully_exit:
            return sglang_scheduler_pb2.HealthCheckResponse(
                healthy=False, message="Server is shutting down"
            )

        is_generation = bool(self.async_llm.is_generation)

        if is_generation:
            GenerateReqInput = _lazy_generate_req_input()
            probe = GenerateReqInput(
                input_ids=[0],
                sampling_params={"max_new_tokens": 1, "temperature": 0.0},
                log_metrics=False,
            )
        else:
            EmbeddingReqInput = _lazy_embedding_req_input()
            probe = EmbeddingReqInput(input_ids=[0])
            probe.log_metrics = False
        probe.rid = rid

        tic = time.time()

        async def _drive_probe() -> bool:
            try:
                async for _ in self.async_llm.generate_request(probe):
                    return True
            except Exception as e:  # noqa: BLE001 — the probe is best-effort.
                logger.warning("Health probe failed: %s", e)
                return False
            return False

        task = asyncio.create_task(_drive_probe())
        try:
            while time.time() - tic < HEALTH_CHECK_TIMEOUT:
                await asyncio.sleep(0.5)
                # Any scheduler push after we started counts as healthy.
                if self.async_llm.last_receive_tstamp > tic:
                    return sglang_scheduler_pb2.HealthCheckResponse(
                        healthy=True,
                        message="Health check passed",
                    )
                if task.done():
                    return sglang_scheduler_pb2.HealthCheckResponse(
                        healthy=bool(task.result()),
                        message=(
                            "Health check passed"
                            if task.result()
                            else "Scheduler returned no output"
                        ),
                    )
        finally:
            if not task.done():
                task.cancel()
            # Best-effort cleanup: the probe rid shouldn't linger.
            self.async_llm.abort_request(rid)

        return sglang_scheduler_pb2.HealthCheckResponse(
            healthy=False,
            message=f"Health check timeout after {HEALTH_CHECK_TIMEOUT}s",
        )

    # ------------------------------------------------------------------
    # Abort (unary)
    # ------------------------------------------------------------------

    async def Abort(
        self,
        request: sglang_scheduler_pb2.AbortRequest,
        _context: grpc.aio.ServicerContext,
    ) -> sglang_scheduler_pb2.AbortResponse:
        """Abort the request + any per-choice expansions from n>1.

        Generate rewrites ``n>1`` requests into a list of rids
        ``[{request_id}-n0, {request_id}-n1, ...]`` so TokenSpeed's batch
        path sees unique rids. Aborting only the original ``request_id``
        would leave those children running — we sweep them all.
        """
        rid = request.request_id
        logger.info("Abort request %s", rid)
        state_map = self.async_llm.rid_to_state

        # Any rid starting with ``{request_id}-n`` is a per-choice child we
        # minted in _build_generate_req; catch the single-rid case too.
        child_prefix = f"{rid}-n"
        targets = [r for r in state_map if r == rid or r.startswith(child_prefix)]

        try:
            for r in targets:
                self.async_llm.abort_request(r)
            known = bool(targets)
            return sglang_scheduler_pb2.AbortResponse(
                success=known,
                message=(
                    f"Aborted {len(targets)} request(s) for {rid}"
                    if known
                    else f"Request {rid} not found"
                ),
            )
        except Exception as e:
            logger.exception("Abort failed for %s", rid)
            return sglang_scheduler_pb2.AbortResponse(success=False, message=str(e))

    # ------------------------------------------------------------------
    # GetModelInfo (unary)
    # ------------------------------------------------------------------

    async def GetModelInfo(
        self,
        _request: sglang_scheduler_pb2.GetModelInfoRequest,
        _context: grpc.aio.ServicerContext,
    ) -> sglang_scheduler_pb2.GetModelInfoResponse:
        model_config = self.async_llm.model_config
        hf_config = getattr(model_config, "hf_config", None)

        eos = getattr(hf_config, "eos_token_id", None) if hf_config else None
        if isinstance(eos, int):
            eos_token_ids = [eos]
        elif isinstance(eos, list):
            eos_token_ids = list(eos)
        else:
            eos_token_ids = []

        max_req_input_len = self.scheduler_info.get("max_req_input_len") or (
            self.async_llm.max_req_input_len or 0
        )

        return sglang_scheduler_pb2.GetModelInfoResponse(
            model_path=self.server_args.model_path,
            tokenizer_path=self.server_args.tokenizer_path or "",
            is_generation=bool(self.async_llm.is_generation),
            preferred_sampling_params=self.server_args.preferred_sampling_params or "",
            weight_version="",
            served_model_name=(self.server_args.served_model_name or self.server_args.model_path),
            max_context_length=int(self.async_llm.context_len),
            vocab_size=int(model_config.vocab_size),
            supports_vision=bool(getattr(model_config, "is_multimodal", False)),
            model_type=(getattr(hf_config, "model_type", "") or "") if hf_config else "",
            architectures=(getattr(hf_config, "architectures", []) or []) if hf_config else [],
            eos_token_ids=eos_token_ids,
            pad_token_id=(getattr(hf_config, "pad_token_id", 0) or 0) if hf_config else 0,
            bos_token_id=(getattr(hf_config, "bos_token_id", 0) or 0) if hf_config else 0,
            max_req_input_len=int(max_req_input_len),
        )

    # ------------------------------------------------------------------
    # GetServerInfo (unary)
    # ------------------------------------------------------------------

    async def GetServerInfo(
        self,
        _request: sglang_scheduler_pb2.GetServerInfoRequest,
        _context: grpc.aio.ServicerContext,
    ) -> sglang_scheduler_pb2.GetServerInfoResponse:
        # TokenSpeed's ``ServerArgs`` is a dataclass, but tests sometimes pass
        # a plain namespace. Fall back to ``__dict__`` so both shapes work.
        if dataclasses.is_dataclass(self.server_args) and not isinstance(self.server_args, type):
            server_args_dict = dataclasses.asdict(self.server_args)
        else:
            server_args_dict = dict(getattr(self.server_args, "__dict__", {}))
        server_args_struct = Struct()
        server_args_struct.update(_make_json_serializable(server_args_dict))

        scheduler_info_struct = Struct()
        scheduler_info_struct.update(_make_json_serializable(dict(self.scheduler_info)))

        uptime = time.time() - self.start_time
        start_timestamp = Timestamp()
        start_timestamp.FromSeconds(int(self.start_time))

        try:
            import tokenspeed  # local import: avoid module-load-time dependency

            version = getattr(tokenspeed, "__version__", "unknown")
        except Exception:  # noqa: BLE001 — fall back gracefully.
            version = "unknown"

        return sglang_scheduler_pb2.GetServerInfoResponse(
            server_args=server_args_struct,
            scheduler_info=scheduler_info_struct,
            active_requests=len(self.async_llm.rid_to_state),
            is_paused=False,
            last_receive_timestamp=float(self.async_llm.last_receive_tstamp),
            uptime_seconds=float(uptime),
            sglang_version=version,  # proto field name — content is tokenspeed's
            server_type="grpc",
            start_time=start_timestamp,
            max_total_num_tokens=int(self.scheduler_info.get("max_total_num_tokens", 0)),
        )

    # ------------------------------------------------------------------
    # GetLoads (unary) — minimal parity stub
    # ------------------------------------------------------------------

    async def GetLoads(
        self,
        _request: sglang_scheduler_pb2.GetLoadsRequest,
        _context: grpc.aio.ServicerContext,
    ) -> sglang_scheduler_pb2.GetLoadsResponse:
        # TokenSpeed doesn't yet expose a get-loads communicator; return an
        # empty response that still round-trips through the SGLang client's
        # load-metrics aggregator without breaking downstream label extraction.
        return sglang_scheduler_pb2.GetLoadsResponse(
            timestamp=datetime.now(timezone.utc).isoformat(),
            version="tokenspeed",
            dp_rank_count=0,
        )

    # ------------------------------------------------------------------
    # GetTokenizer (server-streaming)
    # ------------------------------------------------------------------

    async def GetTokenizer(
        self,
        _request: common_pb2.GetTokenizerRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[common_pb2.GetTokenizerChunk]:
        tokenizer_path = self.server_args.tokenizer_path or self.server_args.model_path
        if not tokenizer_path:
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "Tokenizer path is not configured on this server.",
            )
            return

        try:
            zip_buffer = build_tokenizer_zip(Path(tokenizer_path))
        except Exception as e:  # noqa: BLE001 — surface as gRPC INTERNAL.
            logger.exception("Failed to build tokenizer ZIP")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))
            return

        zip_data = zip_buffer.getbuffer()
        sha256 = hashlib.sha256(zip_data).hexdigest()

        offset = 0
        total = len(zip_data)
        while offset < total:
            end = min(offset + CHUNK_SIZE, total)
            is_last = end == total
            yield common_pb2.GetTokenizerChunk(
                data=bytes(zip_data[offset:end]),
                sha256=sha256 if is_last else "",
            )
            offset = end

    # ------------------------------------------------------------------
    # SubscribeKvEvents — not supported in this runtime
    # ------------------------------------------------------------------

    async def SubscribeKvEvents(
        self,
        _request: common_pb2.SubscribeKvEventsRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[common_pb2.KvEventBatch]:
        await context.abort(
            grpc.StatusCode.UNIMPLEMENTED,
            "KV cache events are not exposed by the TokenSpeed gRPC backend.",
        )
        # Required for the async-generator contract, even after abort.
        return  # noqa: B901 — intentional
        yield  # pragma: no cover

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Graceful shutdown hook — drain AsyncLLM's sigterm watchdog."""
        self.async_llm.gracefully_exit = True
        if self.health_servicer:
            self.health_servicer.set_not_serving()

    def _build_generate_req(self, request: sglang_scheduler_pb2.GenerateRequest):
        """Translate proto GenerateRequest → TokenSpeed GenerateReqInput.

        Keeps the router's pre-tokenized inputs intact (``input_ids`` set,
        ``text`` left blank) so the TokenSpeed InputProcessor skips its own
        tokenizer pass.
        """
        if not request.HasField("tokenized"):
            raise ValueError("GenerateRequest.tokenized is required")

        input_ids = list(request.tokenized.input_ids)
        if not input_ids:
            raise ValueError("GenerateRequest.tokenized.input_ids is empty")

        sampling = self._sampling_params_from_proto(request.sampling_params)

        GenerateReqInput = _lazy_generate_req_input()
        obj = GenerateReqInput(
            input_ids=input_ids,
            sampling_params=sampling,
            stream=bool(request.stream),
            return_logprob=bool(request.return_logprob),
            logprob_start_len=(
                request.logprob_start_len if request.logprob_start_len is not None else -1
            ),
            top_logprobs_num=int(request.top_logprobs_num or 0),
            token_ids_logprob=(
                list(request.token_ids_logprob) if request.token_ids_logprob else None
            ),
            return_hidden_states=bool(request.return_hidden_states),
            # ``log_metrics`` on the proto is a plain bool3 scalar — there's
            # no unset/zero-default distinction. Leaving tokenspeed's default
            # (True) in place matches SGLang's behaviour where the router
            # never opts out of metrics at this layer.
        )
        # Older tokenspeed's ``normalize_batch_and_arguments`` treats n>1 as
        # batched and asserts ``rid`` is a list in that case. One gRPC
        # request carries one rid; expand it to a list of deterministic
        # per-choice rids when the caller asked for multiple samples so the
        # assert doesn't fire (and the scheduler can still deduplicate).
        n = sampling.get("n", 1) or 1
        if n > 1:
            obj.rid = [f"{request.request_id}-n{i}" for i in range(n)]
        else:
            obj.rid = request.request_id

        # Preserve original_text for logging if present (purely cosmetic;
        # tokenization is skipped because input_ids is set).
        if request.tokenized.original_text:
            obj.text = request.tokenized.original_text

        if request.HasField("disaggregated_params"):
            p = request.disaggregated_params
            obj.bootstrap_host = p.bootstrap_host or None
            obj.bootstrap_port = p.bootstrap_port or None
            obj.bootstrap_room = p.bootstrap_room  # 0 is a valid room id

        return obj

    @staticmethod
    def _sampling_params_from_proto(
        params: sglang_scheduler_pb2.SamplingParams,
    ) -> dict[str, Any]:
        """Build the dict that ``GenerateReqInput.sampling_params`` expects.

        TokenSpeed's :class:`SamplingParams` consumes this dict via
        ``SamplingParams(**obj.sampling_params)``, so field names must match
        the Python class (``max_new_tokens``, ``stop``, ``stop_token_ids``, ...).
        """
        out: dict[str, Any] = {}

        # Proto3 scalars (temperature, top_p, ...) are always present with a
        # zero-ish default when the client didn't send them. Treat the
        # zero-ish default as "use TokenSpeed's own default" and only forward
        # fields the client explicitly set. ``max_new_tokens`` and
        # ``stream_interval`` are ``optional`` in the proto, so we can use
        # HasField() for true presence tracking.
        if params.HasField("max_new_tokens"):
            out["max_new_tokens"] = params.max_new_tokens
        if params.temperature:
            out["temperature"] = params.temperature
        if params.top_p:
            out["top_p"] = params.top_p
        # top_k = 0 is invalid for TokenSpeed (treated as "disable").
        # Only forward if positive.
        if params.top_k:
            out["top_k"] = params.top_k
        if params.min_p:
            out["min_p"] = params.min_p
        if params.frequency_penalty:
            out["frequency_penalty"] = params.frequency_penalty
        if params.presence_penalty:
            out["presence_penalty"] = params.presence_penalty
        if params.repetition_penalty:
            out["repetition_penalty"] = params.repetition_penalty
        if params.min_new_tokens:
            out["min_new_tokens"] = params.min_new_tokens

        # Lists
        if params.stop:
            out["stop"] = list(params.stop)
        if params.stop_token_ids:
            out["stop_token_ids"] = list(params.stop_token_ids)

        # Bools (always forwarded — matches SGLang)
        out["skip_special_tokens"] = bool(params.skip_special_tokens)
        out["spaces_between_special_tokens"] = bool(params.spaces_between_special_tokens)
        out["no_stop_trim"] = bool(params.no_stop_trim)
        out["ignore_eos"] = bool(params.ignore_eos)

        # Optional: n (OpenAI-compat, passthrough)
        if params.n:
            out["n"] = params.n
        if params.HasField("stream_interval"):
            out["stream_interval"] = params.stream_interval
        if params.logit_bias:
            out["logit_bias"] = dict(params.logit_bias)

        # Constraint types — exactly one may be set.
        if params.HasField("regex"):
            out["regex"] = params.regex
        elif params.HasField("json_schema"):
            out["json_schema"] = params.json_schema
        elif params.HasField("ebnf_grammar"):
            out["ebnf"] = params.ebnf_grammar
        elif params.HasField("structural_tag"):
            out["structural_tag"] = params.structural_tag

        return out

    def _generated_output_ids(self, output: dict, reason_dict: dict | None) -> list[int]:
        """Return just the newly-generated tokens from a TokenSpeed output dict.

        TokenSpeed's AsyncLLM has two quirks that the SGLang gRPC proto contract
        doesn't expect, both of which break the smg gateway's detokenization
        layer and downstream tool-call parsing:

        1. ``output_ids`` is prefixed with the Llama-3 chat-template assistant
           header: ``[<|eot_id|>, <|start_header_id|>, "assistant",
           <|end_header_id|>, "\\n\\n", ...generated..., <stop>]``. The
           ``skip_special_tokens=True`` detokenization strips the 128xxx
           control tokens but keeps the word tokens ``"assistant"`` (78191)
           and ``"\\n\\n"`` (271), so the final text looks like
           ``assistant\\n\\n{"name": ...}``. The ``llama`` tool parser's
           ``serde_json::from_str`` can't handle leading non-JSON prefix and
           silently returns zero tool calls.
        2. The trailing stop token (e.g. ``<|eom_id|>`` = 128008) is included
           in ``output_ids``; SGLang excludes it. If the gateway ever runs
           with ``skip_special_tokens=False`` the stop leaks into the decoded
           text and breaks JSON parsing for the same reason.

        Slicing the last ``meta_info.completion_tokens`` tokens gives us the
        bare generated sequence that SGLang's ``token_ids`` would carry, and
        we then defensively drop any trailing matched stop token. The
        per-choice ``matched_stop`` fires in a separate proto field, so no
        information is lost.
        """
        raw = list(output.get("output_ids") or [])
        if not raw:
            return raw
        completion = output.get("meta_info", {}).get("completion_tokens")
        if isinstance(completion, int) and 0 < completion <= len(raw):
            token_ids = raw[-completion:]
        else:
            token_ids = raw
        if reason_dict and reason_dict.get("type") == "stop":
            matched = reason_dict.get("matched")
            if isinstance(matched, int) and token_ids and token_ids[-1] == matched:
                token_ids = token_ids[:-1]
        return token_ids

    def _chunk_response(
        self,
        rid: str,
        output: dict,
        reason_dict: dict | None,
        choice_index: int = 0,
    ) -> sglang_scheduler_pb2.GenerateResponse:
        meta = output.get("meta_info", {})
        token_ids = self._generated_output_ids(output, reason_dict)
        return sglang_scheduler_pb2.GenerateResponse(
            request_id=rid,
            chunk=sglang_scheduler_pb2.GenerateStreamChunk(
                token_ids=token_ids,
                prompt_tokens=int(meta.get("prompt_tokens", 0)),
                completion_tokens=int(meta.get("completion_tokens", len(token_ids))),
                cached_tokens=int(meta.get("cached_tokens", 0)),
                index=choice_index,
            ),
        )

    def _complete_response(
        self,
        rid: str,
        output: dict,
        reason_dict: dict | None,
        choice_index: int = 0,
    ) -> sglang_scheduler_pb2.GenerateResponse:
        meta = output.get("meta_info", {})
        token_ids = self._generated_output_ids(output, reason_dict)

        finish_reason = "stop"
        matched_kwargs: dict[str, Any] = {}
        if reason_dict:
            kind = reason_dict.get("type")
            if kind == "length":
                finish_reason = "length"
            elif kind == "abort":
                finish_reason = "abort"
            matched = reason_dict.get("matched")
            if isinstance(matched, int):
                matched_kwargs["matched_token_id"] = matched
            elif isinstance(matched, str):
                matched_kwargs["matched_stop_str"] = matched

        return sglang_scheduler_pb2.GenerateResponse(
            request_id=rid,
            complete=sglang_scheduler_pb2.GenerateComplete(
                output_ids=token_ids,
                finish_reason=finish_reason,
                prompt_tokens=int(meta.get("prompt_tokens", 0)),
                completion_tokens=int(meta.get("completion_tokens", len(token_ids))),
                cached_tokens=int(meta.get("cached_tokens", 0)),
                index=choice_index,
                **matched_kwargs,
            ),
        )


def _abort_status_code(reason: dict) -> grpc.StatusCode:
    status_code = reason.get("status_code")
    if status_code == 400:
        return grpc.StatusCode.INVALID_ARGUMENT
    if status_code in (408, 504):
        return grpc.StatusCode.DEADLINE_EXCEEDED
    if status_code == 429:
        return grpc.StatusCode.RESOURCE_EXHAUSTED
    return grpc.StatusCode.INTERNAL


def _make_json_serializable(obj: Any) -> Any:
    """Flatten an arbitrary dataclass/config graph into JSON-safe primitives."""
    if obj is None or isinstance(obj, str | int | float | bool):
        return obj
    if isinstance(obj, list | tuple | set):
        return [_make_json_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}
    return str(obj)
