"""TokenSpeed gRPC servicer.

Implements the ``tokenspeed.grpc.scheduler.TokenSpeedScheduler`` gRPC service
on top of TokenSpeed's :class:`tokenspeed.runtime.engine.async_llm.AsyncLLM` —
the main-process async frontend that replaced ``TokenizerManager`` in the
AsyncLLM refactor.

Wire identity & message catalog
-------------------------------
TokenSpeed ships a fully independent proto (``proto/tokenspeed_scheduler.proto``)
with a distinct package, service, and message catalog. The Rust gateway's
``DetectBackendStep`` identifies the worker natively from the service name —
no SGLang-look-alike hack, no runtime marker probe. The proto's field set is
intentionally minimal (top-tier LLM serving only): no Embed, no
GetTokenizer, no SubscribeKvEvents, no multimodal, no PD-disaggregated
serving, no LoRA, no hidden-state forwarding, no classifier outputs.
Anything in that list has to be added to the proto first; it doesn't ride
on a shared SGLang message anymore.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import time
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import grpc
from google.protobuf.struct_pb2 import Struct
from google.protobuf.timestamp_pb2 import Timestamp
from smg_grpc_proto import tokenspeed_scheduler_pb2_grpc
from smg_grpc_proto.generated import tokenspeed_scheduler_pb2

from smg_grpc_servicer.tokenspeed.health_servicer import TokenSpeedHealthServicer

if TYPE_CHECKING:
    # Type-only imports — not resolved at module load so the servicer is
    # importable in test environments that stub AsyncLLM / ServerArgs.
    from tokenspeed.runtime.engine.async_llm import AsyncLLM
    from tokenspeed.runtime.utils.server_args import ServerArgs

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


def _finish_reason_to_dict(reason: Any) -> dict | None:
    """Normalise a TokenSpeed finish reason into the SGLang on-wire shape.

    TokenSpeed emits ``BaseFinishReason``-style objects (or an already-normalised
    dict) in ``meta_info["finish_reason"]``; downstream code expects a dict
    with at minimum ``{"type": ...}`` and optionally ``{"matched": int|str}``.
    ``None`` means "still running".

    We duck-type on ``to_json()`` rather than importing the concrete
    ``BaseFinishReason`` class so the servicer module loads without pulling
    in TokenSpeed's full request-processing graph.

    Raises ``TypeError`` for unknown shapes rather than coercing to a fake
    ``stop``: silently flipping ``length``/``abort`` to ``stop`` and leaking
    a debug ``repr()`` into the user-facing ``matched_stop_str`` field would
    hide real bugs and corrupt the OpenAI ``finish_reason`` semantics. The
    caller wraps this in ``try/except`` and turns it into ``StatusCode.INTERNAL``.
    """
    if reason is None:
        return None
    if isinstance(reason, dict):
        return reason
    to_json = getattr(reason, "to_json", None)
    if callable(to_json):
        try:
            result = to_json()
        except Exception as e:  # noqa: BLE001
            raise TypeError(
                f"finish_reason of type {type(reason).__name__!r} raised in "
                f"to_json(); refusing to silently emit a fake stop. {e}"
            ) from e
        if isinstance(result, dict):
            return result
        raise TypeError(
            f"finish_reason {type(reason).__name__!r}.to_json() returned "
            f"{type(result).__name__!r}; expected dict with at least 'type'."
        )
    raise TypeError(
        f"Unknown finish_reason shape {type(reason).__name__!r}; expected "
        f"a dict or an object with a to_json() method."
    )


class TokenSpeedSchedulerServicer(tokenspeed_scheduler_pb2_grpc.TokenSpeedSchedulerServicer):
    """gRPC servicer exposing TokenSpeed's AsyncLLM over the dedicated TokenSpeed proto."""

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
        request: tokenspeed_scheduler_pb2.GenerateRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[tokenspeed_scheduler_pb2.GenerateResponse]:
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

                if reason_dict is not None and reason_dict.get("type") == "abort":
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
    # HealthCheck (unary)
    # ------------------------------------------------------------------

    async def HealthCheck(
        self,
        request: tokenspeed_scheduler_pb2.HealthCheckRequest,
        context: grpc.aio.ServicerContext,
    ) -> tokenspeed_scheduler_pb2.HealthCheckResponse:
        """Deep health probe — sends a 1-token generation to the scheduler.

        Mirrors SGLang's contract exactly: if the scheduler pushes *any*
        output within ``HEALTH_CHECK_TIMEOUT`` seconds, we consider it alive.
        We bypass the normal AsyncLLM lock/metrics by crafting a dedicated
        request with ``log_metrics=False`` so health checks don't skew
        Prometheus counters.
        """
        rid = f"HEALTH_CHECK_{time.time()}"

        if self.async_llm.gracefully_exit:
            return tokenspeed_scheduler_pb2.HealthCheckResponse(
                healthy=False, message="Server is shutting down"
            )

        # TokenSpeed only serves generative LLMs at this layer (the proto
        # has no Embed RPC), so the probe is always a 1-token generate.
        GenerateReqInput = _lazy_generate_req_input()
        probe = GenerateReqInput(
            input_ids=[0],
            sampling_params={"max_new_tokens": 1, "temperature": 0.0},
            log_metrics=False,
        )
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
                    return tokenspeed_scheduler_pb2.HealthCheckResponse(
                        healthy=True,
                        message="Health check passed",
                    )
                if task.done():
                    return tokenspeed_scheduler_pb2.HealthCheckResponse(
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

        return tokenspeed_scheduler_pb2.HealthCheckResponse(
            healthy=False,
            message=f"Health check timeout after {HEALTH_CHECK_TIMEOUT}s",
        )

    # ------------------------------------------------------------------
    # Abort (unary)
    # ------------------------------------------------------------------

    async def Abort(
        self,
        request: tokenspeed_scheduler_pb2.AbortRequest,
        _context: grpc.aio.ServicerContext,
    ) -> tokenspeed_scheduler_pb2.AbortResponse:
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
            return tokenspeed_scheduler_pb2.AbortResponse(
                success=known,
                message=(
                    f"Aborted {len(targets)} request(s) for {rid}"
                    if known
                    else f"Request {rid} not found"
                ),
            )
        except Exception as e:
            logger.exception("Abort failed for %s", rid)
            return tokenspeed_scheduler_pb2.AbortResponse(success=False, message=str(e))

    # ------------------------------------------------------------------
    # GetModelInfo (unary)
    # ------------------------------------------------------------------

    async def GetModelInfo(
        self,
        _request: tokenspeed_scheduler_pb2.GetModelInfoRequest,
        _context: grpc.aio.ServicerContext,
    ) -> tokenspeed_scheduler_pb2.GetModelInfoResponse:
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

        # TokenSpeed's GetModelInfoResponse intentionally drops
        # ``is_generation`` (always true), ``supports_vision`` (always false),
        # and ``id2label_json`` / ``num_labels`` (not a classifier serving
        # path). The Rust client fills those slots back in when translating
        # to its SGLang-shaped wrapper.
        return tokenspeed_scheduler_pb2.GetModelInfoResponse(
            model_path=self.server_args.model_path,
            tokenizer_path=self.server_args.tokenizer_path or "",
            preferred_sampling_params=self.server_args.preferred_sampling_params or "",
            weight_version="",
            served_model_name=(self.server_args.served_model_name or self.server_args.model_path),
            max_context_length=int(self.async_llm.context_len),
            vocab_size=int(model_config.vocab_size),
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
        _request: tokenspeed_scheduler_pb2.GetServerInfoRequest,
        _context: grpc.aio.ServicerContext,
    ) -> tokenspeed_scheduler_pb2.GetServerInfoResponse:
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

        return tokenspeed_scheduler_pb2.GetServerInfoResponse(
            server_args=server_args_struct,
            scheduler_info=scheduler_info_struct,
            active_requests=len(self.async_llm.rid_to_state),
            is_paused=False,
            uptime_seconds=float(uptime),
            tokenspeed_version=version,
            start_time=start_timestamp,
            max_total_num_tokens=int(self.scheduler_info.get("max_total_num_tokens", 0)),
        )

    # ------------------------------------------------------------------
    # GetLoads (unary) — minimal parity stub
    # ------------------------------------------------------------------

    async def GetLoads(
        self,
        _request: tokenspeed_scheduler_pb2.GetLoadsRequest,
        _context: grpc.aio.ServicerContext,
    ) -> tokenspeed_scheduler_pb2.GetLoadsResponse:
        # TokenSpeed doesn't yet expose a get-loads communicator; return an
        # empty response that still round-trips through the SGLang client's
        # load-metrics aggregator without breaking downstream label extraction.
        return tokenspeed_scheduler_pb2.GetLoadsResponse(
            timestamp=datetime.now(timezone.utc).isoformat(),
            version="tokenspeed",
            dp_rank_count=0,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def shutdown(self, drain_timeout_secs: float = 30.0) -> None:
        """Graceful shutdown — drain in-flight requests, then kill scheduler children.

        AsyncLLM's ``sigterm_watchdog`` polls ``gracefully_exit`` every 5s,
        drains ``rid_to_state`` and finally calls
        ``kill_process_tree(getpid, include_parent=True)``. That works in
        steady-state but the gRPC server's main coroutine may unwind before
        the watchdog ticks again, in which case the scheduler subprocesses
        outlive the parent and end up orphaned. To avoid that, we:

        1. Flag ``gracefully_exit`` so AsyncLLM stops accepting work and
           the watchdog will eventually run its own cleanup.
        2. Wait up to ``drain_timeout_secs`` for ``rid_to_state`` to empty.
        3. Forcibly kill the subprocess tree (``include_parent=False``) so
           the scheduler children are reaped regardless of whether the
           watchdog tick fires before this coroutine returns. Idempotent
           with the watchdog's own ``kill_process_tree`` call.
        """
        self.async_llm.gracefully_exit = True
        if self.health_servicer:
            self.health_servicer.set_not_serving()

        deadline = time.monotonic() + drain_timeout_secs
        while time.monotonic() < deadline:
            if not getattr(self.async_llm, "rid_to_state", None):
                break
            await asyncio.sleep(0.5)
        else:
            logger.warning(
                "shutdown drain timed out after %.1fs with %d in-flight requests; "
                "killing scheduler children anyway",
                drain_timeout_secs,
                len(getattr(self.async_llm, "rid_to_state", {}) or {}),
            )

        # Reap the scheduler subprocesses without taking down our own PID;
        # server.py's stop sequence still needs us alive to finish gRPC drain.
        try:
            from tokenspeed.runtime.utils.process import kill_process_tree
        except ImportError:
            logger.exception(
                "Could not import tokenspeed.runtime.utils.process.kill_process_tree; "
                "scheduler subprocesses may be orphaned"
            )
            return
        kill_process_tree(os.getpid(), include_parent=False)

    def _build_generate_req(self, request: tokenspeed_scheduler_pb2.GenerateRequest):
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
            # ``logprob_start_len`` is ``optional int32`` on the wire — use
            # presence-tracking, not the proto3 zero-default, to distinguish
            # "client omitted" (→ SGLang's ``-1`` = no input logprobs) from
            # an explicit ``0`` (→ start input logprobs at position 0).
            logprob_start_len=(
                request.logprob_start_len if request.HasField("logprob_start_len") else -1
            ),
            top_logprobs_num=int(request.top_logprobs_num or 0),
            token_ids_logprob=(
                list(request.token_ids_logprob) if request.token_ids_logprob else None
            ),
            # Hidden-state forwarding, multimodal inputs, PD-disaggregated
            # serving, LoRA hot-swap and ``log_metrics`` are intentionally
            # absent from TokenSpeed's wire — leaving the engine defaults in
            # place keeps the call shape simple.
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

        return obj

    @staticmethod
    def _sampling_params_from_proto(
        params: tokenspeed_scheduler_pb2.SamplingParams,
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

        # Bools (always forwarded)
        out["skip_special_tokens"] = bool(params.skip_special_tokens)
        out["spaces_between_special_tokens"] = bool(params.spaces_between_special_tokens)
        out["ignore_eos"] = bool(params.ignore_eos)

        # n (OpenAI-compat, passthrough)
        if params.n:
            out["n"] = params.n
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
    ) -> tokenspeed_scheduler_pb2.GenerateResponse:
        meta = output.get("meta_info", {})
        token_ids = self._generated_output_ids(output, reason_dict)
        return tokenspeed_scheduler_pb2.GenerateResponse(
            request_id=rid,
            chunk=tokenspeed_scheduler_pb2.GenerateStreamChunk(
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
    ) -> tokenspeed_scheduler_pb2.GenerateResponse:
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

        return tokenspeed_scheduler_pb2.GenerateResponse(
            request_id=rid,
            complete=tokenspeed_scheduler_pb2.GenerateComplete(
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
