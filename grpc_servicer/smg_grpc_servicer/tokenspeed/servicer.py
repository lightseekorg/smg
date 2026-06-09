"""TokenSpeed gRPC servicer.

Implements ``tokenspeed.grpc.scheduler.TokenSpeedScheduler`` on top of
:class:`tokenspeed.runtime.engine.async_llm.AsyncLLM`. The proto field set
is intentionally minimal — generative LLM serving plus precomputed multimodal;
no Embed / GetTokenizer / SubscribeKvEvents / PD-disaggregated / LoRA /
hidden states / classifier outputs.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import re
import time
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import grpc
import numpy as np
import torch
from google.protobuf.struct_pb2 import Struct
from google.protobuf.timestamp_pb2 import Timestamp
from smg_grpc_proto import tokenspeed_scheduler_pb2_grpc
from smg_grpc_proto.generated import common_pb2, tokenspeed_scheduler_pb2
from tokenspeed.runtime.multimodal.inputs import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)

from smg_grpc_servicer.tokenspeed.health_servicer import TokenSpeedHealthServicer

if TYPE_CHECKING:
    # Type-only — keeps these out of the cold-path graph when the servicer is
    # imported by tooling that stubs the engine surface.
    from tokenspeed.runtime.engine.async_llm import AsyncLLM
    from tokenspeed.runtime.utils.server_args import ServerArgs

logger = logging.getLogger(__name__)

HEALTH_CHECK_TIMEOUT = int(os.getenv("TOKENSPEED_HEALTH_CHECK_TIMEOUT", "20"))
# Profile round-trips include trace serialization, which can take minutes.
PROFILE_TIMEOUT = 600.0
LOG_MM_TENSOR_DATA = os.getenv("TOKENSPEED_LOG_MM_TENSOR_DATA", "").lower() in (
    "1",
    "true",
    "yes",
)


def _lazy_generate_req_input():
    """Late import for ``tokenspeed.runtime.engine.io_struct.GenerateReqInput``.

    Kept lazy so the top of this module loads in test environments that stub
    the TokenSpeed engine surface (unit tests don't need a fully-working
    TokenSpeed install to exercise proto ↔ request-input conversion).
    """
    from tokenspeed.runtime.engine.io_struct import GenerateReqInput

    return GenerateReqInput


def _finish_reason_to_dict(reason: Any) -> dict | None:
    """Normalise a TokenSpeed finish reason into a dict.

    TokenSpeed emits ``BaseFinishReason``-style objects (or an already-
    normalised dict) in ``meta_info["finish_reason"]``; downstream code
    expects a dict with at minimum ``{"type": ...}`` and optionally
    ``{"matched": int|str}``. ``None`` means "still running".

    We duck-type on ``to_json()`` so the servicer module loads without
    pulling in TokenSpeed's full request-processing graph. Unknown shapes
    raise ``TypeError`` rather than silently flipping ``length`` / ``abort``
    to ``stop`` — the caller maps that to ``StatusCode.INTERNAL``.
    """
    if reason is None or isinstance(reason, dict):
        return reason
    to_json = getattr(reason, "to_json", None)
    if callable(to_json):
        result = to_json()
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
        except Exception as e:  # noqa: BLE001
            logger.exception("Failed to build generate request for %s", rid)
            await context.abort(grpc.StatusCode.INTERNAL, str(e))
            return

        # n>1 emits a *list* of final dicts (non-streaming) or per-choice
        # streamed dicts tagged with ``index`` — both handled below.
        expanded_rid = getattr(req_obj, "rid", None)

        # Threaded through the response builders so the matched stop token
        # stays in ``output_ids`` when the client asked to keep it.
        no_stop_trim = bool(request.sampling_params.no_stop_trim)

        aborted = False
        try:
            async for output in self.async_llm.generate_request(req_obj):
                # Non-streaming n>1 emits a list of final dicts in one yield.
                # Pre-scan for aborts so we don't yield partial successes
                # before raising on a later aborted choice.
                if isinstance(output, list):
                    item_reasons = [
                        _finish_reason_to_dict(item.get("meta_info", {}).get("finish_reason"))
                        for item in output
                    ]
                    for r in item_reasons:
                        if r and r.get("type") == "abort":
                            code = _abort_status_code(r)
                            await context.abort(code, r.get("message") or "aborted")
                            return
                    for idx, (item, item_reason) in enumerate(zip(output, item_reasons)):
                        ci = int(item.get("index", idx))
                        yield self._complete_response(
                            rid, item, item_reason, ci, no_stop_trim=no_stop_trim
                        )
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
                    yield self._chunk_response(
                        rid, output, reason_dict, choice_index, no_stop_trim=no_stop_trim
                    )
                    if is_finished:
                        yield self._complete_response(
                            rid, output, reason_dict, choice_index, no_stop_trim=no_stop_trim
                        )
                elif is_finished:
                    yield self._complete_response(
                        rid, output, reason_dict, choice_index, no_stop_trim=no_stop_trim
                    )

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

        Any scheduler push within ``HEALTH_CHECK_TIMEOUT`` counts as alive.
        ``log_metrics=False`` so health checks don't skew Prometheus counters.
        """
        rid = f"HEALTH_CHECK_{time.time()}"

        if self.async_llm.gracefully_exit:
            return tokenspeed_scheduler_pb2.HealthCheckResponse(
                healthy=False, message="Server is shutting down"
            )

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

        # Anchored regex avoids matching unrelated rids like "{rid}-name".
        child_pattern = re.compile(rf"^{re.escape(rid)}-n\d+$")
        targets = [r for r in state_map if r == rid or child_pattern.match(r)]

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

        # Upstream renamed ``model_path`` → ``model`` and
        # ``tokenizer_path`` → ``tokenizer``; accept either so the servicer
        # works against both old and new builds.
        model_path = getattr(self.server_args, "model", None) or getattr(
            self.server_args, "model_path", ""
        )
        tokenizer_path = getattr(self.server_args, "tokenizer", None) or getattr(
            self.server_args, "tokenizer_path", ""
        )
        supports_vision = bool(getattr(model_config, "is_multimodal", False))
        image_modality = getattr(tokenspeed_scheduler_pb2, "IMAGE", 1)
        video_modality = getattr(tokenspeed_scheduler_pb2, "VIDEO", 3)
        supported_modalities = []
        if supports_vision:
            supported_modalities.append(image_modality)
            if hf_config is not None and getattr(hf_config, "video_token_id", None) is not None:
                supported_modalities.append(video_modality)

        response_kwargs = dict(
            model_path=model_path,
            tokenizer_path=tokenizer_path or "",
            default_sampling_params_json=self.server_args.preferred_sampling_params or "",
            weight_version="",
            served_model_name=(self.server_args.served_model_name or model_path),
            max_context_length=int(self.async_llm.context_len),
            vocab_size=int(model_config.vocab_size),
            model_type=(getattr(hf_config, "model_type", "") or "") if hf_config else "",
            architectures=(getattr(hf_config, "architectures", []) or []) if hf_config else [],
            eos_token_ids=eos_token_ids,
            pad_token_id=(getattr(hf_config, "pad_token_id", 0) or 0) if hf_config else 0,
            bos_token_id=(getattr(hf_config, "bos_token_id", 0) or 0) if hf_config else 0,
            max_req_input_len=int(max_req_input_len),
            supports_vision=supports_vision,
        )
        fields = tokenspeed_scheduler_pb2.GetModelInfoResponse.DESCRIPTOR.fields_by_name
        if "supports_multimodal" in fields:
            response_kwargs["supports_multimodal"] = supports_vision
        if "supported_modalities" in fields:
            response_kwargs["supported_modalities"] = supported_modalities
        dtype = self._torch_dtype_to_proto(getattr(model_config, "dtype", None))
        if "model_dtype" in fields:
            response_kwargs["model_dtype"] = dtype
        if "multimodal_encoder_dtype" in fields:
            response_kwargs["multimodal_encoder_dtype"] = dtype
        return tokenspeed_scheduler_pb2.GetModelInfoResponse(**response_kwargs)

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
    # GetLoads (unary) — bridges to TokenSpeed's scheduler-side load metrics
    # ------------------------------------------------------------------

    async def GetLoads(
        self,
        request: tokenspeed_scheduler_pb2.GetLoadsRequest,
        context: grpc.aio.ServicerContext,
    ) -> tokenspeed_scheduler_pb2.GetLoadsResponse:
        """Return per-DP-rank scheduler load (optionally filtered to one rank).

        ``AsyncLLM.get_load()`` round-trips a ``GetLoadReqInput`` over the
        scheduler zmq channel; each reply carries ``num_reqs`` (running +
        waiting), ``num_waiting_reqs``, and ``num_pages`` (KV pages in use).
        """
        try:
            load_outputs = await asyncio.wait_for(
                self.async_llm.get_load(), timeout=HEALTH_CHECK_TIMEOUT
            )
        except TimeoutError:
            await context.abort(
                grpc.StatusCode.DEADLINE_EXCEEDED,
                f"tokenspeed scheduler did not respond to GetLoad within {HEALTH_CHECK_TIMEOUT}s",
            )
            return
        except Exception as e:  # noqa: BLE001
            logger.exception("GetLoads failed")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))
            return

        # Honor the optional ``dp_rank`` filter before any aggregation.
        if request.HasField("dp_rank"):
            wanted = int(request.dp_rank)
            load_outputs = [lo for lo in load_outputs if int(lo.dp_rank) == wanted]

        page_size = int(getattr(self.async_llm.server_args, "page_size", 1) or 1)
        # Fall back to ``server_args.max_total_num_tokens`` for SimpleNamespace test stubs.
        max_total_num_tokens = int(
            (self.scheduler_info.get("max_total_num_tokens") if self.scheduler_info else None)
            or getattr(self.async_llm.server_args, "max_total_num_tokens", 0)
            or 0
        )

        scheduler_loads: list[tokenspeed_scheduler_pb2.SchedulerLoad] = []
        total_running = 0
        total_waiting = 0
        token_usages: list[float] = []
        for lo in load_outputs:
            num_running = max(0, int(lo.num_reqs) - int(lo.num_waiting_reqs))
            num_used_tokens = int(lo.num_pages) * page_size
            token_usage = (
                num_used_tokens / max_total_num_tokens if max_total_num_tokens > 0 else 0.0
            )
            scheduler_loads.append(
                tokenspeed_scheduler_pb2.SchedulerLoad(
                    dp_rank=int(lo.dp_rank),
                    num_running_reqs=num_running,
                    num_waiting_reqs=int(lo.num_waiting_reqs),
                    num_total_reqs=int(lo.num_reqs),
                    num_used_tokens=num_used_tokens,
                    max_total_num_tokens=max_total_num_tokens,
                    token_usage=token_usage,
                )
            )
            total_running += num_running
            total_waiting += int(lo.num_waiting_reqs)
            token_usages.append(token_usage)

        aggregate = tokenspeed_scheduler_pb2.AggregateMetrics(
            total_running_reqs=total_running,
            total_waiting_reqs=total_waiting,
            total_reqs=total_running + total_waiting,
            avg_token_usage=(sum(token_usages) / len(token_usages)) if token_usages else 0.0,
        )

        return tokenspeed_scheduler_pb2.GetLoadsResponse(
            timestamp=datetime.now(timezone.utc).isoformat(),
            version="tokenspeed",
            dp_rank_count=len(scheduler_loads),
            loads=scheduler_loads,
            aggregate=aggregate,
        )

    async def FlushCache(
        self,
        request: common_pb2.FlushCacheRequest,
        context: grpc.aio.ServicerContext,
    ) -> common_pb2.FlushCacheResponse:
        """Flush the KV cache on the scheduler.

        TokenSpeed's ``FlushCacheReqInput`` carries no wait-for-idle knob, so
        ``timeout_s`` only widens the gRPC-side wait for the scheduler reply;
        the flush itself is immediate (fails if requests are in flight).
        """
        logger.debug("Receive flush cache request")
        comm_timeout = max(30.0, request.timeout_s + 10.0)
        try:
            result = await asyncio.wait_for(self.async_llm.flush_cache(), timeout=comm_timeout)
        except TimeoutError:
            await context.abort(
                grpc.StatusCode.DEADLINE_EXCEEDED,
                f"Flush cache timed out after {comm_timeout}s",
            )
            return
        except Exception as e:  # noqa: BLE001
            logger.exception("FlushCache failed")
            await context.abort(grpc.StatusCode.INTERNAL, f"Flush cache failed: {e}")
            return

        # TokenSpeed's FlushCacheReqOutput only carries `success` (no message
        # field); tolerate one appearing upstream later.
        message = getattr(result, "message", "") or (
            "Cache flushed successfully" if result.success else "Cache flush failed"
        )
        return common_pb2.FlushCacheResponse(success=bool(result.success), message=message)

    async def StartProfile(
        self,
        request: common_pb2.StartProfileRequest,
        context: grpc.aio.ServicerContext,
    ) -> common_pb2.ProfileResponse:
        """Start the profiler on the scheduler.

        ``AsyncLLM.start_profile`` owns the business logic (env-var defaults,
        ProfileReq construction) and raises on failure.
        """
        logger.debug("Receive start profile request")
        try:
            await asyncio.wait_for(
                self.async_llm.start_profile(
                    output_dir=request.output_dir if request.HasField("output_dir") else None,
                    start_step=request.start_step if request.HasField("start_step") else None,
                    num_steps=request.num_steps if request.HasField("num_steps") else None,
                    activities=list(request.activities) if request.activities else None,
                    with_stack=request.with_stack if request.HasField("with_stack") else None,
                    record_shapes=(
                        request.record_shapes if request.HasField("record_shapes") else None
                    ),
                    profile_by_stage=request.profile_by_stage,
                ),
                timeout=PROFILE_TIMEOUT,
            )
        except TimeoutError:
            await context.abort(
                grpc.StatusCode.DEADLINE_EXCEEDED,
                f"Start profiling timed out after {PROFILE_TIMEOUT}s",
            )
            return
        except Exception as e:  # noqa: BLE001
            logger.exception("StartProfile failed")
            await context.abort(grpc.StatusCode.INTERNAL, f"Start profiling failed: {e}")
            return

        return common_pb2.ProfileResponse(success=True, message="Start profiling succeeded")

    async def StopProfile(
        self,
        request: common_pb2.StopProfileRequest,
        context: grpc.aio.ServicerContext,
    ) -> common_pb2.ProfileResponse:
        """Stop the profiler on the scheduler and export traces."""
        logger.debug("Receive stop profile request")
        try:
            await asyncio.wait_for(self.async_llm.stop_profile(), timeout=PROFILE_TIMEOUT)
        except TimeoutError:
            await context.abort(
                grpc.StatusCode.DEADLINE_EXCEEDED,
                f"Stop profiling timed out after {PROFILE_TIMEOUT}s",
            )
            return
        except Exception as e:  # noqa: BLE001
            logger.exception("StopProfile failed")
            await context.abort(grpc.StatusCode.INTERNAL, f"Stop profiling failed: {e}")
            return

        return common_pb2.ProfileResponse(success=True, message="Stop profiling succeeded")

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

        sampling = self._sampling_params_from_proto(
            request.sampling_params,
            reasoning_parser=getattr(self.server_args, "reasoning_parser", None),
        )

        # Decode the precomputed multimodal payload, if the request carries one.
        precomputed_mm = None
        if request.HasField("mm_inputs"):
            precomputed_mm = self._mm_inputs_from_proto(
                request.mm_inputs,
                model_dtype=getattr(self.async_llm.model_config, "dtype", None),
            )

        GenerateReqInput = _lazy_generate_req_input()
        obj = GenerateReqInput(
            input_ids=input_ids,
            sampling_params=sampling,
            stream=bool(request.stream),
            return_logprob=bool(request.return_logprob),
            # presence-tracking distinguishes "client omitted" (→ ``-1`` =
            # no input logprobs) from explicit ``0`` (start at position 0).
            logprob_start_len=(
                request.logprob_start_len if request.HasField("logprob_start_len") else -1
            ),
            top_logprobs_num=int(request.top_logprobs_num or 0),
            token_ids_logprob=(
                list(request.token_ids_logprob) if request.token_ids_logprob else None
            ),
            precomputed_multimodal_inputs=precomputed_mm,
        )
        # ``normalize_batch_and_arguments`` asserts ``rid`` is a list when
        # n>1; expand to deterministic per-choice rids so the assert holds.
        n = sampling.get("n", 1) or 1
        if n > 1:
            obj.rid = [f"{request.request_id}-n{i}" for i in range(n)]
        else:
            obj.rid = request.request_id

        # Don't set ``obj.text`` even when the proto carries
        # ``original_text``: the HTTP path passes ``input_ids=[...], text=None``
        # and setting both perturbs the engine's input-processor.

        return obj

    @staticmethod
    def _sampling_params_from_proto(
        params: tokenspeed_scheduler_pb2.SamplingParams,
        *,
        reasoning_parser: str | None = None,
    ) -> dict[str, Any]:
        """Build the dict that ``GenerateReqInput.sampling_params`` expects.

        Field names must match :class:`SamplingParams.__init__`
        (``max_new_tokens``, ``stop``, ``stop_token_ids``, ...).
        """
        out: dict[str, Any] = {}

        # Sampling scalars are ``optional``; ``HasField()`` forwards only
        # what the client explicitly set so absent fields fall through to
        # engine defaults. Avoids the truthy-check pitfall that would drop
        # an explicit ``temperature=0`` (greedy decoding).
        for _field in (
            "max_new_tokens",
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "frequency_penalty",
            "presence_penalty",
            "repetition_penalty",
        ):
            if params.HasField(_field):
                out[_field] = getattr(params, _field)

        if params.min_new_tokens:
            # ``min_new_tokens`` is non-optional; 0 is the "no minimum" sentinel.
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
        # Keeps the matched stop token in ``output_ids`` so it reaches the
        # gateway's detokenizer when ``skip_special_tokens=False``.
        out["no_stop_trim"] = bool(params.no_stop_trim)

        # n (OpenAI-compat, passthrough)
        if params.n:
            out["n"] = params.n
        if params.logit_bias:
            out["logit_bias"] = dict(params.logit_bias)

        # Constraint types — exactly one may be set.
        if params.HasField("regex"):
            out["regex"] = params.regex
        elif params.HasField("json_schema"):
            # For reasoning parsers with an xgrammar template (e.g.
            # ``gpt-oss`` → ``harmony``), wrap the JSON schema as a
            # structural tag so the grammar only activates inside the
            # response channel — otherwise xgrammar fights the channel
            # preamble (``<|channel|>analysis<|message|>…``) and the model
            # stalls until ``max_tokens``.
            wrapped: str | None = None
            if reasoning_parser:
                try:
                    from tokenspeed.runtime.grammar.reasoning_structural_tag import (
                        structural_tag_for_reasoning_json_schema,
                    )

                    wrapped = structural_tag_for_reasoning_json_schema(
                        reasoning_parser, json.loads(params.json_schema)
                    )
                except (ImportError, json.JSONDecodeError):
                    wrapped = None
            if wrapped is not None:
                out["structural_tag"] = wrapped
            else:
                out["json_schema"] = params.json_schema
        elif params.HasField("ebnf_grammar"):
            out["ebnf"] = params.ebnf_grammar
        elif params.HasField("structural_tag"):
            out["structural_tag"] = params.structural_tag

        return out

    def _mm_inputs_from_proto(
        self,
        mm_inputs: tokenspeed_scheduler_pb2.MultimodalInputs,
        *,
        model_dtype: torch.dtype | None = None,
    ):
        """Reconstruct the engine's ``MultimodalInputs`` from the precomputed proto.

        The gateway already preprocessed, so the engine skips its own preprocessing
        (``precomputed_multimodal_inputs`` is set); this just boxes the tensors and
        placeholder offsets into the engine's data class.
        """
        return self._mm_inputs_from_itemized_proto(mm_inputs, model_dtype=model_dtype)

    def _mm_inputs_from_itemized_proto(
        self,
        mm_inputs: tokenspeed_scheduler_pb2.MultimodalInputs,
        *,
        model_dtype: torch.dtype | None = None,
    ):
        items = []
        im_token_id = None
        video_token_id = None

        for item_proto in mm_inputs.items:
            modality = self._modality_from_proto(item_proto.modality)
            if not item_proto.HasField("encoder_input"):
                raise ValueError("MultimodalItem must include encoder_input")

            feature = self._tensor_from_proto(item_proto.encoder_input, cast_to=model_dtype)
            if LOG_MM_TENSOR_DATA:
                encoder_input = item_proto.encoder_input
                payload = encoder_input.WhichOneof("payload")
                inline_nbytes = (
                    len(encoder_input.inline) if payload == "inline" else None
                )
                logger.info(
                    "Multimodal encoder_input received: modality=%s proto_dtype=%s "
                    "shape=%s payload=%s inline_nbytes=%s torch_dtype=%s cast_to=%s",
                    modality,
                    encoder_input.dtype,
                    list(encoder_input.shape),
                    payload,
                    inline_nbytes,
                    feature.dtype,
                    model_dtype,
                )
            model_specific_data = {
                name: self._tensor_from_proto(tensor_data, cast_to=model_dtype)
                for name, tensor_data in item_proto.model_specific_tensors.items()
            }
            self._validate_item_tensor_consistency(modality, model_specific_data)

            if not item_proto.placeholders:
                raise ValueError("MultimodalItem carried no placeholders")
            if any(p.length <= 0 for p in item_proto.placeholders):
                raise ValueError("MultimodalItem.placeholders.length must be > 0")
            offsets = [
                (p.offset, p.offset + p.length - 1) for p in item_proto.placeholders
            ]

            content_hash = bytes(item_proto.content_hash)
            mm_item = MultimodalDataItem(
                modality=modality,
                feature=feature,
                model_specific_data=model_specific_data,
                offsets=offsets,
                hash=int.from_bytes(content_hash[:8], "little") if content_hash else None,
            )
            mm_item.set_pad_value()
            items.append(mm_item)

            if item_proto.HasField("placeholder_token_id"):
                if modality == Modality.IMAGE:
                    im_token_id = int(item_proto.placeholder_token_id)
                elif modality == Modality.VIDEO:
                    video_token_id = int(item_proto.placeholder_token_id)

        if not items:
            raise ValueError("MultimodalInputs.items is empty")
        return MultimodalInputs(
            mm_items=items,
            im_token_id=im_token_id,
            video_token_id=video_token_id,
        )

    @staticmethod
    def _modality_from_proto(modality: int) -> Modality:
        if modality == getattr(tokenspeed_scheduler_pb2, "IMAGE", 1):
            return Modality.IMAGE
        if modality == getattr(tokenspeed_scheduler_pb2, "VIDEO", 3):
            return Modality.VIDEO
        if modality == getattr(tokenspeed_scheduler_pb2, "AUDIO", 2):
            raise ValueError("TokenSpeed audio multimodal inputs are not supported yet")
        raise ValueError(f"Unsupported multimodal item modality: {modality}")

    @staticmethod
    def _validate_item_tensor_consistency(
        modality: Modality, model_specific_data: dict[str, torch.Tensor]
    ) -> None:
        has_image_grid = "image_grid_thw" in model_specific_data
        has_video_grid = "video_grid_thw" in model_specific_data
        if modality == Modality.IMAGE and has_video_grid:
            raise ValueError("IMAGE MultimodalItem must not carry video_grid_thw")
        if modality == Modality.VIDEO and has_image_grid:
            raise ValueError("VIDEO MultimodalItem must not carry image_grid_thw")
        if modality == Modality.VIDEO and not has_video_grid:
            raise ValueError("VIDEO MultimodalItem must carry video_grid_thw")

    @staticmethod
    def _tensor_from_proto(
        tensor_data: tokenspeed_scheduler_pb2.TensorData,
        cast_to: torch.dtype | None = None,
    ):
        """Reconstruct a torch.Tensor from a proto TensorData.

        Floats are cast to ``cast_to``, fused into the decode; the buffer is
        copied so it never aliases the transient proto bytes.
        """
        shape = list(tensor_data.shape)
        raw = TokenSpeedSchedulerServicer._tensor_payload_bytes(tensor_data)

        if tensor_data.dtype == "bfloat16":
            # numpy has no bfloat16 — read the raw bits as uint16, reinterpret.
            expected = int(np.prod(shape, dtype=np.int64)) * np.dtype(np.uint16).itemsize
            if len(raw) != expected:
                raise ValueError(
                    f"TensorData byte length mismatch for bfloat16 shape={shape}: "
                    f"expected {expected}, got {len(raw)}"
                )
            t = torch.from_numpy(
                np.frombuffer(raw, dtype=np.uint16).reshape(shape)
            ).view(torch.bfloat16)
        else:
            dtype = np.dtype(tensor_data.dtype)
            expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
            if len(raw) != expected:
                raise ValueError(
                    f"TensorData byte length mismatch for dtype={tensor_data.dtype}, "
                    f"shape={shape}: expected {expected}, got {len(raw)}"
                )
            t = torch.from_numpy(
                np.frombuffer(raw, dtype=dtype).reshape(shape)
            )

        if cast_to is not None and t.dtype != cast_to and t.is_floating_point():
            return t.to(cast_to)
        return t.clone()

    @staticmethod
    def _tensor_payload_bytes(tensor_data: tokenspeed_scheduler_pb2.TensorData) -> bytes:
        payload = tensor_data.WhichOneof("payload")
        if payload == "inline":
            return bytes(tensor_data.inline)
        if payload == "shm":
            raise ValueError("TensorData.shm payload is not implemented yet")
        if payload == "remote":
            raise ValueError("TensorData.remote payload is not implemented yet")
        raise ValueError("TensorData payload is required")

    @staticmethod
    def _torch_dtype_to_proto(dtype: torch.dtype | None) -> str:
        if dtype is torch.bfloat16:
            return "bfloat16"
        if dtype is torch.float16:
            return "float16"
        if dtype is torch.float32:
            return "float32"
        return ""

    def _generated_output_ids(
        self,
        output: dict,
        reason_dict: dict | None,
        *,
        no_stop_trim: bool = False,
    ) -> list[int]:
        """Return just the newly-generated tokens from an AsyncLLM output dict.

        ``output_ids`` is prefixed with the Llama-3 assistant chat-template
        header (``<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n``)
        and suffixed with the trailing matched stop token. Slicing the last
        ``meta_info.completion_tokens`` strips the prefix; we then drop any
        trailing matched stop. The per-choice ``matched_stop`` rides in a
        dedicated proto field, so nothing is lost.
        """
        raw = list(output.get("output_ids") or [])
        if not raw:
            return raw
        completion = output.get("meta_info", {}).get("completion_tokens")
        if isinstance(completion, int) and 0 <= completion <= len(raw):
            # ``raw[-0:]`` is the whole list, not empty — guard explicitly.
            token_ids = raw[-completion:] if completion > 0 else []
        else:
            token_ids = raw
        if not no_stop_trim and reason_dict and reason_dict.get("type") == "stop":
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
        *,
        no_stop_trim: bool = False,
    ) -> tokenspeed_scheduler_pb2.GenerateResponse:
        meta = output.get("meta_info", {})
        token_ids = self._generated_output_ids(output, reason_dict, no_stop_trim=no_stop_trim)
        return tokenspeed_scheduler_pb2.GenerateResponse(
            request_id=rid,
            chunk=tokenspeed_scheduler_pb2.GenerateStreamChunk(
                token_ids=token_ids,
                prompt_tokens=int(meta.get("prompt_tokens", 0)),
                completion_tokens=int(meta.get("completion_tokens", len(token_ids))),
                cached_tokens=int(meta.get("cached_tokens", 0)),
                output_logprobs=self._convert_output_logprobs_to_proto(output, len(token_ids)),
                index=choice_index,
            ),
        )

    def _complete_response(
        self,
        rid: str,
        output: dict,
        reason_dict: dict | None,
        choice_index: int = 0,
        *,
        no_stop_trim: bool = False,
    ) -> tokenspeed_scheduler_pb2.GenerateResponse:
        meta = output.get("meta_info", {})
        token_ids = self._generated_output_ids(output, reason_dict, no_stop_trim=no_stop_trim)

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
                output_logprobs=self._convert_output_logprobs_to_proto(output, len(token_ids)),
                index=choice_index,
                **matched_kwargs,
            ),
        )

    @staticmethod
    def _convert_output_logprobs_to_proto(
        output: dict, n_keep: int
    ) -> tokenspeed_scheduler_pb2.OutputLogProbs | None:
        """Build an ``OutputLogProbs`` proto from a tokenspeed output dict.

        TokenSpeed accumulates the request's logprobs in per-request state
        across chunks; ``meta_info["output_token_logprobs"]`` is therefore the
        running cumulative list of detokenized
        ``(logprob: float, token_id: int, text: Optional[str])`` tuples, and
        ``meta_info["output_top_logprobs"]`` is the parallel list of top-K
        alternatives per position (each entry is ``None`` or a list of the
        same tuple shape).

        We slice the cumulative list down to just **this frame's tokens** by
        taking the last ``len(output["output_ids"])`` entries — that's how
        many new tokens this frame emitted — and then keep only the first
        ``n_keep`` of those, so the alignment matches whatever
        ``_generated_output_ids`` returned (it strips a trailing stop token
        when the finish reason is ``stop``, leaving the last logprob entry
        with no corresponding output id).

        Returns ``None`` when there are no logprobs to emit — either the
        client did not request them, or the server was started without
        ``--enable-output-logprobs`` (in which case TokenSpeed silently
        leaves these meta_info lists empty rather than raising).
        """
        if n_keep <= 0:
            return None
        meta = output.get("meta_info", {}) or {}
        raw_token = meta.get("output_token_logprobs") or []
        if not raw_token:
            return None
        n_chunk = len(output.get("output_ids", []) or [])
        if n_chunk <= 0:
            return None

        raw_top = meta.get("output_top_logprobs") or []
        chunk_token = raw_token[-n_chunk:] if len(raw_token) >= n_chunk else raw_token
        chunk_top = raw_top[-n_chunk:] if len(raw_top) >= n_chunk else raw_top
        delta_token = chunk_token[:n_keep]
        # Pad ``delta_top`` to align with ``delta_token`` — TokenSpeed leaves
        # ``output_top_logprobs`` empty when ``--enable-top-logprobs`` is off,
        # but the gateway expects one ``TopLogProbs`` per emitted token.
        delta_top = chunk_top[:n_keep] + [None] * max(0, len(delta_token) - len(chunk_top))

        top_proto = []
        for entry in delta_top:
            if entry:
                top_proto.append(
                    tokenspeed_scheduler_pb2.TopLogProbs(
                        values=[t[0] for t in entry],
                        token_ids=[t[1] for t in entry],
                    )
                )
            else:
                # Position with no top-K data (e.g. ``--enable-top-logprobs``
                # is not yet implemented in TokenSpeed; we still emit a
                # placeholder per position so the gateway can align indices).
                top_proto.append(tokenspeed_scheduler_pb2.TopLogProbs())

        return tokenspeed_scheduler_pb2.OutputLogProbs(
            token_logprobs=[t[0] for t in delta_token],
            token_ids=[t[1] for t in delta_token],
            top_logprobs=top_proto,
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
