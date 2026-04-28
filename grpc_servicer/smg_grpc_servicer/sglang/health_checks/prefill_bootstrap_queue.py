"""Prefill bootstrap-queue stall detector.

Detects when a disaggregation-prefill engine's bootstrap queue is
saturated with no forward progress: every queued request is stuck in
bootstrap and the inflight/waiting queues are both empty. A healthy
busy server at its queue cap always has inflight > 0 because real
transfers are completing, so the all-zero-except-bootstrap tuple is an
unambiguous "pipeline stalled" signature under the current sglang
transports.

The check is cause-agnostic — it observes that forward progress has
stopped, not why. Common causes include transport-level failures (any
KV-transfer engine), decoder-side slot-allocation backpressure, ZMQ
control-channel breakage, or a decoder OOM. Operators inspecting the
diagnostics can correlate with transport / scheduler / network
telemetry to identify the specific root cause for a given incident.
"""

import asyncio
import logging
import time

logger = logging.getLogger(__name__)

# Cache TTL for the scheduler get_loads IPC call. Kubernetes liveness
# probes and the SMG router both probe the health endpoint at high
# frequency; caching the result within this window avoids stampeding
# the scheduler with redundant IPCs.
_GET_LOADS_CACHE_TTL_SEC = 1.0

# Hard timeout on the scheduler IPC. A wedged scheduler is itself a
# failure mode the checker should surface (as a raised exception, which
# the servicer's central fail-safe handles). Deliberately larger than
# the cache TTL so cache hits are cheap.
_GET_LOADS_IPC_TIMEOUT_SEC = 5.0


class PrefillBootstrapQueueStalledChecker:
    """HealthChecker detecting a stalled prefill bootstrap pipeline.

    Active only when ``server_args.disaggregation_mode == "prefill"`` and
    ``server_args.max_queued_requests`` is a positive int; otherwise
    always returns None (healthy). Cause-agnostic: fires on the stall
    signature regardless of which sub-system caused it.
    """

    name = "prefill_bootstrap_queue_stalled"

    def __init__(self, request_manager, server_args):
        """
        Args:
            request_manager: GrpcRequestManager instance used to poll
                scheduler/DP-rank load state via ``get_loads(["disagg"])``.
            server_args: sglang ServerArgs. Read for ``disaggregation_mode``
                (must be "prefill" for the checker to activate) and
                ``max_queued_requests`` (the bootstrap queue cap against
                which the stall signature is compared).
        """
        self._request_manager = request_manager
        self._server_args = server_args
        self._cache: list | None = None
        self._cache_ts: float = 0.0
        self._warned_all_ranks_missing_disagg: bool = False

    async def is_unhealthy(self) -> dict | None:
        sa = self._server_args
        if getattr(sa, "disaggregation_mode", "null") != "prefill":
            return None
        limit = getattr(sa, "max_queued_requests", None)
        if not isinstance(limit, int) or limit <= 0:
            return None

        loads = await self._get_loads_cached()
        if not loads:
            return None

        # A stall on ANY scheduler/DP rank is enough to declare the pod
        # unhealthy — once one rank's bootstrap queue is wedged, new
        # requests scheduled to it will stall indefinitely.
        ranks_with_disagg = 0
        for load in loads:
            disagg = getattr(load, "disaggregation", None)
            if disagg is None:
                continue
            ranks_with_disagg += 1
            bootstrap = getattr(disagg, "prefill_prealloc_queue_reqs", 0)
            inflight = getattr(disagg, "prefill_inflight_queue_reqs", 0)
            # num_waiting_reqs sums waiting_queue + disagg_prefill_bootstrap_queue
            # on the prefill side (see sglang scheduler_metrics_mixin.py),
            # so subtract to recover the real main-queue wait count. Clamp
            # at zero to guard against skew from independent counter reads.
            num_waiting_reqs = getattr(load, "num_waiting_reqs", 0)
            waiting = max(0, num_waiting_reqs - bootstrap)

            if bootstrap >= limit and inflight == 0 and waiting == 0:
                return {
                    "dp_rank": getattr(load, "dp_rank", None),
                    "bootstrap": bootstrap,
                    "inflight": inflight,
                    "waiting": waiting,
                    "limit": limit,
                }

        # Schema-drift canary: if we got load replies but no rank exposed
        # the disaggregation struct, the checker is effectively disabled.
        # Warn once so the silent-failure mode surfaces.
        if ranks_with_disagg == 0 and not self._warned_all_ranks_missing_disagg:
            logger.warning(
                "%s: received %d load replies but NONE contained a "
                "disaggregation struct. Either ['disagg'] stats were not "
                "requested, or the upstream GetLoadsReqOutput schema has "
                "changed. Checker disabled until fixed.",
                self.name,
                len(loads),
            )
            self._warned_all_ranks_missing_disagg = True

        return None

    async def _get_loads_cached(self) -> list:
        """TTL-cached, timeout-bounded wrapper around get_loads(["disagg"]).

        Under asyncio the ``await`` inside the refresh is the only
        suspension point, so two concurrent probes can both observe a
        stale cache and both issue IPCs — still a massive reduction vs.
        every probe. Correctness of the stall signature does not depend
        on strong cache consistency.
        """
        now = time.monotonic()
        if self._cache is not None and (now - self._cache_ts) < _GET_LOADS_CACHE_TTL_SEC:
            return self._cache
        loads = await asyncio.wait_for(
            self._request_manager.get_loads(include=["disagg"]),
            timeout=_GET_LOADS_IPC_TIMEOUT_SEC,
        )
        self._cache = loads
        self._cache_ts = now
        return loads
