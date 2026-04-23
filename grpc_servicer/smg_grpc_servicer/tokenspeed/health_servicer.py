"""Standard ``grpc.health.v1.Health`` servicer for the TokenSpeed backend.

Mirrors ``smg_grpc_servicer.sglang.health_servicer.SGLangHealthServicer`` —
same service-name semantics, same lifecycle (NOT_SERVING → SERVING → NOT_SERVING),
same ``check/watch`` contract — but sources liveness signals from a TokenSpeed
:class:`AsyncLLM` instead of an SGLang ``GrpcRequestManager``.

The Rust router uses this health check to auto-detect the backend runtime.
We advertise the SGLang service name because TokenSpeed speaks the SGLang
gRPC proto on the wire (see ``servicer.py`` docstring for rationale).
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

if TYPE_CHECKING:
    from tokenspeed.runtime.engine.async_llm import AsyncLLM

logger = logging.getLogger(__name__)

# Seconds of scheduler silence — with pending requests — before we report
# NOT_SERVING. Matches the SGLang equivalent so oncall dashboards are aligned.
STUCK_SCHEDULER_THRESHOLD_SEC = 30.0


class TokenSpeedHealthServicer(health_pb2_grpc.HealthServicer):
    """Health servicer that tracks TokenSpeed's AsyncLLM liveness.

    Advertises two service levels:

    * ``""`` (empty) — overall server health, flipped to SERVING once the
      warmup request succeeds and back to NOT_SERVING on shutdown.
    * ``sglang.grpc.scheduler.SglangScheduler`` — readiness: the base status,
      plus a scheduler-responsiveness check (if there are pending requests
      but the scheduler hasn't pushed output for >30s, report NOT_SERVING).
    """

    OVERALL_SERVER = ""
    SGLANG_SERVICE = "sglang.grpc.scheduler.SglangScheduler"

    def __init__(self, async_llm: AsyncLLM, scheduler_info: dict):
        self.async_llm = async_llm
        self.scheduler_info = scheduler_info
        self._serving_status: dict[str, int] = {
            self.OVERALL_SERVER: health_pb2.HealthCheckResponse.NOT_SERVING,
            self.SGLANG_SERVICE: health_pb2.HealthCheckResponse.NOT_SERVING,
        }
        logger.info("TokenSpeed gRPC health service initialized")

    def set_serving(self) -> None:
        """Flip both services to SERVING (call after successful warmup)."""
        self._serving_status[self.OVERALL_SERVER] = health_pb2.HealthCheckResponse.SERVING
        self._serving_status[self.SGLANG_SERVICE] = health_pb2.HealthCheckResponse.SERVING
        logger.info("TokenSpeed gRPC health status -> SERVING")

    def set_not_serving(self) -> None:
        """Flip both services to NOT_SERVING (call on shutdown)."""
        self._serving_status[self.OVERALL_SERVER] = health_pb2.HealthCheckResponse.NOT_SERVING
        self._serving_status[self.SGLANG_SERVICE] = health_pb2.HealthCheckResponse.NOT_SERVING
        logger.info("TokenSpeed gRPC health status -> NOT_SERVING")

    async def Check(
        self,
        request: health_pb2.HealthCheckRequest,
        context: grpc.aio.ServicerContext,
    ) -> health_pb2.HealthCheckResponse:
        service_name = request.service
        logger.debug("Health check request for service=%r", service_name)

        if self.async_llm.gracefully_exit:
            return health_pb2.HealthCheckResponse(status=health_pb2.HealthCheckResponse.NOT_SERVING)

        if service_name == self.OVERALL_SERVER:
            return health_pb2.HealthCheckResponse(
                status=self._serving_status.get(
                    self.OVERALL_SERVER, health_pb2.HealthCheckResponse.NOT_SERVING
                )
            )

        if service_name == self.SGLANG_SERVICE:
            base = self._serving_status.get(
                self.SGLANG_SERVICE, health_pb2.HealthCheckResponse.NOT_SERVING
            )
            if base != health_pb2.HealthCheckResponse.SERVING:
                return health_pb2.HealthCheckResponse(status=base)

            # Scheduler-stuck check: pending work but no recent output.
            time_since_last_receive = time.time() - self.async_llm.last_receive_tstamp
            pending = len(self.async_llm.rid_to_state)
            if time_since_last_receive > STUCK_SCHEDULER_THRESHOLD_SEC and pending > 0:
                logger.warning(
                    "Scheduler appears stuck: %.1fs since last receive, %d pending requests",
                    time_since_last_receive,
                    pending,
                )
                return health_pb2.HealthCheckResponse(
                    status=health_pb2.HealthCheckResponse.NOT_SERVING
                )

            return health_pb2.HealthCheckResponse(status=health_pb2.HealthCheckResponse.SERVING)

        context.set_code(grpc.StatusCode.NOT_FOUND)
        context.set_details(f"Unknown service: {service_name}")
        return health_pb2.HealthCheckResponse(status=health_pb2.HealthCheckResponse.SERVICE_UNKNOWN)

    async def Watch(
        self,
        request: health_pb2.HealthCheckRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[health_pb2.HealthCheckResponse]:
        # K8s probes use Check, not Watch — we emit the current status once.
        yield await self.Check(request, context)
