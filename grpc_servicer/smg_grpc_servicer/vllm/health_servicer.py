"""
Standard gRPC health check service for vLLM Kubernetes probes.

Implements grpc.health.v1.Health protocol, delegating health status
to AsyncLLM.check_health() from the vLLM EngineClient protocol.
"""

from typing import TYPE_CHECKING

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc
from vllm.logger import init_logger

from smg_grpc_servicer.health_watch import HealthWatchMixin

if TYPE_CHECKING:
    from vllm.v1.engine.async_llm import AsyncLLM

logger = init_logger(__name__)

SERVING = health_pb2.HealthCheckResponse.SERVING
NOT_SERVING = health_pb2.HealthCheckResponse.NOT_SERVING
SERVICE_UNKNOWN = health_pb2.HealthCheckResponse.SERVICE_UNKNOWN


class VllmHealthServicer(HealthWatchMixin, health_pb2_grpc.HealthServicer):
    """
    Standard gRPC health check service for Kubernetes probes.
    Implements grpc.health.v1.Health protocol.

    Supports two service levels:
    1. Overall server health (service="") - for liveness probes
    2. VllmEngine service health (service="vllm.grpc.engine.VllmEngine")
       - for readiness probes

    Health is determined by calling async_llm.check_health(), the same
    EngineClient protocol method used by vLLM's HTTP /health endpoint.
    """

    OVERALL_SERVER = ""
    VLLM_SERVICE = "vllm.grpc.engine.VllmEngine"

    def __init__(self, async_llm: "AsyncLLM"):
        """
        Initialize health servicer.

        Args:
            async_llm: AsyncLLM instance for checking engine health
        """
        self.async_llm = async_llm
        self._shutting_down = False
        self._init_watch()
        logger.info("Standard gRPC health service initialized")

    def set_not_serving(self):
        """Mark all services as NOT_SERVING during graceful shutdown."""
        self._shutting_down = True
        self._notify_shutdown()
        logger.info("Health service status set to NOT_SERVING")

    async def Check(
        self,
        request: health_pb2.HealthCheckRequest,
        context: grpc.aio.ServicerContext,
    ) -> health_pb2.HealthCheckResponse:
        """
        Standard health check for Kubernetes probes.

        Args:
            request: Contains service name ("" for overall, or specific service)
            context: gRPC context

        Returns:
            HealthCheckResponse with SERVING/NOT_SERVING/SERVICE_UNKNOWN status
        """
        service_name = request.service
        logger.debug(f"Health check request for service: '{service_name}'")

        if self._shutting_down:
            logger.debug("Health check: Server is shutting down")
            return health_pb2.HealthCheckResponse(status=health_pb2.HealthCheckResponse.NOT_SERVING)

        if service_name in (self.OVERALL_SERVER, self.VLLM_SERVICE):
            try:
                await self.async_llm.check_health()
                logger.debug(f"Health check for '{service_name}': SERVING")
                return health_pb2.HealthCheckResponse(status=health_pb2.HealthCheckResponse.SERVING)
            except Exception:
                logger.exception("Health check failed for service '%s'", service_name)
                return health_pb2.HealthCheckResponse(
                    status=health_pb2.HealthCheckResponse.NOT_SERVING
                )

        logger.debug(f"Health check for unknown service: '{service_name}'")
        context.set_code(grpc.StatusCode.NOT_FOUND)
        context.set_details(f"Unknown service: {service_name}")
        return health_pb2.HealthCheckResponse(status=health_pb2.HealthCheckResponse.SERVICE_UNKNOWN)

    def _is_shutting_down(self) -> bool:
        return self._shutting_down

    async def _compute_watch_status(self, service_name: str) -> int:
        """Async status computation -- delegates to check_health()."""
        if self._shutting_down:
            return NOT_SERVING

        if service_name in (self.OVERALL_SERVER, self.VLLM_SERVICE):
            try:
                await self.async_llm.check_health()
                return SERVING
            except Exception:
                logger.debug(
                    "Health watch check failed for '%s'",
                    service_name,
                    exc_info=True,
                )
                return NOT_SERVING

        return SERVICE_UNKNOWN
