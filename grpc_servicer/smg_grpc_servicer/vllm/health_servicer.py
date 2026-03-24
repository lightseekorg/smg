"""
Standard gRPC health check service for vLLM Kubernetes probes.

Implements grpc.health.v1.Health protocol, delegating health status
to AsyncLLM.check_health() from the vLLM EngineClient protocol.
"""

import logging
from collections.abc import AsyncIterator

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

logger = logging.getLogger(__name__)


class VllmHealthServicer(health_pb2_grpc.HealthServicer):
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

    def __init__(self, async_llm):
        """
        Initialize health servicer.

        Args:
            async_llm: AsyncLLM instance for checking engine health
        """
        self.async_llm = async_llm
        self._shutting_down = False
        logger.info("Standard gRPC health service initialized")

    def set_not_serving(self):
        """Mark all services as NOT_SERVING during graceful shutdown."""
        self._shutting_down = True
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

    async def Watch(
        self,
        request: health_pb2.HealthCheckRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[health_pb2.HealthCheckResponse]:
        """
        Streaming health check - sends current status once.

        For now, sends current status once (Kubernetes doesn't use Watch).
        A full implementation would monitor status changes and stream updates.

        Args:
            request: Contains service name
            context: gRPC context

        Yields:
            HealthCheckResponse messages
        """
        service_name = request.service
        logger.debug(f"Health watch request for service: '{service_name}'")

        response = await self.Check(request, context)
        yield response
