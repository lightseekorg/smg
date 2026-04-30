"""
Standard gRPC health check service for MLX.

Implements grpc.health.v1.Health protocol with simple liveness tracking.
"""

import logging
from collections.abc import AsyncIterator

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

logger = logging.getLogger(__name__)


class MlxHealthServicer(health_pb2_grpc.HealthServicer):
    """Standard gRPC health check for MLX inference server."""

    OVERALL_SERVER = ""
    MLX_SERVICE = "mlx.grpc.engine.MlxEngine"

    def __init__(self):
        self._serving = False
        logger.info("MlxHealthServicer initialized")

    def set_serving(self):
        self._serving = True
        logger.info("Health status set to SERVING")

    def set_not_serving(self):
        self._serving = False
        logger.info("Health status set to NOT_SERVING")

    async def Check(
        self,
        request: health_pb2.HealthCheckRequest,
        context: grpc.aio.ServicerContext,
    ) -> health_pb2.HealthCheckResponse:
        service_name = request.service

        if service_name in (self.OVERALL_SERVER, self.MLX_SERVICE):
            status = (
                health_pb2.HealthCheckResponse.SERVING
                if self._serving
                else health_pb2.HealthCheckResponse.NOT_SERVING
            )
            return health_pb2.HealthCheckResponse(status=status)

        context.set_code(grpc.StatusCode.NOT_FOUND)
        context.set_details(f"Unknown service: {service_name}")
        return health_pb2.HealthCheckResponse(status=health_pb2.HealthCheckResponse.SERVICE_UNKNOWN)

    async def Watch(
        self,
        request: health_pb2.HealthCheckRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[health_pb2.HealthCheckResponse]:
        service_name = request.service

        if service_name in (self.OVERALL_SERVER, self.MLX_SERVICE):
            status = (
                health_pb2.HealthCheckResponse.SERVING
                if self._serving
                else health_pb2.HealthCheckResponse.NOT_SERVING
            )
        else:
            status = health_pb2.HealthCheckResponse.SERVICE_UNKNOWN

        yield health_pb2.HealthCheckResponse(status=status)
