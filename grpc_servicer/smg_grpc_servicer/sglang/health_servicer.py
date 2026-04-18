"""
Standard gRPC health check service implementation for Kubernetes probes.

This module implements the grpc.health.v1.Health service protocol, enabling
native Kubernetes gRPC health probes for liveness and readiness checks.
"""

import logging
import time

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

from smg_grpc_servicer.health_watch import HealthWatchMixin

logger = logging.getLogger(__name__)

SERVING = health_pb2.HealthCheckResponse.SERVING
NOT_SERVING = health_pb2.HealthCheckResponse.NOT_SERVING
SERVICE_UNKNOWN = health_pb2.HealthCheckResponse.SERVICE_UNKNOWN


class SGLangHealthServicer(HealthWatchMixin, health_pb2_grpc.HealthServicer):
    """
    Standard gRPC health check service implementation for Kubernetes probes.
    Implements grpc.health.v1.Health protocol.

    Supports two service levels:
    1. Overall server health (service="") - for liveness probes
    2. SGLang service health (service="sglang.grpc.scheduler.SglangScheduler") - for readiness probes

    Health status lifecycle:
    - NOT_SERVING: Initial state, model loading, or shutting down
    - SERVING: Model loaded and ready to serve requests
    """

    SCHEDULER_RESPONSIVENESS_TIMEOUT_S = 30

    # Service names we support
    OVERALL_SERVER = ""  # Empty string for overall server health
    SGLANG_SERVICE = "sglang.grpc.scheduler.SglangScheduler"

    def __init__(self, request_manager, scheduler_info: dict):
        """
        Initialize health servicer.

        Args:
            request_manager: GrpcRequestManager instance for checking server state
            scheduler_info: Dict containing scheduler metadata
        """
        self.request_manager = request_manager
        self.scheduler_info = scheduler_info
        self._serving_status = {}

        # Initially set to NOT_SERVING until model is loaded
        self._serving_status[self.OVERALL_SERVER] = health_pb2.HealthCheckResponse.NOT_SERVING
        self._serving_status[self.SGLANG_SERVICE] = health_pb2.HealthCheckResponse.NOT_SERVING

        self._init_watch()
        logger.info("Standard gRPC health service initialized")

    def set_serving(self):
        """Mark services as SERVING - call this after model is loaded."""
        self._serving_status[self.OVERALL_SERVER] = health_pb2.HealthCheckResponse.SERVING
        self._serving_status[self.SGLANG_SERVICE] = health_pb2.HealthCheckResponse.SERVING
        logger.info("Health service status set to SERVING")

    def set_not_serving(self):
        """Mark services as NOT_SERVING - call this during shutdown."""
        self._serving_status[self.OVERALL_SERVER] = health_pb2.HealthCheckResponse.NOT_SERVING
        self._serving_status[self.SGLANG_SERVICE] = health_pb2.HealthCheckResponse.NOT_SERVING
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

        # Check if shutting down
        if self.request_manager.gracefully_exit:
            logger.debug("Health check: Server is shutting down")
            return health_pb2.HealthCheckResponse(status=health_pb2.HealthCheckResponse.NOT_SERVING)

        # Overall server health - just check if process is alive
        if service_name == self.OVERALL_SERVER:
            status = self._serving_status.get(
                self.OVERALL_SERVER, health_pb2.HealthCheckResponse.NOT_SERVING
            )
            logger.debug(
                f"Overall health check: {health_pb2.HealthCheckResponse.ServingStatus.Name(status)}"
            )
            return health_pb2.HealthCheckResponse(status=status)

        # Specific service health - check if ready to serve
        elif service_name == self.SGLANG_SERVICE:
            # Additional checks for service readiness

            # Check base status first
            base_status = self._serving_status.get(
                self.SGLANG_SERVICE, health_pb2.HealthCheckResponse.NOT_SERVING
            )

            if base_status != health_pb2.HealthCheckResponse.SERVING:
                logger.debug("Service health check: NOT_SERVING (base status)")
                return health_pb2.HealthCheckResponse(status=base_status)

            # Check if scheduler is responsive (received data recently)
            time_since_last_receive = time.time() - self.request_manager.last_receive_tstamp

            # If no recent activity and we have active requests, might be stuck
            if time_since_last_receive > self.SCHEDULER_RESPONSIVENESS_TIMEOUT_S and len(self.request_manager.rid_to_state) > 0:
                logger.warning(
                    f"Service health check: Scheduler not responsive "
                    f"({time_since_last_receive:.1f}s since last receive, "
                    f"{len(self.request_manager.rid_to_state)} pending requests)"
                )
                return health_pb2.HealthCheckResponse(
                    status=health_pb2.HealthCheckResponse.NOT_SERVING
                )

            logger.debug("Service health check: SERVING")
            return health_pb2.HealthCheckResponse(status=health_pb2.HealthCheckResponse.SERVING)

        # Unknown service
        else:
            logger.debug(f"Health check for unknown service: '{service_name}'")
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Unknown service: {service_name}")
            return health_pb2.HealthCheckResponse(
                status=health_pb2.HealthCheckResponse.SERVICE_UNKNOWN
            )

    def _is_shutting_down(self) -> bool:
        # _watch_notified_shutdown is set by _notify_shutdown() in set_not_serving();
        # gracefully_exit covers external shutdown from the request manager.
        return self.request_manager.gracefully_exit or self._watch_notified_shutdown

    def _compute_watch_status(self, service_name: str) -> int:
        """Sync status computation -- no I/O needed."""
        if self.request_manager.gracefully_exit:
            return NOT_SERVING

        if service_name == self.OVERALL_SERVER:
            return self._serving_status.get(self.OVERALL_SERVER, NOT_SERVING)

        if service_name == self.SGLANG_SERVICE:
            base_status = self._serving_status.get(self.SGLANG_SERVICE, NOT_SERVING)
            if base_status != SERVING:
                return base_status
            time_since = time.time() - self.request_manager.last_receive_tstamp
            if time_since > self.SCHEDULER_RESPONSIVENESS_TIMEOUT_S and len(self.request_manager.rid_to_state) > 0:
                logger.warning(
                    "Scheduler not responsive (%.1fs, %d pending)",
                    time_since,
                    len(self.request_manager.rid_to_state),
                )
                return NOT_SERVING
            return SERVING

        return SERVICE_UNKNOWN
