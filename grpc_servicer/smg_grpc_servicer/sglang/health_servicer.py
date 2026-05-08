"""Standard gRPC health check service implementation for Kubernetes probes.

Implements grpc.health.v1.Health and layers an extensible checker
registry on top. Registered HealthChecker predicates can flip Check() to
NOT_SERVING when they observe engine-level failures that the base
serving-status flags (startup / shutdown) don't catch — dead scheduler
threads, pipeline deadlocks, backpressure starvation, etc.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Sequence

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

from smg_grpc_servicer.sglang.health_checks import HealthChecker

logger = logging.getLogger(__name__)

# Escalate per-checker exception logs at these consecutive-failure counts
# so a silent dependency outage surfaces at WARNING instead of hiding at
# DEBUG, without flooding the log at probe rate.
_CHECKER_FAILURE_LOG_STEPS = (1, 10, 100)


class SGLangHealthServicer(health_pb2_grpc.HealthServicer):
    """
    Standard gRPC health check service implementation for Kubernetes probes.
    Implements grpc.health.v1.Health protocol.

    Supports two service levels:
    1. Overall server health (service="") - for liveness probes
    2. SGLang service health (service="sglang.grpc.scheduler.SglangScheduler") - for readiness probes

    Base health status lifecycle:
    - NOT_SERVING: Initial state, model loading, or shutting down
    - SERVING: Model loaded and ready to serve requests

    Beyond the base status, registered HealthChecker instances can
    override a SERVING response with NOT_SERVING when they observe an
    engine-level failure (e.g. a stalled pipeline). The first checker
    returning an unhealthy result wins.
    """

    # Service names we support
    OVERALL_SERVER = ""  # Empty string for overall server health
    SGLANG_SERVICE = "sglang.grpc.scheduler.SglangScheduler"

    def __init__(
        self,
        request_manager,
        scheduler_info: dict,
        checkers: Sequence[HealthChecker] | None = None,
    ):
        """
        Initialize health servicer.

        Args:
            request_manager: GrpcRequestManager instance for checking server state
            scheduler_info: Dict containing scheduler metadata
            checkers: Optional sequence of HealthChecker predicates. Each
                is polled on every Check() that would otherwise return
                SERVING; the first one returning an unhealthy diagnostic
                flips the response to NOT_SERVING. An empty sequence
                preserves legacy behavior verbatim.
        """
        self.request_manager = request_manager
        self.scheduler_info = scheduler_info
        self._checkers: list[HealthChecker] = list(checkers) if checkers else []
        self._serving_status = {}

        # Initially set to NOT_SERVING until model is loaded
        self._serving_status[self.OVERALL_SERVER] = health_pb2.HealthCheckResponse.NOT_SERVING
        self._serving_status[self.SGLANG_SERVICE] = health_pb2.HealthCheckResponse.NOT_SERVING

        # Aggregate transition state: True iff the most recent determined
        # checker run saw any checker unhealthy. Starts False so the first
        # real detection emits a transition warning (covers the "pod came
        # up already unhealthy" case).
        self._any_checker_unhealthy: bool = False

        # Per-checker consecutive-failure counts to escalate exception
        # log level without flooding at probe rate.
        self._checker_failure_counts: dict[str, int] = {c.name: 0 for c in self._checkers}

        self._log_checker_registration()

        logger.info("Standard gRPC health service initialized")

    def _log_checker_registration(self) -> None:
        """Describe the registered checker set at startup so a missing
        checker (e.g. caller forgot to wire one up) is visible in logs
        instead of silently leaving the feature disabled for the life of
        the pod.
        """
        if not self._checkers:
            logger.info("Health-check registry: 0 checkers registered")
            return
        names = ", ".join(c.name for c in self._checkers)
        logger.info(
            "Health-check registry: %d checker(s) registered: [%s]",
            len(self._checkers),
            names,
        )

    def set_serving(self):
        """Mark services as SERVING - call this after model is loaded."""
        self._serving_status[self.OVERALL_SERVER] = health_pb2.HealthCheckResponse.SERVING
        self._serving_status[self.SGLANG_SERVICE] = health_pb2.HealthCheckResponse.SERVING
        logger.info("Health service status set to SERVING")

    def set_not_serving(self):
        """Mark services as NOT_SERVING - call this during shutdown."""
        self._serving_status[self.OVERALL_SERVER] = health_pb2.HealthCheckResponse.NOT_SERVING
        self._serving_status[self.SGLANG_SERVICE] = health_pb2.HealthCheckResponse.NOT_SERVING
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
            if (
                status == health_pb2.HealthCheckResponse.SERVING
                and await self._any_unhealthy_checker()
            ):
                status = health_pb2.HealthCheckResponse.NOT_SERVING
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
            # NOTE: 30s timeout is hardcoded. This is more conservative than
            # HEALTH_CHECK_TIMEOUT (20s) used for custom HealthCheck RPC.
            # Consider making this configurable via environment variable in the future
            # if different workloads need different responsiveness thresholds.
            if time_since_last_receive > 30 and len(self.request_manager.rid_to_state) > 0:
                logger.warning(
                    f"Service health check: Scheduler not responsive "
                    f"({time_since_last_receive:.1f}s since last receive, "
                    f"{len(self.request_manager.rid_to_state)} pending requests)"
                )
                return health_pb2.HealthCheckResponse(
                    status=health_pb2.HealthCheckResponse.NOT_SERVING
                )

            # Poll registered HealthChecker predicates. Any one reporting
            # unhealthy flips the response to NOT_SERVING.
            if await self._any_unhealthy_checker():
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

    async def Watch(
        self,
        request: health_pb2.HealthCheckRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[health_pb2.HealthCheckResponse]:
        """
        Streaming health check - sends updates when status changes.

        For now, just send current status once (Kubernetes doesn't use Watch).
        A full implementation would monitor status changes and stream updates.

        Args:
            request: Contains service name
            context: gRPC context

        Yields:
            HealthCheckResponse messages when status changes
        """
        service_name = request.service
        logger.debug(f"Health watch request for service: '{service_name}'")

        # Send current status. Because this delegates to Check(), the
        # registered checkers apply uniformly to the single emitted frame.
        # A fuller Watch implementation that streams transitions is future work.
        response = await self.Check(request, context)
        yield response

        # Note: Full Watch implementation would monitor status changes
        # and stream updates. For K8s probes, Check is sufficient.

    async def _any_unhealthy_checker(self) -> tuple[str, dict] | None:
        """Run registered checkers in order. Return (name, diagnostics)
        on the first unhealthy result; None if all healthy, all failed
        to determine, or none registered.

        Fail-safe: exceptions from a checker are logged (with consecutive-
        failure escalation) and the iteration continues. Cancellation
        propagates unchanged.

        Aggregate transition state is updated only when at least one
        checker produced a determined result; all-raise cycles do not
        flip state, which keeps transient IPC outages from oscillating
        the logged state.
        """
        if not self._checkers:
            return None

        unhealthy: tuple[str, dict] | None = None
        any_determined = False
        for checker in self._checkers:
            try:
                result = await checker.is_unhealthy()
            except asyncio.CancelledError:
                # Never swallow cancellation — callers (gRPC cancels,
                # graceful shutdown) rely on it propagating.
                raise
            except Exception as e:
                self._register_checker_failure(checker.name, e)
                continue
            self._register_checker_success(checker.name)
            any_determined = True
            if result is not None:
                unhealthy = (checker.name, result)
                break

        if any_determined:
            self._update_aggregate_state(unhealthy)
        return unhealthy

    def _register_checker_failure(self, name: str, exc: BaseException) -> None:
        count = self._checker_failure_counts.get(name, 0) + 1
        self._checker_failure_counts[name] = count
        if count in _CHECKER_FAILURE_LOG_STEPS or count % 1000 == 0:
            logger.warning(
                "Health checker %r failed (%d consecutive failures): %s",
                name,
                count,
                exc,
            )

    def _register_checker_success(self, name: str) -> None:
        prior = self._checker_failure_counts.get(name, 0)
        if prior:
            logger.info(
                "Health checker %r recovered after %d consecutive failures",
                name,
                prior,
            )
            self._checker_failure_counts[name] = 0

    def _update_aggregate_state(self, unhealthy: tuple[str, dict] | None) -> None:
        """Log SERVING <-> NOT_SERVING transitions only; probe-rate
        steady-state logs would drown out everything else.
        """
        is_unhealthy = unhealthy is not None
        if is_unhealthy == self._any_checker_unhealthy:
            return
        if is_unhealthy:
            name, diag = unhealthy  # type: ignore[misc]
            logger.warning(
                "Health checker %r reporting unhealthy; responding NOT_SERVING. Diagnostics: %s",
                name,
                diag,
            )
        else:
            logger.warning("All health checkers reporting healthy; responding SERVING.")
        self._any_checker_unhealthy = is_unhealthy
