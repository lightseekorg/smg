# SPDX-License-Identifier: Apache-2.0
"""Shared Watch() continuous streaming for gRPC health servicers."""

import asyncio
import inspect
import logging
from collections.abc import AsyncIterator

import grpc
from grpc_health.v1 import health_pb2

logger = logging.getLogger(__name__)


class HealthWatchMixin:
    """Continuous Watch() streaming for gRPC health servicers.

    Implements the gRPC Health Checking Protocol Watch RPC as a long-lived
    server-streaming response that sends updates on status change.

    Subclasses must:
    1. Call self._init_watch() in __init__
    2. Implement _compute_watch_status(service_name) -> ServingStatus
    3. Implement _is_shutting_down() -> bool
    4. Call self._notify_shutdown() in set_not_serving()
    """

    WATCH_POLL_INTERVAL_S = 5.0

    def _init_watch(self) -> None:
        """Initialize Watch state. Must be called in subclass __init__.

        Note: on Python 3.10-3.11, asyncio.Event() captures the running
        event loop at construction time. Both SGLang and vLLM construct
        their servicers within the async server context (verified against
        smg sglang/server.py and vllm grpc_server.py), so this is safe.
        Python 3.12+ removed the loop binding entirely.
        """
        self._watch_shutdown_event = asyncio.Event()
        self._watch_notified_shutdown = False

    def _notify_shutdown(self) -> None:
        """Wake all Watch streams to detect shutdown immediately.
        Must be called in subclass set_not_serving(). Sets
        _watch_notified_shutdown so _is_shutting_down() implementations
        can check it alongside their own shutdown flags."""
        self._watch_notified_shutdown = True
        self._watch_shutdown_event.set()

    def _compute_watch_status(self, service_name: str) -> int:
        """Compute current health status for the given service.

        Must not call context.set_code() -- that would pollute the
        streaming response. Return a ServingStatus enum value instead.

        May be overridden as async def for servicers that need I/O
        (e.g., VllmHealthServicer calls await async_llm.check_health()).
        """
        raise NotImplementedError(f"{type(self).__name__} must implement _compute_watch_status()")

    def _is_shutting_down(self) -> bool:
        """Return True if the server is shutting down."""
        raise NotImplementedError(f"{type(self).__name__} must implement _is_shutting_down()")

    async def _resolve_watch_status(self, service_name: str) -> int:
        """Call _compute_watch_status, handling both sync and async impls."""
        result = self._compute_watch_status(service_name)
        if inspect.isawaitable(result):
            return await result
        return result

    async def Watch(
        self,
        request: health_pb2.HealthCheckRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[health_pb2.HealthCheckResponse]:
        """gRPC Health Watch -- continuous streaming implementation.

        Behavior per gRPC Health Checking Protocol:
        - Immediately sends the current serving status
        - Sends a new message whenever status changes
        - Stream ends on server shutdown or client cancellation

        Deviation from spec: for unknown services, the stream sends
        SERVICE_UNKNOWN once then exits. The spec says to keep the stream
        open for dynamic service registration, but smg services are
        statically defined and never registered at runtime.
        """
        service_name = request.service
        logger.debug("Health watch request for service: '%s'", service_name)

        last_status = None
        try:
            while True:
                status = await self._resolve_watch_status(service_name)

                if status != last_status:
                    yield health_pb2.HealthCheckResponse(status=status)
                    last_status = status

                if self._is_shutting_down():
                    return

                if status == health_pb2.HealthCheckResponse.SERVICE_UNKNOWN:
                    return

                try:
                    await asyncio.wait_for(
                        self._watch_shutdown_event.wait(),
                        timeout=self.WATCH_POLL_INTERVAL_S,
                    )
                except asyncio.TimeoutError:
                    pass

        except asyncio.CancelledError:
            logger.debug(
                "Health watch cancelled by client for service: '%s'",
                service_name,
            )
