"""Pluggable health check protocol for SGLangHealthServicer.

Checkers are focused predicates that inspect scheduler state and decide
whether the engine is healthy. The servicer aggregates registered
checkers: any one checker reporting unhealthy flips the gRPC
Health.Check response to NOT_SERVING.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class HealthChecker(Protocol):
    """A single health predicate.

    Implementations must:

    * Set ``name`` to a short stable identifier used in logs and
      diagnostics (e.g. ``"scheduler_stall"``, ``"decoder_backpressure"``).
    * Make ``is_unhealthy`` idempotent and cheap — the servicer may call
      it as often as every K8s probe arrives. If the underlying signal
      is expensive to read, maintain an internal TTL cache.
    * Return ``None`` when the component is healthy.
    * Return a free-form diagnostic dict when the component is unhealthy.
      The dict is logged on transitions and surfaced to operators.
    * Raise on "unable to determine" conditions (IPC error, timeout,
      schema drift, etc.). The servicer catches all exceptions centrally,
      counts consecutive failures, and fails-safe to SERVING. Cancellation
      must propagate.
    """

    name: str

    async def is_unhealthy(self) -> dict | None: ...
