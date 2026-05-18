"""Session-scoped worker pool for E2E tests.

Caches workers by ``(engine, model_id, mode, worker_type)`` so consecutive
test classes that need the same backend don't pay the multi-minute worker
startup cost on every class boundary.

The pool holds at most one *active* key at a time. Switching keys evicts
(stops) the cached workers before starting the new set — required because
GPU resources are exclusive. Combined with the collection-ordering hook in
``fixtures/hooks.py`` (which clusters items by backend/model), this keeps
the worker alive across every test class that uses the same backend.

PD-disaggregation paths (prefill+decode) bypass the cache: they're rare,
they hold multiple workers concurrently, and they live in a separate CI
matrix. ``setup_backend`` calls them through the unchanged ``start_workers``
/``stop_workers`` helpers.

Lifecycle is managed via ``pytest_sessionfinish`` in
``fixtures/hooks.py``; an ``atexit`` handler covers the case where pytest
exits before the hook runs (e.g. SIGINT / ``pytest.exit``).
"""

from __future__ import annotations

import atexit
import logging
import threading

from .constants import DEFAULT_STARTUP_TIMEOUT, ConnectionMode, WorkerType
from .worker import Worker, start_workers, stop_workers

logger = logging.getLogger(__name__)


_PoolKey = tuple[str, str, ConnectionMode, WorkerType]


class WorkerPool:
    """One-slot worker cache shared across pytest classes.

    Not safe for concurrent use across pytest-xdist workers — each xdist
    worker would need its own pool with non-overlapping GPU offsets.
    Current CI runs sequentially on GPU runners, so a single-slot pool is
    sufficient.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._key: _PoolKey | None = None
        self._workers: list[Worker] = []
        self._closed = False

    def acquire(
        self,
        *,
        model_id: str,
        engine: str,
        mode: ConnectionMode = ConnectionMode.HTTP,
        count: int = 1,
        worker_type: WorkerType = WorkerType.REGULAR,
        timeout: int = DEFAULT_STARTUP_TIMEOUT,
        log_dir: str | None = None,
    ) -> list[Worker]:
        """Return ``count`` healthy workers for the given key.

        Reuses the cached set when the key matches and the cache holds at
        least ``count`` workers. Any other case evicts the current slot and
        starts a fresh set.

        Raises whatever ``start_workers`` raises on launch failure; the
        cache is left empty in that case.
        """
        if worker_type != WorkerType.REGULAR:
            # PD prefill/decode bypass the cache (see module docstring).
            return start_workers(
                model_id=model_id,
                engine=engine,
                mode=mode,
                count=count,
                worker_type=worker_type,
                timeout=timeout,
                log_dir=log_dir,
            )

        with self._lock:
            if self._closed:
                raise RuntimeError("WorkerPool has been closed")

            key: _PoolKey = (engine, model_id, mode, worker_type)

            if self._key == key and len(self._workers) >= count:
                logger.info(
                    "WorkerPool: reusing %d cached worker(s) for %s",
                    count,
                    key,
                )
                return list(self._workers[:count])

            if self._key is not None:
                logger.info(
                    "WorkerPool: evicting %s to start %s",
                    self._key,
                    key,
                )
                self._evict_locked()

            new_workers = start_workers(
                model_id=model_id,
                engine=engine,
                mode=mode,
                count=count,
                worker_type=worker_type,
                timeout=timeout,
                log_dir=log_dir,
            )
            self._key = key
            self._workers = new_workers
            return list(new_workers)

    def cleanup(self) -> None:
        """Stop all cached workers. Idempotent; safe to call multiple times."""
        with self._lock:
            self._evict_locked()
            self._closed = True

    def _evict_locked(self) -> None:
        if self._workers:
            stop_workers(self._workers)
        self._workers = []
        self._key = None


_POOL: WorkerPool | None = None
_POOL_LOCK = threading.Lock()


def get_pool() -> WorkerPool:
    """Return the session-wide worker pool, creating it on first use."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is None or _POOL._closed:
            _POOL = WorkerPool()
            atexit.register(_POOL.cleanup)
        return _POOL


def cleanup_pool() -> None:
    """Tear down the session-wide pool if it exists. Called from session-end hook."""
    with _POOL_LOCK:
        if _POOL is not None:
            _POOL.cleanup()
