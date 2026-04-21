"""Unit tests for PrefillBootstrapQueueStalledChecker.

The checker detects when a disaggregation-prefill engine's bootstrap
queue is saturated with no forward progress — signature:
``bootstrap >= limit AND inflight == 0 AND waiting == 0`` on any
scheduler/DP rank. Cause-agnostic: any sub-system that stalls the
bootstrap pipeline (transport failure, decoder backpressure, control-
channel breakage, decoder OOM, etc.) produces this signature.

Stdlib-only (unittest + unittest.mock). Run with:
    python -m unittest grpc_servicer.tests.test_prefill_bootstrap_queue_stalled
"""

import importlib.util
import pathlib
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

# Direct-import the checker module to avoid pulling sglang/msgspec.
_PKG_ROOT = pathlib.Path(__file__).resolve().parents[1] / "smg_grpc_servicer" / "sglang"
_spec = importlib.util.spec_from_file_location(
    "_smg_prefill_bootstrap_queue_under_test",
    _PKG_ROOT / "health_checks" / "prefill_bootstrap_queue.py",
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
PrefillBootstrapQueueStalledChecker = _module.PrefillBootstrapQueueStalledChecker


def _make_load(*, bootstrap, inflight, waiting, dp_rank=0):
    """Build a load object that looks like GetLoadsReqOutput.

    The checker reads prefill counts through ``.disaggregation`` and
    computes the real waiting count as
    ``num_waiting_reqs - prefill_prealloc_queue_reqs`` because sglang
    includes the bootstrap queue in the waiting total on the prefill side.
    """
    return SimpleNamespace(
        dp_rank=dp_rank,
        num_waiting_reqs=bootstrap + waiting,
        num_running_reqs=inflight,
        disaggregation=SimpleNamespace(
            prefill_prealloc_queue_reqs=bootstrap,
            prefill_inflight_queue_reqs=inflight,
        ),
    )


def _make_checker(*, mode, limit, loads_result=None, loads_side_effect=None):
    request_manager = SimpleNamespace()
    request_manager.get_loads = AsyncMock(return_value=loads_result, side_effect=loads_side_effect)
    server_args = SimpleNamespace(disaggregation_mode=mode, max_queued_requests=limit)
    return (
        PrefillBootstrapQueueStalledChecker(
            request_manager=request_manager, server_args=server_args
        ),
        request_manager,
    )


class ActivationGuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_non_disaggregation_mode_is_healthy(self):
        """Monolithic (non-PD) servers must never trip the checker."""
        checker, rm = _make_checker(mode="null", limit=16, loads_result=[])
        self.assertIsNone(await checker.is_unhealthy())
        rm.get_loads.assert_not_called()

    async def test_decoder_mode_is_healthy(self):
        """The stall signature is meaningful only for prefill-mode engines."""
        checker, rm = _make_checker(mode="decode", limit=16, loads_result=[])
        self.assertIsNone(await checker.is_unhealthy())
        rm.get_loads.assert_not_called()

    async def test_missing_max_queued_requests_is_healthy(self):
        """Without a positive queue cap there's no 'at limit' condition."""
        checker, _ = _make_checker(mode="prefill", limit=None, loads_result=[])
        self.assertIsNone(await checker.is_unhealthy())

        checker, _ = _make_checker(mode="prefill", limit=0, loads_result=[])
        self.assertIsNone(await checker.is_unhealthy())


class SignatureTests(unittest.IsolatedAsyncioTestCase):
    async def test_idle_is_healthy(self):
        loads = [_make_load(bootstrap=0, inflight=0, waiting=0)]
        checker, _ = _make_checker(mode="prefill", limit=16, loads_result=loads)
        self.assertIsNone(await checker.is_unhealthy())

    async def test_busy_but_healthy_returns_none(self):
        """Queue at cap but inflight > 0 means real work is flowing."""
        loads = [_make_load(bootstrap=16, inflight=8, waiting=0)]
        checker, _ = _make_checker(mode="prefill", limit=16, loads_result=loads)
        self.assertIsNone(await checker.is_unhealthy())

    async def test_stalled_returns_diagnostics(self):
        """The exact stall pattern: slots full, zero flowing."""
        loads = [_make_load(bootstrap=16, inflight=0, waiting=0, dp_rank=2)]
        checker, _ = _make_checker(mode="prefill", limit=16, loads_result=loads)
        result = await checker.is_unhealthy()
        self.assertEqual(
            result,
            {"dp_rank": 2, "bootstrap": 16, "inflight": 0, "waiting": 0, "limit": 16},
        )

    async def test_boundary_below_limit_is_healthy(self):
        """Bootstrap one under the limit must NOT trip — guards `bootstrap >= limit`."""
        loads = [_make_load(bootstrap=15, inflight=0, waiting=0)]
        checker, _ = _make_checker(mode="prefill", limit=16, loads_result=loads)
        self.assertIsNone(await checker.is_unhealthy())

    async def test_waiting_positive_is_healthy(self):
        """Real main-queue wait > 0 means forward progress will happen.

        Constructs ``num_waiting_reqs`` directly (not via _make_load) so
        the ``waiting = max(0, num_waiting_reqs - bootstrap)`` subtraction
        is actually exercised rather than baked into the fixture.
        """
        load = SimpleNamespace(
            dp_rank=0,
            num_waiting_reqs=20,
            num_running_reqs=0,
            disaggregation=SimpleNamespace(
                prefill_prealloc_queue_reqs=16,
                prefill_inflight_queue_reqs=0,
            ),
        )
        checker, _ = _make_checker(mode="prefill", limit=16, loads_result=[load])
        self.assertIsNone(await checker.is_unhealthy())


class MultiDPTests(unittest.IsolatedAsyncioTestCase):
    async def test_any_rank_stalled_trips_checker(self):
        """Multi-DP: one stuck rank is enough — future requests to it will hang."""
        loads = [
            _make_load(bootstrap=4, inflight=2, waiting=0, dp_rank=0),
            _make_load(bootstrap=16, inflight=0, waiting=0, dp_rank=1),
        ]
        checker, _ = _make_checker(mode="prefill", limit=16, loads_result=loads)
        result = await checker.is_unhealthy()
        self.assertEqual(result["dp_rank"], 1)

    async def test_all_ranks_healthy_multi_dp_returns_none(self):
        """Symmetric counterpart — pins the loop semantics."""
        loads = [
            _make_load(bootstrap=4, inflight=2, waiting=0, dp_rank=0),
            _make_load(bootstrap=16, inflight=4, waiting=0, dp_rank=1),
            _make_load(bootstrap=8, inflight=1, waiting=0, dp_rank=2),
        ]
        checker, _ = _make_checker(mode="prefill", limit=16, loads_result=loads)
        self.assertIsNone(await checker.is_unhealthy())


class IPCErrorPropagationTests(unittest.IsolatedAsyncioTestCase):
    """The checker does NOT catch IPC errors; the servicer handles them centrally."""

    async def test_ipc_runtime_error_propagates(self):
        checker, _ = _make_checker(
            mode="prefill", limit=16, loads_side_effect=RuntimeError("scheduler down")
        )
        with self.assertRaises(RuntimeError):
            await checker.is_unhealthy()

    async def test_ipc_timeout_propagates(self):
        checker, _ = _make_checker(mode="prefill", limit=16, loads_side_effect=TimeoutError("slow"))
        with self.assertRaises(TimeoutError):
            await checker.is_unhealthy()


class SchemaDriftCanaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_ranks_missing_disagg_warns_once(self):
        """Schema-drift canary: non-empty loads with no `.disaggregation`."""
        loads = [SimpleNamespace(dp_rank=0, num_waiting_reqs=0, num_running_reqs=0)]
        checker, _ = _make_checker(mode="prefill", limit=16, loads_result=loads)

        self.assertIsNone(await checker.is_unhealthy())
        self.assertTrue(checker._warned_all_ranks_missing_disagg)

        # Second call must not re-warn (latched).
        self.assertIsNone(await checker.is_unhealthy())
        self.assertTrue(checker._warned_all_ranks_missing_disagg)


class TTLCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_repeated_calls_within_ttl_hit_cache(self):
        loads = [_make_load(bootstrap=16, inflight=0, waiting=0)]
        checker, rm = _make_checker(mode="prefill", limit=16, loads_result=loads)

        await checker.is_unhealthy()
        await checker.is_unhealthy()
        await checker.is_unhealthy()

        self.assertEqual(rm.get_loads.call_count, 1)

    async def test_call_after_ttl_expiry_refreshes(self):
        loads = [_make_load(bootstrap=16, inflight=0, waiting=0)]
        checker, rm = _make_checker(mode="prefill", limit=16, loads_result=loads)

        await checker.is_unhealthy()
        # Force expiry by rewinding the cache timestamp past the TTL.
        checker._cache_ts = time.monotonic() - 10.0
        await checker.is_unhealthy()

        self.assertEqual(rm.get_loads.call_count, 2)


if __name__ == "__main__":
    unittest.main()
