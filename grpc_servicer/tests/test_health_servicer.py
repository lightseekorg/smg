"""Unit tests for SGLangHealthServicer's generic health-check registry.

Covers the checker-agnostic plumbing: aggregation, fail-safe on exception,
consecutive-failure escalation, transition-only logging, and the two
service-name branches of Check(). Concrete checkers are tested in their
own files.

Stdlib-only (unittest + unittest.mock). Run with:
    python -m unittest grpc_servicer.tests.test_health_servicer
"""

import asyncio
import importlib.util
import pathlib
import sys
import time
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from grpc_health.v1 import health_pb2

# Import health_servicer directly by file path. The package __init__
# imports servicer.py, which pulls in msgspec/sglang/etc. — heavy deps
# that aren't needed for this unit test. Direct import keeps the test
# runnable with only grpcio-health-checking available.
_PKG_ROOT = pathlib.Path(__file__).resolve().parents[1] / "smg_grpc_servicer" / "sglang"


def _load_module(module_name: str, file_path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_stub_packages() -> type:
    """Register lightweight stubs for the smg_grpc_servicer package tree so
    health_servicer's ``from smg_grpc_servicer.sglang.health_checks import
    HealthChecker`` resolves without dragging in the real package __init__
    (which imports msgspec/sglang).
    """
    base = _load_module("_smg_health_checks_base", _PKG_ROOT / "health_checks" / "base.py")
    stub_pkg = types.ModuleType("smg_grpc_servicer")
    stub_pkg.__path__ = []
    stub_sglang = types.ModuleType("smg_grpc_servicer.sglang")
    stub_sglang.__path__ = []
    stub_health_checks = types.ModuleType("smg_grpc_servicer.sglang.health_checks")
    stub_health_checks.HealthChecker = base.HealthChecker
    sys.modules.setdefault("smg_grpc_servicer", stub_pkg)
    sys.modules.setdefault("smg_grpc_servicer.sglang", stub_sglang)
    sys.modules.setdefault("smg_grpc_servicer.sglang.health_checks", stub_health_checks)
    return base.HealthChecker


HealthChecker = _install_stub_packages()
_servicer_module = _load_module("_smg_health_servicer_under_test", _PKG_ROOT / "health_servicer.py")
SGLangHealthServicer = _servicer_module.SGLangHealthServicer


class _FakeChecker:
    """Minimal HealthChecker Protocol implementation for tests."""

    def __init__(self, name, results=None, exc=None):
        self.name = name
        # results: callable returning the next result; or a fixed value.
        self._results = results
        self._exc = exc
        self.call_count = 0

    async def is_unhealthy(self):
        self.call_count += 1
        if self._exc is not None:
            raise self._exc
        if callable(self._results):
            return self._results()
        return self._results


def _make_servicer(checkers=()):
    request_manager = SimpleNamespace(
        gracefully_exit=False,
        last_receive_tstamp=time.time(),
        rid_to_state={},
    )
    servicer = SGLangHealthServicer(
        request_manager=request_manager,
        scheduler_info={},
        checkers=checkers,
    )
    # Check()'s checker branches run only after the base SERVING check passes.
    servicer.set_serving()
    return servicer, request_manager


class AggregationTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_checkers_is_healthy(self):
        servicer, _ = _make_servicer(checkers=[])
        self.assertIsNone(await servicer._any_unhealthy_checker())

    async def test_single_healthy_checker_returns_none(self):
        c = _FakeChecker("scheduler_stall", results=None)
        servicer, _ = _make_servicer(checkers=[c])
        self.assertIsNone(await servicer._any_unhealthy_checker())
        self.assertEqual(c.call_count, 1)

    async def test_single_unhealthy_checker_returns_diagnostic(self):
        c = _FakeChecker("scheduler_stall", results={"reason": "stuck for 10m"})
        servicer, _ = _make_servicer(checkers=[c])
        result = await servicer._any_unhealthy_checker()
        self.assertEqual(result, ("scheduler_stall", {"reason": "stuck for 10m"}))

    async def test_first_unhealthy_short_circuits(self):
        """Subsequent checkers are not polled once one reports unhealthy."""
        a = _FakeChecker("a", results={"why": "bad"})
        b = _FakeChecker("b", results=None)
        servicer, _ = _make_servicer(checkers=[a, b])
        result = await servicer._any_unhealthy_checker()
        self.assertEqual(result[0], "a")
        self.assertEqual(a.call_count, 1)
        self.assertEqual(b.call_count, 0)

    async def test_all_healthy_returns_none(self):
        a = _FakeChecker("a", results=None)
        b = _FakeChecker("b", results=None)
        servicer, _ = _make_servicer(checkers=[a, b])
        self.assertIsNone(await servicer._any_unhealthy_checker())
        self.assertEqual(a.call_count, 1)
        self.assertEqual(b.call_count, 1)


class FailSafeTests(unittest.IsolatedAsyncioTestCase):
    async def test_checker_exception_is_swallowed(self):
        """Exceptions do not propagate out of _any_unhealthy_checker."""
        c = _FakeChecker("flaky", exc=RuntimeError("boom"))
        servicer, _ = _make_servicer(checkers=[c])
        self.assertIsNone(await servicer._any_unhealthy_checker())

    async def test_exception_does_not_block_subsequent_checkers(self):
        a = _FakeChecker("raises", exc=RuntimeError("boom"))
        b = _FakeChecker("reports", results={"why": "bad"})
        servicer, _ = _make_servicer(checkers=[a, b])
        result = await servicer._any_unhealthy_checker()
        self.assertEqual(result[0], "reports")

    async def test_cancelled_error_propagates(self):
        c = _FakeChecker("cancelled", exc=asyncio.CancelledError())
        servicer, _ = _make_servicer(checkers=[c])
        with self.assertRaises(asyncio.CancelledError):
            await servicer._any_unhealthy_checker()

    async def test_all_exceptions_do_not_flip_aggregate_state(self):
        """Transient IPC outages shouldn't oscillate the aggregate state."""
        c = _FakeChecker("flaky", exc=TimeoutError("slow"))
        servicer, _ = _make_servicer(checkers=[c])
        # Seed prior state as "unhealthy" to prove it's not reset to healthy
        # by an all-raise cycle.
        servicer._any_checker_unhealthy = True
        self.assertIsNone(await servicer._any_unhealthy_checker())
        self.assertTrue(servicer._any_checker_unhealthy)


class FailureEscalationTests(unittest.IsolatedAsyncioTestCase):
    async def test_failure_count_increments_and_resets_on_success(self):
        """Escalation counter tracks consecutive failures per checker."""
        fail_first = {"n": 3}

        def results():
            fail_first["n"] -= 1
            if fail_first["n"] >= 0:
                raise RuntimeError("boom")
            return None

        c = _FakeChecker("recoverer", results=results)
        servicer, _ = _make_servicer(checkers=[c])

        for _ in range(3):
            await servicer._any_unhealthy_checker()
        self.assertEqual(servicer._checker_failure_counts["recoverer"], 3)

        # Successful call resets the counter.
        await servicer._any_unhealthy_checker()
        self.assertEqual(servicer._checker_failure_counts["recoverer"], 0)


class TransitionLoggingTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_unhealthy_detection_updates_state(self):
        c = _FakeChecker("a", results={"why": "bad"})
        servicer, _ = _make_servicer(checkers=[c])
        self.assertFalse(servicer._any_checker_unhealthy)
        await servicer._any_unhealthy_checker()
        self.assertTrue(servicer._any_checker_unhealthy)

    async def test_recovery_clears_state(self):
        calls = {"n": 0}

        def results():
            calls["n"] += 1
            return {"why": "bad"} if calls["n"] == 1 else None

        c = _FakeChecker("a", results=results)
        servicer, _ = _make_servicer(checkers=[c])
        await servicer._any_unhealthy_checker()
        self.assertTrue(servicer._any_checker_unhealthy)
        await servicer._any_unhealthy_checker()
        self.assertFalse(servicer._any_checker_unhealthy)


class CheckResponseTests(unittest.IsolatedAsyncioTestCase):
    """End-to-end through Check() for both service-name branches."""

    async def test_overall_server_not_serving_when_checker_unhealthy(self):
        c = _FakeChecker("a", results={"why": "bad"})
        servicer, _ = _make_servicer(checkers=[c])
        request = health_pb2.HealthCheckRequest(service=SGLangHealthServicer.OVERALL_SERVER)
        response = await servicer.Check(request, context=None)
        self.assertEqual(response.status, health_pb2.HealthCheckResponse.NOT_SERVING)

    async def test_overall_server_serving_when_all_healthy(self):
        c = _FakeChecker("a", results=None)
        servicer, _ = _make_servicer(checkers=[c])
        request = health_pb2.HealthCheckRequest(service=SGLangHealthServicer.OVERALL_SERVER)
        response = await servicer.Check(request, context=None)
        self.assertEqual(response.status, health_pb2.HealthCheckResponse.SERVING)

    async def test_sglang_service_not_serving_when_checker_unhealthy(self):
        c = _FakeChecker("a", results={"why": "bad"})
        servicer, _ = _make_servicer(checkers=[c])
        request = health_pb2.HealthCheckRequest(service=SGLangHealthServicer.SGLANG_SERVICE)
        response = await servicer.Check(request, context=None)
        self.assertEqual(response.status, health_pb2.HealthCheckResponse.NOT_SERVING)

    async def test_sglang_service_serving_when_no_checkers_registered(self):
        """Empty registry preserves legacy behavior."""
        servicer, _ = _make_servicer(checkers=[])
        request = health_pb2.HealthCheckRequest(service=SGLangHealthServicer.SGLANG_SERVICE)
        response = await servicer.Check(request, context=None)
        self.assertEqual(response.status, health_pb2.HealthCheckResponse.SERVING)

    async def test_unknown_service_returns_service_unknown(self):
        servicer, _ = _make_servicer(checkers=[])
        request = health_pb2.HealthCheckRequest(service="not.a.real.service")
        context = Mock()
        response = await servicer.Check(request, context=context)
        self.assertEqual(response.status, health_pb2.HealthCheckResponse.SERVICE_UNKNOWN)
        context.set_code.assert_called_once()


if __name__ == "__main__":
    unittest.main()
