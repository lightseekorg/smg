"""Unit tests for graceful shutdown wiring in the SGLang gRPC servicer.

Covers the drain-signal plumbing added for K8s scale-down: the request
manager's begin_drain() method and the health servicer's response to the
gracefully_exit flag.
"""

import inspect
from unittest.mock import MagicMock

import pytest
from grpc_health.v1 import health_pb2
from smg_grpc_servicer.sglang.health_servicer import SGLangHealthServicer
from smg_grpc_servicer.sglang.request_manager import GrpcRequestManager

_SENTINEL_RID = "rid-under-drain"
_SENTINEL_TASK = "fake-task"


def _make_bare_request_manager() -> GrpcRequestManager:
    """Return a GrpcRequestManager with only the attributes begin_drain touches.

    The real __init__ constructs ZMQ sockets, a scheduler channel, etc., none
    of which begin_drain depends on. object.__new__ + explicit attribute
    assignment gives us an isolated unit under test. We seed rid_to_state
    and asyncio_tasks with sentinels so the non-destructive assertion is
    meaningful (an empty-state fixture would pass trivially).
    """
    mgr = object.__new__(GrpcRequestManager)
    mgr.gracefully_exit = False
    mgr.rid_to_state = {_SENTINEL_RID: MagicMock(finished=False)}
    mgr.asyncio_tasks = {_SENTINEL_TASK}
    return mgr


def test_begin_drain_sets_flag():
    mgr = _make_bare_request_manager()

    mgr.begin_drain()

    assert mgr.gracefully_exit is True
    # begin_drain must not cancel tasks or evict in-flight request state.
    assert _SENTINEL_RID in mgr.rid_to_state
    assert mgr.asyncio_tasks == {_SENTINEL_TASK}


def test_begin_drain_idempotent():
    mgr = _make_bare_request_manager()

    mgr.begin_drain()
    mgr.begin_drain()

    assert mgr.gracefully_exit is True
    assert _SENTINEL_RID in mgr.rid_to_state
    assert mgr.asyncio_tasks == {_SENTINEL_TASK}


def test_handle_loop_not_gated_on_gracefully_exit():
    """Regression guard for the drain invariant.

    handle_loop must run `while True:`, not `while not self.gracefully_exit:`.
    If it gates on the flag, begin_drain() exits the loop on the next ZMQ
    recv, stalling in-flight streaming requests and defeating the drain.
    See docs/superpowers/specs/2026-04-16-grpc-graceful-shutdown-design.md.
    """
    src = inspect.getsource(GrpcRequestManager.handle_loop)

    assert "while True:" in src, (
        "handle_loop must run `while True:` so scheduler outputs keep "
        "flowing to in-flight streams during drain"
    )
    assert "while not self.gracefully_exit" not in src, (
        "handle_loop must not gate on gracefully_exit; doing so stalls "
        "streaming requests after begin_drain() is called"
    )


@pytest.mark.asyncio
async def test_health_reflects_drain_flag():
    request_manager = MagicMock()
    request_manager.gracefully_exit = False

    health = SGLangHealthServicer(request_manager=request_manager, scheduler_info={})
    health.set_serving()

    ctx = MagicMock()
    req = health_pb2.HealthCheckRequest(service="")

    # Sanity check: SERVING before drain starts.
    resp = await health.Check(req, ctx)
    assert resp.status == health_pb2.HealthCheckResponse.SERVING

    # After begin_drain flips the flag, health must flip to NOT_SERVING.
    request_manager.gracefully_exit = True
    resp = await health.Check(req, ctx)
    assert resp.status == health_pb2.HealthCheckResponse.NOT_SERVING
