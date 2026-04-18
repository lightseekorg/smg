# SPDX-License-Identifier: Apache-2.0
"""Tests for SGLangHealthServicer Watch() continuous streaming."""

import asyncio
from unittest.mock import MagicMock

import pytest
from grpc_health.v1 import health_pb2
from smg_grpc_servicer.sglang.health_servicer import SGLangHealthServicer

SERVING = health_pb2.HealthCheckResponse.SERVING
NOT_SERVING = health_pb2.HealthCheckResponse.NOT_SERVING
SERVICE_UNKNOWN = health_pb2.HealthCheckResponse.SERVICE_UNKNOWN


@pytest.fixture
def request_manager():
    mgr = MagicMock()
    mgr.gracefully_exit = False
    # float("inf") ensures scheduler-responsiveness timeout (30s) never triggers
    mgr.last_receive_tstamp = float("inf")
    mgr.rid_to_state = {}
    return mgr


@pytest.fixture
def servicer(request_manager):
    return SGLangHealthServicer(
        request_manager=request_manager,
        scheduler_info={"model_path": "test"},
    )


@pytest.mark.asyncio
async def test_watch_sends_initial_status(servicer, request_msg, grpc_context):
    """Watch must immediately send the current status."""
    servicer.set_serving()
    request_msg.service = ""

    received = []
    async for response in servicer.Watch(request_msg, grpc_context):
        received.append(response.status)
        if len(received) == 1:
            servicer.set_not_serving()

    assert received[0] == SERVING


@pytest.mark.asyncio
async def test_watch_yields_on_status_change(servicer, request_msg, grpc_context, request_manager):
    """Watch must send a new response when status changes."""
    servicer.set_serving()
    request_msg.service = ""

    received = []

    async def trigger_shutdown():
        await asyncio.sleep(0.1)
        servicer.set_not_serving()

    task = asyncio.create_task(trigger_shutdown())

    async for response in servicer.Watch(request_msg, grpc_context):
        received.append(response.status)

    await task
    assert received == [SERVING, NOT_SERVING]


@pytest.mark.asyncio
async def test_watch_exits_on_shutdown(servicer, request_msg, grpc_context):
    """set_not_serving() must cause Watch to end the stream."""
    servicer.set_serving()
    request_msg.service = ""

    async def trigger_shutdown():
        await asyncio.sleep(0.05)
        servicer.set_not_serving()

    task = asyncio.create_task(trigger_shutdown())

    received = []
    async for response in servicer.Watch(request_msg, grpc_context):
        received.append(response.status)

    await task
    assert len(received) >= 1
    assert received[-1] == NOT_SERVING


@pytest.mark.asyncio
async def test_watch_handles_client_cancel(servicer, request_msg, grpc_context):
    """Task cancellation (real client disconnect) must not raise unexpected errors."""
    servicer.set_serving()
    request_msg.service = ""

    async def consume_forever():
        async for _ in servicer.Watch(request_msg, grpc_context):
            pass

    task = asyncio.create_task(consume_forever())
    await asyncio.sleep(0.05)
    task.cancel()
    # Watch() catches CancelledError internally. The task may complete
    # normally or propagate cancellation depending on asyncio internals.
    # Either outcome is correct -- verify no unexpected exception.
    try:
        await task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_watch_unknown_service(servicer, request_msg, grpc_context):
    """Unknown service: single SERVICE_UNKNOWN, no context.set_code()."""
    servicer.set_serving()
    request_msg.service = "nonexistent.Service"

    received = []
    async for response in servicer.Watch(request_msg, grpc_context):
        received.append(response.status)

    assert received == [SERVICE_UNKNOWN]
    grpc_context.set_code.assert_not_called()


@pytest.mark.asyncio
async def test_watch_no_duplicate_on_stable_status(servicer, request_msg, grpc_context):
    """Stable status must not yield duplicate responses."""
    servicer.set_serving()
    request_msg.service = ""

    original_interval = servicer.WATCH_POLL_INTERVAL_S
    servicer.WATCH_POLL_INTERVAL_S = 0.05

    received = []

    async def stop_after_polls():
        await asyncio.sleep(0.2)
        servicer.set_not_serving()

    task = asyncio.create_task(stop_after_polls())

    async for response in servicer.Watch(request_msg, grpc_context):
        received.append(response.status)

    await task
    servicer.WATCH_POLL_INTERVAL_S = original_interval

    assert received == [SERVING, NOT_SERVING]


@pytest.mark.asyncio
async def test_watch_detects_graceful_exit_via_poll(
    servicer, request_msg, grpc_context, request_manager
):
    """Watch must detect request_manager.gracefully_exit on next poll cycle,
    even without _notify_shutdown() (simulates external shutdown signal)."""
    servicer.set_serving()
    request_msg.service = ""

    servicer.WATCH_POLL_INTERVAL_S = 0.05

    received = []

    async def trigger_graceful_exit():
        await asyncio.sleep(0.1)
        request_manager.gracefully_exit = True

    task = asyncio.create_task(trigger_graceful_exit())

    async for response in servicer.Watch(request_msg, grpc_context):
        received.append(response.status)

    await task
    assert received[0] == SERVING
    assert received[-1] == NOT_SERVING
