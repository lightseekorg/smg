# SPDX-License-Identifier: Apache-2.0
"""Tests for VllmHealthServicer Watch() continuous streaming."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from grpc_health.v1 import health_pb2
from smg_grpc_servicer.vllm.health_servicer import VllmHealthServicer

SERVING = health_pb2.HealthCheckResponse.SERVING
NOT_SERVING = health_pb2.HealthCheckResponse.NOT_SERVING
SERVICE_UNKNOWN = health_pb2.HealthCheckResponse.SERVICE_UNKNOWN


@pytest.fixture
def async_llm():
    mock = MagicMock()
    mock.check_health = AsyncMock()
    return mock


@pytest.fixture
def servicer(async_llm):
    return VllmHealthServicer(async_llm)


@pytest.mark.asyncio
async def test_watch_sends_initial_serving(servicer, request_msg, grpc_context):
    """Watch must immediately send SERVING when engine is healthy."""
    request_msg.service = ""

    received = []
    async for response in servicer.Watch(request_msg, grpc_context):
        received.append(response.status)
        if len(received) == 1:
            servicer.set_not_serving()

    assert received[0] == SERVING


@pytest.mark.asyncio
async def test_watch_yields_on_engine_failure(servicer, request_msg, grpc_context, async_llm):
    """Watch must send NOT_SERVING when check_health() starts failing."""
    request_msg.service = ""
    servicer.WATCH_POLL_INTERVAL_S = 0.05

    received = []
    poll_count = 0

    original_check = async_llm.check_health

    async def check_health_with_failure():
        nonlocal poll_count
        poll_count += 1
        if poll_count >= 3:
            raise Exception("engine dead")
        await original_check()

    async_llm.check_health = AsyncMock(side_effect=check_health_with_failure)

    async def stop_eventually():
        await asyncio.sleep(0.5)
        servicer.set_not_serving()

    task = asyncio.create_task(stop_eventually())

    async for response in servicer.Watch(request_msg, grpc_context):
        received.append(response.status)

    await task
    assert SERVING in received
    assert NOT_SERVING in received
    serving_idx = received.index(SERVING)
    not_serving_idx = received.index(NOT_SERVING)
    assert serving_idx < not_serving_idx


@pytest.mark.asyncio
async def test_watch_exits_on_shutdown(servicer, request_msg, grpc_context):
    """set_not_serving() must wake Watch and end the stream."""
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
    request_msg.service = "fake.Service"

    received = []
    async for response in servicer.Watch(request_msg, grpc_context):
        received.append(response.status)

    assert received == [SERVICE_UNKNOWN]
    grpc_context.set_code.assert_not_called()


@pytest.mark.asyncio
async def test_watch_no_duplicate_on_stable_status(servicer, request_msg, grpc_context):
    """Stable SERVING must not yield duplicates across poll cycles."""
    request_msg.service = ""
    servicer.WATCH_POLL_INTERVAL_S = 0.05

    received = []

    async def stop_after_polls():
        await asyncio.sleep(0.2)
        servicer.set_not_serving()

    task = asyncio.create_task(stop_after_polls())

    async for response in servicer.Watch(request_msg, grpc_context):
        received.append(response.status)

    await task
    assert received == [SERVING, NOT_SERVING]


@pytest.mark.asyncio
async def test_watch_shutdown_overrides_healthy(servicer, request_msg, grpc_context, async_llm):
    """After set_not_serving(), Watch returns NOT_SERVING even if
    check_health() would succeed."""
    servicer.set_not_serving()
    request_msg.service = ""

    received = []
    async for response in servicer.Watch(request_msg, grpc_context):
        received.append(response.status)

    assert received == [NOT_SERVING]
    async_llm.check_health.assert_not_awaited()
