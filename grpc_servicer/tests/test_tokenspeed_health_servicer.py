"""Unit tests for ``smg_grpc_servicer.tokenspeed.health_servicer``."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import grpc
import pytest
from grpc_health.v1 import health_pb2  # noqa: E402
from smg_grpc_servicer.tokenspeed.health_servicer import (  # noqa: E402
    TokenSpeedHealthServicer,
)


@dataclass
class FakeEngine:
    gracefully_exit: bool = False
    last_receive_tstamp: float = 0.0
    rid_to_state: dict[str, Any] = field(default_factory=dict)


@pytest.fixture
def servicer() -> TokenSpeedHealthServicer:
    return TokenSpeedHealthServicer(
        async_llm=FakeEngine(),
        scheduler_info={},
    )


@pytest.mark.asyncio
async def test_initial_state_is_not_serving(servicer: TokenSpeedHealthServicer):
    ctx = MagicMock(spec=grpc.aio.ServicerContext)
    resp = await servicer.Check(health_pb2.HealthCheckRequest(service=""), ctx)
    assert resp.status == health_pb2.HealthCheckResponse.NOT_SERVING


@pytest.mark.asyncio
async def test_set_serving_flips_both_levels(servicer: TokenSpeedHealthServicer):
    ctx = MagicMock(spec=grpc.aio.ServicerContext)
    servicer.set_serving()

    # overall
    resp = await servicer.Check(health_pb2.HealthCheckRequest(service=""), ctx)
    assert resp.status == health_pb2.HealthCheckResponse.SERVING

    # specific
    resp = await servicer.Check(
        health_pb2.HealthCheckRequest(service=servicer.TOKENSPEED_SERVICE), ctx
    )
    assert resp.status == health_pb2.HealthCheckResponse.SERVING


@pytest.mark.asyncio
async def test_shutdown_flips_back(servicer: TokenSpeedHealthServicer):
    ctx = MagicMock(spec=grpc.aio.ServicerContext)
    servicer.set_serving()
    servicer.async_llm.gracefully_exit = True
    resp = await servicer.Check(health_pb2.HealthCheckRequest(service=""), ctx)
    assert resp.status == health_pb2.HealthCheckResponse.NOT_SERVING


@pytest.mark.asyncio
async def test_unknown_service_returns_unknown(servicer: TokenSpeedHealthServicer):
    ctx = MagicMock(spec=grpc.aio.ServicerContext)
    resp = await servicer.Check(health_pb2.HealthCheckRequest(service="bogus.Service"), ctx)
    assert resp.status == health_pb2.HealthCheckResponse.SERVICE_UNKNOWN
    ctx.set_code.assert_called_once_with(grpc.StatusCode.NOT_FOUND)


@pytest.mark.asyncio
async def test_stuck_scheduler_flips_to_not_serving(
    servicer: TokenSpeedHealthServicer,
):
    ctx = MagicMock(spec=grpc.aio.ServicerContext)
    servicer.set_serving()
    # Simulate "pending requests, but scheduler hasn't pushed output for 45s"
    servicer.async_llm.last_receive_tstamp = time.time() - 45
    servicer.async_llm.rid_to_state["rid-1"] = object()

    resp = await servicer.Check(
        health_pb2.HealthCheckRequest(service=servicer.TOKENSPEED_SERVICE), ctx
    )
    assert resp.status == health_pb2.HealthCheckResponse.NOT_SERVING


@pytest.mark.asyncio
async def test_recent_activity_keeps_serving(servicer: TokenSpeedHealthServicer):
    ctx = MagicMock(spec=grpc.aio.ServicerContext)
    servicer.set_serving()
    servicer.async_llm.last_receive_tstamp = time.time() - 1
    servicer.async_llm.rid_to_state["rid-1"] = object()
    resp = await servicer.Check(
        health_pb2.HealthCheckRequest(service=servicer.TOKENSPEED_SERVICE), ctx
    )
    assert resp.status == health_pb2.HealthCheckResponse.SERVING
