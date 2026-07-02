"""GetServerInfo advertises the metrics sidecar address in ``server_args``.

These exercise the real servicer ``GetServerInfo`` RPCs, so they require the
engine packages (``tokenspeed`` / ``sglang``) to be importable and are skipped
otherwise. The engine-free advertisement logic itself is covered directly in
``test_metrics_sidecar.py`` via ``metrics_server_args``.

Run with: pytest grpc_servicer/tests/test_servicer_metrics_advertise.py
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("smg_grpc_proto")


def _struct_to_dict(struct) -> dict:
    return {k: struct[k] for k in struct.keys()}


def test_tokenspeed_get_server_info_advertises_metrics():
    pytest.importorskip("tokenspeed")
    from smg_grpc_servicer.tokenspeed.servicer import TokenSpeedSchedulerServicer

    servicer = TokenSpeedSchedulerServicer.__new__(TokenSpeedSchedulerServicer)
    # Bypass __init__ (which spins up AsyncLLM loops); set just what
    # GetServerInfo touches.
    servicer.async_llm = SimpleNamespace(rid_to_state={})
    servicer.server_args = SimpleNamespace(host="10.0.0.5", served_model_name="m")
    servicer.scheduler_info = {"max_total_num_tokens": 0}
    servicer.health_servicer = None
    servicer.metrics_port = 9101
    servicer.start_time = 0.0

    resp = asyncio.run(servicer.GetServerInfo(object(), object()))
    args = _struct_to_dict(resp.server_args)
    assert args["metrics_port"] == 9101
    assert args["metrics_url"] == "http://10.0.0.5:9101/metrics"


def test_tokenspeed_get_server_info_omits_metrics_when_disabled():
    pytest.importorskip("tokenspeed")
    from smg_grpc_servicer.tokenspeed.servicer import TokenSpeedSchedulerServicer

    servicer = TokenSpeedSchedulerServicer.__new__(TokenSpeedSchedulerServicer)
    servicer.async_llm = SimpleNamespace(rid_to_state={})
    servicer.server_args = SimpleNamespace(host="10.0.0.5", served_model_name="m")
    servicer.scheduler_info = {"max_total_num_tokens": 0}
    servicer.health_servicer = None
    servicer.metrics_port = None
    servicer.start_time = 0.0

    resp = asyncio.run(servicer.GetServerInfo(object(), object()))
    args = _struct_to_dict(resp.server_args)
    assert "metrics_port" not in args
    assert "metrics_url" not in args


def test_tokenspeed_load_snapshot_reports_running_only():
    pytest.importorskip("tokenspeed")
    from smg_grpc_servicer.tokenspeed.servicer import TokenSpeedSchedulerServicer

    servicer = TokenSpeedSchedulerServicer.__new__(TokenSpeedSchedulerServicer)
    # Two running + one finished-but-not-cleaned entry: waiting can't be told
    # apart from running without a scheduler round-trip, so it's reported as 0
    # and total mirrors running (excludes the finished entry).
    servicer.async_llm = SimpleNamespace(
        rid_to_state={
            "a": SimpleNamespace(finished=False),
            "b": SimpleNamespace(finished=False),
            "c": SimpleNamespace(finished=True),
        }
    )
    snap = servicer.load_snapshot()
    assert snap["num_running_reqs"] == 2.0
    assert snap["num_waiting_reqs"] == 0.0
    assert snap["num_total_reqs"] == 2.0
    assert snap["token_usage"] == 0.0


def test_sglang_get_server_info_advertises_metrics():
    pytest.importorskip("sglang")
    from smg_grpc_servicer.sglang.servicer import SGLangSchedulerServicer

    servicer = SGLangSchedulerServicer.__new__(SGLangSchedulerServicer)
    servicer.request_manager = SimpleNamespace(
        get_server_info=lambda: {
            "active_requests": 0,
            "paused": False,
            "last_receive_time": 0.0,
        }
    )
    # Wildcard host: only the port is advertised (gateway knows the real address).
    servicer.server_args = SimpleNamespace(host="0.0.0.0")
    servicer.scheduler_info = {"max_total_num_tokens": 0}
    servicer.metrics_port = 9102
    servicer.start_time = 0.0

    resp = asyncio.run(servicer.GetServerInfo(object(), object()))
    args = _struct_to_dict(resp.server_args)
    assert args["metrics_port"] == 9102
    assert "metrics_url" not in args
