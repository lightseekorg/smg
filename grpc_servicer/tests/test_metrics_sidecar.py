"""Tests for the shared Prometheus ``/metrics`` sidecar (engine-free).

``smg_grpc_servicer.metrics`` depends only on stdlib + prometheus_client, so
these run without any inference engine installed. They drive the async sidecar
via ``asyncio.run`` so no pytest-asyncio config is required.

Run with: pytest grpc_servicer/tests/test_metrics_sidecar.py
"""

from __future__ import annotations

import asyncio
import urllib.error
import urllib.request

import pytest

pytest.importorskip("prometheus_client")

from prometheus_client import CollectorRegistry  # noqa: E402
from prometheus_client.parser import text_string_to_metric_families  # noqa: E402
from smg_grpc_servicer import metrics as metrics_mod  # noqa: E402
from smg_grpc_servicer.metrics import (  # noqa: E402
    METRICS_PORT_ENV,
    SchedulerLoadCollector,
    metrics_server_args,
    metrics_url,
    resolve_metrics_port,
    start_metrics_sidecar,
)


def _get(url: str) -> tuple[int, str, str]:
    with urllib.request.urlopen(url, timeout=5) as resp:  # noqa: S310 — loopback only.
        return resp.status, resp.headers.get("Content-Type", ""), resp.read().decode()


async def _fetch(loop, url: str) -> tuple[int, str, str]:
    """Run the blocking urlopen off the event loop so the sidecar can serve it."""
    return await loop.run_in_executor(None, _get, url)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_metrics_url_skips_wildcard_hosts():
    assert metrics_url("0.0.0.0", 9100) is None
    assert metrics_url("::", 9100) is None
    assert metrics_url("", 9100) is None


def test_metrics_url_skips_loopback_hosts():
    # Loopback isn't reachable from a gateway in another network namespace, so we
    # advertise the port only and let the gateway pair it with the worker address.
    assert metrics_url("127.0.0.1", 9100) is None
    assert metrics_url("127.0.0.5", 9100) is None
    assert metrics_url("localhost", 9100) is None
    assert metrics_url("LocalHost", 9100) is None
    assert metrics_url("::1", 9100) is None
    assert metrics_url("[::1]", 9100) is None


def test_metrics_url_for_routable_host():
    assert metrics_url("10.1.2.3", 9100) == "http://10.1.2.3:9100/metrics"


def test_metrics_url_brackets_ipv6_literal():
    assert metrics_url("2001:db8::1", 9100) == "http://[2001:db8::1]:9100/metrics"


def test_metrics_url_does_not_double_bracket_ipv6():
    assert metrics_url("[2001:db8::1]", 9100) == "http://[2001:db8::1]:9100/metrics"


def test_metrics_server_args_disabled_is_empty():
    assert metrics_server_args("10.1.2.3", None) == {}


def test_metrics_server_args_routable_host_advertises_port_and_url():
    args = metrics_server_args("10.1.2.3", 9100)
    assert args == {"metrics_port": 9100, "metrics_url": "http://10.1.2.3:9100/metrics"}


def test_metrics_server_args_wildcard_host_advertises_port_only():
    # Gateway combines metrics_port with the worker address it discovered.
    assert metrics_server_args("0.0.0.0", 9100) == {"metrics_port": 9100}


def test_metrics_server_args_loopback_host_advertises_port_only():
    # Same as wildcard: a loopback bind yields no routable URL, only the port.
    assert metrics_server_args("127.0.0.1", 9100) == {"metrics_port": 9100}
    assert metrics_server_args("localhost", 9100) == {"metrics_port": 9100}


def test_resolve_metrics_port_prefers_explicit():
    assert resolve_metrics_port(9100) == 9100


@pytest.mark.parametrize("value", [0, 70000, -1])
def test_resolve_metrics_port_rejects_bad_explicit(monkeypatch, value):
    # Explicit port is range-validated like the env var (was previously bypassed).
    # Set the env to a *valid* port to prove the bad explicit value isn't silently
    # falling through to it.
    monkeypatch.setenv(METRICS_PORT_ENV, "9300")
    assert resolve_metrics_port(value) is None


def test_resolve_metrics_port_none_without_env(monkeypatch):
    monkeypatch.delenv(METRICS_PORT_ENV, raising=False)
    assert resolve_metrics_port(None) is None


def test_resolve_metrics_port_reads_env(monkeypatch):
    monkeypatch.setenv(METRICS_PORT_ENV, "9200")
    assert resolve_metrics_port(None) == 9200


@pytest.mark.parametrize("value", ["abc", "0", "70000", "-1"])
def test_resolve_metrics_port_rejects_bad_env(monkeypatch, value):
    monkeypatch.setenv(METRICS_PORT_ENV, value)
    assert resolve_metrics_port(None) is None


# ---------------------------------------------------------------------------
# SchedulerLoadCollector
# ---------------------------------------------------------------------------


def test_scheduler_load_collector_emits_gauges():
    snapshot = {
        "num_running_reqs": 3,
        "num_waiting_reqs": 2,
        "num_total_reqs": 5,
        "token_usage": 0.42,
    }
    registry = CollectorRegistry()
    registry.register(SchedulerLoadCollector(lambda: snapshot))

    assert registry.get_sample_value("smg_scheduler_running_requests") == 3.0
    assert registry.get_sample_value("smg_scheduler_waiting_requests") == 2.0
    assert registry.get_sample_value("smg_scheduler_total_requests") == 5.0
    assert registry.get_sample_value("smg_scheduler_token_usage") == 0.42


def test_scheduler_load_collector_tolerates_missing_keys():
    registry = CollectorRegistry()
    registry.register(SchedulerLoadCollector(dict))  # empty snapshot
    assert registry.get_sample_value("smg_scheduler_running_requests") == 0.0
    assert registry.get_sample_value("smg_scheduler_token_usage") == 0.0


def test_scheduler_load_collector_tolerates_raising_snapshot():
    def boom():
        raise RuntimeError("snapshot unavailable")

    registry = CollectorRegistry()
    registry.register(SchedulerLoadCollector(boom))
    # A failing snapshot_fn must not break the scrape: gauges fall back to 0.
    assert registry.get_sample_value("smg_scheduler_running_requests") == 0.0
    assert registry.get_sample_value("smg_scheduler_total_requests") == 0.0


# ---------------------------------------------------------------------------
# HTTP sidecar
# ---------------------------------------------------------------------------


def test_metrics_endpoint_returns_valid_exposition():
    async def scenario():
        registry = CollectorRegistry()
        registry.register(
            SchedulerLoadCollector(
                lambda: {
                    "num_running_reqs": 7,
                    "num_waiting_reqs": 1,
                    "num_total_reqs": 8,
                    "token_usage": 0.5,
                }
            )
        )
        sidecar = await start_metrics_sidecar("127.0.0.1", 0, registry=registry)
        assert sidecar is not None
        loop = asyncio.get_running_loop()
        try:
            status, content_type, body = await _fetch(
                loop, f"http://127.0.0.1:{sidecar.port}/metrics"
            )
        finally:
            await sidecar.close()
        return status, content_type, body

    status, content_type, body = asyncio.run(scenario())
    assert status == 200
    assert content_type.startswith("text/plain")
    families = {f.name: f for f in text_string_to_metric_families(body)}
    assert "smg_scheduler_running_requests" in families
    assert families["smg_scheduler_running_requests"].samples[0].value == 7.0


def test_root_path_also_serves_metrics():
    async def scenario():
        sidecar = await start_metrics_sidecar("127.0.0.1", 0, registry=CollectorRegistry())
        assert sidecar is not None
        loop = asyncio.get_running_loop()
        try:
            return await _fetch(loop, f"http://127.0.0.1:{sidecar.port}/")
        finally:
            await sidecar.close()

    status, _, body = asyncio.run(scenario())
    assert status == 200
    # Empty registry still yields valid (possibly empty) exposition.
    list(text_string_to_metric_families(body))


def test_unknown_path_returns_404():
    async def scenario():
        sidecar = await start_metrics_sidecar("127.0.0.1", 0, registry=CollectorRegistry())
        assert sidecar is not None
        loop = asyncio.get_running_loop()
        try:
            await _fetch(loop, f"http://127.0.0.1:{sidecar.port}/healthz")
        finally:
            await sidecar.close()

    with pytest.raises(urllib.error.HTTPError) as exc:
        asyncio.run(scenario())
    assert exc.value.code == 404


def test_query_string_is_ignored():
    async def scenario():
        sidecar = await start_metrics_sidecar("127.0.0.1", 0, registry=CollectorRegistry())
        assert sidecar is not None
        loop = asyncio.get_running_loop()
        try:
            return await _fetch(loop, f"http://127.0.0.1:{sidecar.port}/metrics?foo=bar")
        finally:
            await sidecar.close()

    status, _, _ = asyncio.run(scenario())
    assert status == 200


def test_start_is_best_effort_on_bind_failure():
    async def scenario():
        registry = CollectorRegistry()
        first = await start_metrics_sidecar("127.0.0.1", 0, registry=registry)
        assert first is not None
        try:
            # Second bind to the same port must fail-soft (return None).
            second = await start_metrics_sidecar("127.0.0.1", first.port, registry=registry)
            return second
        finally:
            await first.close()

    assert asyncio.run(scenario()) is None


def test_close_is_idempotent():
    async def scenario():
        sidecar = await start_metrics_sidecar("127.0.0.1", 0, registry=CollectorRegistry())
        assert sidecar is not None
        await sidecar.close()
        await sidecar.close()  # second close must not raise

    asyncio.run(scenario())


def test_advertises_actually_bound_port_not_zero():
    # Mirrors server.py: started on port 0, the servicer advertises the *bound*
    # port (sidecar.port), never the requested 0. The advertised host is the
    # configured server host, independent of the loopback bind used here.
    async def scenario():
        sidecar = await start_metrics_sidecar("127.0.0.1", 0, registry=CollectorRegistry())
        assert sidecar is not None
        try:
            return metrics_server_args("10.0.0.5", sidecar.port)
        finally:
            await sidecar.close()

    args = asyncio.run(scenario())
    assert args["metrics_port"] != 0
    assert args["metrics_url"] == f"http://10.0.0.5:{args['metrics_port']}/metrics"


def test_failed_bind_advertises_nothing():
    # When start returns None, server.py leaves metrics_port None → no advertisement.
    async def scenario():
        first = await start_metrics_sidecar("127.0.0.1", 0, registry=CollectorRegistry())
        assert first is not None
        try:
            second = await start_metrics_sidecar(
                "127.0.0.1", first.port, registry=CollectorRegistry()
            )
            advertised_port = second.port if second is not None else None
            return metrics_server_args("127.0.0.1", advertised_port)
        finally:
            await first.close()

    assert asyncio.run(scenario()) == {}


def test_handle_returns_on_immediate_disconnect():
    # EOF before any request line must not hang the handler; the server keeps
    # serving subsequent connections.
    async def scenario():
        sidecar = await start_metrics_sidecar("127.0.0.1", 0, registry=CollectorRegistry())
        assert sidecar is not None
        loop = asyncio.get_running_loop()
        try:
            # Open and immediately close without sending anything (EOF).
            reader, writer = await asyncio.open_connection("127.0.0.1", sidecar.port)
            writer.close()
            await writer.wait_closed()
            # A normal request still succeeds afterwards.
            return await _fetch(loop, f"http://127.0.0.1:{sidecar.port}/metrics")
        finally:
            await sidecar.close()

    status, _, _ = asyncio.run(scenario())
    assert status == 200


def test_handle_times_out_slow_request(monkeypatch):
    # A client that opens a connection but never finishes the request headers
    # must be dropped by the read timeout rather than pinning the handler open.
    monkeypatch.setattr(metrics_mod, "_READ_TIMEOUT", 0.2)

    async def scenario():
        sidecar = await start_metrics_sidecar("127.0.0.1", 0, registry=CollectorRegistry())
        assert sidecar is not None
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", sidecar.port)
            # Request line only — never send the blank line that ends headers.
            writer.write(b"GET /metrics HTTP/1.1\r\n")
            await writer.drain()
            # The server's read timeout (0.2s) closes the connection: read() then
            # returns EOF well within this wait_for, so the handler didn't hang.
            data = await asyncio.wait_for(reader.read(), timeout=3.0)
            writer.close()
            await writer.wait_closed()
            return data
        finally:
            await sidecar.close()

    assert asyncio.run(scenario()) == b""


def test_handle_times_out_slow_header_drip(monkeypatch):
    # Slowloris: a client that drips one header line per interval *shorter* than
    # the timeout but never finishes the block. A per-line timeout would reset the
    # clock on every drip and hold the handler open forever; the whole-head-phase
    # deadline must cut it off regardless of the drip rate.
    monkeypatch.setattr(metrics_mod, "_READ_TIMEOUT", 0.3)

    async def scenario():
        sidecar = await start_metrics_sidecar("127.0.0.1", 0, registry=CollectorRegistry())
        assert sidecar is not None
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", sidecar.port)
            writer.write(b"GET /metrics HTTP/1.1\r\n")
            await writer.drain()

            async def drip():
                # Each drip lands well inside _READ_TIMEOUT (0.3s) yet the total
                # span (>1s) blows the whole-phase budget; never sends the blank
                # line that would end the header block.
                for i in range(10):
                    await asyncio.sleep(0.15)
                    writer.write(f"X-Drip-{i}: keep-alive\r\n".encode())
                    await writer.drain()

            dripper = asyncio.ensure_future(drip())
            try:
                # The whole-phase deadline closes the connection: read() returns
                # EOF far sooner than the dripper would finish (10 * 0.15 = 1.5s).
                data = await asyncio.wait_for(reader.read(), timeout=1.0)
            finally:
                dripper.cancel()
                try:
                    await dripper
                except (asyncio.CancelledError, ConnectionError):
                    pass
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass
            return data
        finally:
            await sidecar.close()

    assert asyncio.run(scenario()) == b""
