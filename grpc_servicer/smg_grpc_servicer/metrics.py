"""Shared best-effort Prometheus ``/metrics`` HTTP sidecar for the servicers.

The gRPC servicers (sglang, tokenspeed) start :func:`start_metrics_sidecar`
alongside their ``serve_grpc`` so the Rust gateway can scrape per-worker
Prometheus metrics over plain HTTP. The sidecar is intentionally dependency-light
(stdlib ``asyncio`` + ``prometheus_client`` only — no aiohttp/uvicorn) and
*best-effort*: any bind/serve failure is logged and swallowed so the gRPC server
keeps serving.

Metrics come from each engine's native ``prometheus_client`` registry. For
engines without one (tokenspeed), :class:`SchedulerLoadCollector` re-exposes the
same scheduler load snapshot that ``GetLoads`` already computes.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
from collections.abc import Callable

from prometheus_client import CollectorRegistry
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.exposition import CONTENT_TYPE_LATEST, generate_latest
from prometheus_client.registry import Collector

logger = logging.getLogger(__name__)

# Env fallback for the sidecar port when a servicer's launcher can't take a
# ``--metrics-port`` flag (e.g. sglang's entrypoint comes from upstream).
METRICS_PORT_ENV = "SMG_METRICS_PORT"

# Wildcard bind hosts: the gateway reaches the worker only at its real address
# (which it discovered from registration), never at a wildcard.
_WILDCARD_HOSTS = frozenset({"0.0.0.0", "::", "[::]", ""})


def _is_unroutable_host(host: str) -> bool:
    """True for hosts the gateway can't reach across the network.

    Covers wildcard binds and loopback (``127.0.0.0/8``, ``::1``, ``localhost``):
    for these we advertise the numeric ``metrics_port`` only and let the gateway
    pair it with the worker address it discovered. Any other hostname is assumed
    routable.
    """
    if host in _WILDCARD_HOSTS or host.lower() == "localhost":
        return True
    candidate = host[1:-1] if host.startswith("[") and host.endswith("]") else host
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


# One budget for the whole request-head read (request line + headers) so a
# Slowloris-style client dripping headers can't pin a handler open.
_READ_TIMEOUT = 5.0


def _coerce_port(value: object, source: str) -> int | None:
    """Coerce ``value`` to a valid ``0 < port < 65536`` int, else log + return None."""
    try:
        port = int(value)
    except (TypeError, ValueError):
        logger.warning("%s=%r is not an int; metrics sidecar disabled", source, value)
        return None
    if not 0 < port < 65536:
        logger.warning("%s=%d out of range; metrics sidecar disabled", source, port)
        return None
    return port


def resolve_metrics_port(explicit: int | None = None) -> int | None:
    """Resolve the sidecar port from an explicit value or ``SMG_METRICS_PORT``.

    Returns ``None`` when neither is set (sidecar disabled) or the value is not a
    usable port. Both the explicit argument and the env var are range-validated;
    a malformed value is logged and treated as unset rather than aborting startup.
    """
    if explicit is not None:
        return _coerce_port(explicit, "metrics_port")
    raw = os.getenv(METRICS_PORT_ENV)
    if not raw:
        return None
    return _coerce_port(raw, METRICS_PORT_ENV)


def metrics_url(host: str, port: int) -> str | None:
    """Return an ``http://host:port/metrics`` URL, or ``None`` for unroutable binds.

    A servicer bound to a wildcard (``0.0.0.0``) or loopback (``127.0.0.1``,
    ``localhost``, ``::1``) address isn't reachable by the gateway across the
    network — the gateway reaches it at the worker address it discovered — so
    advertising such a URL would be misleading. The numeric ``metrics_port`` is
    advertised regardless; the gateway combines it with that worker address.
    """
    if _is_unroutable_host(host):
        return None
    # Bracket bare IPv6 literals so the URL parses (``http://[::1]:9100/...``).
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{port}/metrics"


def metrics_server_args(host: str, metrics_port: int | None) -> dict[str, object]:
    """Build the ``server_args`` keys advertising the sidecar to the gateway.

    Returns ``{"metrics_port": int}`` (always, when enabled) plus
    ``{"metrics_url": str}`` when ``host`` is routable. The Rust gateway picks
    these out of ``GetServerInfo.server_args`` via its key allowlist, so the key
    names here are a cross-component contract. Empty dict when disabled.
    """
    if metrics_port is None:
        return {}
    port = int(metrics_port)
    out: dict[str, object] = {"metrics_port": port}
    url = metrics_url(host or "", port)
    if url is not None:
        out["metrics_url"] = url
    return out


class SchedulerLoadCollector(Collector):
    """Re-expose a scheduler load snapshot as Prometheus gauges.

    ``snapshot_fn`` returns the same dict shape ``GetLoads`` builds its response
    from: ``num_running_reqs``, ``num_waiting_reqs``, ``num_total_reqs`` and a
    ``token_usage`` ratio in ``[0, 1]``. It is called on every scrape; a raising
    ``snapshot_fn`` is caught and exposed as an empty snapshot so one failing call
    doesn't break the whole ``/metrics`` response.
    """

    def __init__(self, snapshot_fn: Callable[[], dict[str, float]]):
        self._snapshot_fn = snapshot_fn

    def collect(self):
        try:
            snapshot = self._snapshot_fn() or {}
        except Exception:  # noqa: BLE001 — a failing snapshot must not break the scrape.
            logger.warning("scheduler load snapshot failed; exposing empty", exc_info=True)
            snapshot = {}
        gauges = (
            ("smg_scheduler_running_requests", "Requests currently running", "num_running_reqs"),
            ("smg_scheduler_waiting_requests", "Requests waiting in queue", "num_waiting_reqs"),
            ("smg_scheduler_total_requests", "Running + waiting requests", "num_total_reqs"),
            ("smg_scheduler_token_usage", "KV-cache token usage ratio [0,1]", "token_usage"),
        )
        for name, doc, key in gauges:
            metric = GaugeMetricFamily(name, doc)
            metric.add_metric([], float(snapshot.get(key, 0.0) or 0.0))
            yield metric


class MetricsSidecar:
    """A running ``/metrics`` HTTP server backed by ``asyncio``.

    Serves Prometheus exposition for ``GET /metrics`` (and ``/``); every other
    path gets ``404``. Kept minimal on purpose — it exists only so the gateway
    can scrape, not as a general HTTP frontend.
    """

    def __init__(self, registry: CollectorRegistry, host: str, port: int):
        self._registry = registry
        self.host = host
        self.port = port
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, self.host, self.port)
        # When started with port 0 the OS assigns an ephemeral port; surface it
        # so callers (and tests) can read the real bound port.
        if self.port == 0 and self._server.sockets:
            self.port = self._server.sockets[0].getsockname()[1]

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            try:
                await self._server.wait_closed()
            except Exception:  # noqa: BLE001 — shutdown is best-effort.
                logger.debug("metrics sidecar wait_closed raised", exc_info=True)
            self._server = None

    @staticmethod
    async def _read_head(reader: asyncio.StreamReader) -> bytes:
        """Read the request line + drain headers, returning the request line.

        Reads until the blank line that ends the header block (or EOF). Draining
        the headers keeps the client's write side from seeing a reset. The caller
        wraps this in a single ``wait_for`` so the *whole* head-read phase shares
        one deadline; a per-line timeout would let a Slowloris client reset the
        clock by dripping one header every interval and hold the handler open.
        """
        request_line = await reader.readline()
        # Empty request line means EOF / immediate disconnect — nothing to serve.
        if not request_line:
            return b""
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break
        return request_line

    async def _handle(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            # One deadline for the entire head read (request line + all headers),
            # not one per line — see _read_head.
            request_line = await asyncio.wait_for(self._read_head(reader), timeout=_READ_TIMEOUT)
            # Empty request line means EOF / immediate disconnect — nothing to serve.
            if not request_line:
                return

            parts = request_line.split()
            method = parts[0] if parts else b""
            path = parts[1].split(b"?", 1)[0] if len(parts) > 1 else b""

            if method != b"GET":
                await self._write_response(writer, 405, b"text/plain", b"method not allowed")
            elif path in (b"/metrics", b"/"):
                payload = generate_latest(self._registry)
                await self._write_response(writer, 200, CONTENT_TYPE_LATEST.encode(), payload)
            else:
                await self._write_response(writer, 404, b"text/plain", b"not found")
        except Exception:  # noqa: BLE001 — never let a scrape kill the loop.
            logger.debug("metrics sidecar request failed", exc_info=True)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:  # noqa: BLE001 — fully release the socket, best-effort.
                pass

    @staticmethod
    async def _write_response(
        writer: asyncio.StreamWriter,
        status: int,
        content_type: bytes,
        body: bytes,
    ) -> None:
        reason = {200: "OK", 404: "Not Found", 405: "Method Not Allowed"}.get(status, "OK")
        head = (
            f"HTTP/1.1 {status} {reason}\r\n".encode()
            + b"Content-Type: "
            + content_type
            + b"\r\n"
            + f"Content-Length: {len(body)}\r\n".encode()
            + b"Connection: close\r\n\r\n"
        )
        writer.write(head + body)
        await writer.drain()


async def start_metrics_sidecar(
    host: str,
    port: int,
    *,
    registry: CollectorRegistry | None = None,
) -> MetricsSidecar | None:
    """Best-effort: start the ``/metrics`` sidecar; return it, or ``None`` on failure.

    A bind/serve failure is logged at WARNING and swallowed — the caller's gRPC
    server must keep running regardless. ``registry`` defaults to the global
    ``prometheus_client`` registry, which is what the engines populate.
    """
    if registry is None:
        # Import lazily so the module loads even if prometheus_client's default
        # registry is unavailable for some reason.
        from prometheus_client import REGISTRY

        registry = REGISTRY

    sidecar = MetricsSidecar(registry, host, port)
    try:
        await sidecar.start()
    except Exception:  # noqa: BLE001 — sidecar is non-fatal.
        logger.warning(
            "Failed to start Prometheus /metrics sidecar on %s:%d; continuing without it",
            host,
            port,
            exc_info=True,
        )
        return None

    logger.info("Prometheus /metrics sidecar listening on %s:%d", host, sidecar.port)
    return sidecar
