"""CLI entrypoint for the TokenSpeed gRPC server.

Usage::

    python -m smg_grpc_servicer.tokenspeed --model <model> --host 127.0.0.1 --port 50051

All :class:`ServerArgs` flags are accepted — argv is parsed by
``prepare_server_args`` so there is no flag drift vs the HTTP frontend.
``--metrics-port`` (handled here, not by ``ServerArgs``) starts the Prometheus
``/metrics`` sidecar.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from tokenspeed.runtime.utils.server_args import prepare_server_args

from smg_grpc_servicer.tokenspeed.server import serve_grpc

try:
    import uvloop
except ImportError:  # uvloop is optional — fall back to the default loop.
    uvloop = None


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    # Pull --metrics-port out before handing the rest to ServerArgs, which
    # doesn't know the flag; SMG_METRICS_PORT still applies when it's omitted.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--metrics-port", type=int, default=None)
    known, rest = parser.parse_known_args(argv)

    server_args = prepare_server_args(rest)
    if uvloop is not None:
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    asyncio.run(serve_grpc(server_args, metrics_port=known.metrics_port))


if __name__ == "__main__":
    main()
