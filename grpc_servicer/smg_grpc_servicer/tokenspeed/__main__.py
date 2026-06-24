"""CLI entrypoint for the TokenSpeed gRPC server.

Usage::

    python -m smg_grpc_servicer.tokenspeed --model <model> --host 127.0.0.1 --port 50051

All :class:`ServerArgs` flags are accepted — argv is parsed by
``prepare_server_args`` so there is no flag drift vs the HTTP frontend.
"""

from __future__ import annotations

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

    server_args = prepare_server_args(argv)
    if uvloop is not None:
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    asyncio.run(serve_grpc(server_args))


if __name__ == "__main__":
    main()
