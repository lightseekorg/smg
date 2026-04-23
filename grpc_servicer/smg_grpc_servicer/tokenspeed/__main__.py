"""CLI entrypoint for the TokenSpeed gRPC server.

Usage::

    python -m smg_grpc_servicer.tokenspeed --model-path <model> --host 127.0.0.1 --port 50051

All :class:`tokenspeed.runtime.server_args.ServerArgs` flags are accepted
verbatim (we reuse TokenSpeed's own ``prepare_server_args`` so there is no
flag drift between the HTTP and gRPC frontends).
"""

from __future__ import annotations

import asyncio
import logging
import sys

import uvloop
from tokenspeed.runtime.server_args import prepare_server_args

from smg_grpc_servicer.tokenspeed.server import serve_grpc


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    server_args = prepare_server_args(argv)
    # The scheduler processes will read these env vars; make sure we ran
    # through TokenSpeed's shared env/resource setup path instead of
    # duplicating it here.
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    asyncio.run(serve_grpc(server_args))


if __name__ == "__main__":
    main()
