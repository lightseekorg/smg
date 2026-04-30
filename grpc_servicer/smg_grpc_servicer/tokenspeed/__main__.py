"""CLI entrypoint for the TokenSpeed gRPC server.

Usage::

    python -m smg_grpc_servicer.tokenspeed --model-path <model> --host 127.0.0.1 --port 50051

All :class:`tokenspeed.runtime.utils.server_args.ServerArgs` flags are accepted
verbatim (we reuse TokenSpeed's own ``prepare_server_args`` so there is no
flag drift between the HTTP and gRPC frontends).
"""

from __future__ import annotations

import asyncio
import logging
import sys

import uvloop
from tokenspeed.runtime.utils.server_args import prepare_server_args

from smg_grpc_servicer.tokenspeed.server import serve_grpc


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    # TokenSpeed's ``ServerArgs.resolve_kernel_backends`` defaults
    # ``sampling_backend`` to ``"greedy"`` when the user doesn't pass
    # ``--sampling-backend``. The greedy backend is argmax-only and
    # ignores per-request ``temperature``/``top_p``/``top_k`` — fine for
    # the legacy CLI where users opt in to sampling explicitly, but
    # disastrous for a gateway-fronted gRPC servicer where per-request
    # sampling params arrive on every call. With Llama-3.2-1B the
    # always-argmax behavior collapses into single-token loops
    # (\\n×N, ' ('×N, "no"×N) within a few hundred steps and
    # generation runs to ``max_new_tokens`` — the smg e2e function-calling
    # suite makes this directly observable. Force a sampling-respecting
    # default unless the operator explicitly chose one.
    if not any(a == "--sampling-backend" or a.startswith("--sampling-backend=") for a in argv):
        argv = [*argv, "--sampling-backend", "flashinfer"]

    # TokenSpeed's logprob computation is gated by ``--enable-output-logprobs``
    # (default OFF, see ``ServerArgs.enable_output_logprobs``); without the
    # flag, requests asking for logprobs receive empty arrays rather than an
    # error. The smg gateway's OpenAI-compat path expects per-token logprobs
    # whenever ``logprobs=True`` is set, so flip the flag on by default for a
    # gateway-fronted gRPC servicer. Operators who want the smaller CUDA-graph
    # footprint can pass ``--enable-output-logprobs=False`` explicitly.
    # ``--enable-top-logprobs`` is intentionally NOT injected: TokenSpeed
    # raises at startup when it's set (the path is not yet implemented).
    if not any(
        a == "--enable-output-logprobs" or a.startswith("--enable-output-logprobs=") for a in argv
    ):
        argv = [*argv, "--enable-output-logprobs"]

    server_args = prepare_server_args(argv)
    # The scheduler processes will read these env vars; make sure we ran
    # through TokenSpeed's shared env/resource setup path instead of
    # duplicating it here.
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    asyncio.run(serve_grpc(server_args))


if __name__ == "__main__":
    main()
