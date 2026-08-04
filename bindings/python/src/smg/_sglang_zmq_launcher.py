"""Headless SGLang scheduler launcher for the SMG direct-ZMQ backend.

On the direct-ZMQ path SMG plays the SGLang TokenizerManager role itself: it
binds the PUSH input and PULL output ``ipc://`` sockets, sends tokenized
generate requests, and receives batched token outputs straight off the wire.
This module therefore launches *only* the scheduler process(es) — no
TokenizerManager, no DetokenizerManager — wired to SMG's two sockets through a
hand-built ``PortArgs``, with ``skip_tokenizer_init`` so the scheduler runs
tokenizer-free and routes its outputs back over the tokenizer socket (which SMG
owns).

Unmodified SGLang exposes no CLI flag or env var to point a bare scheduler at
externally-chosen sockets, so SMG drives SGLang's own ``PortArgs`` /
``Engine._launch_scheduler_processes`` primitives directly. Every argument other
than the two SMG-owned endpoints (``--smg-input-ipc`` / ``--smg-output-ipc``) is
a native SGLang server argument forwarded verbatim.

The scheduler defaults to pickle over ZMQ; SMG only decodes msgpack, so
``SGLANG_USE_PICKLE_IPC`` is forced off before SGLang is imported (the flag is
snapshotted at import time).

Invoked as a subprocess by the ``sglang`` launcher in ``serve.py``.
"""

import os

# Must precede any sglang import: io_struct snapshots this at module load to
# choose msgpack vs pickle for the ZMQ wire, and SMG only speaks msgpack. Hard
# override (not setdefault): an inherited "1" would silently pickle the wire.
os.environ["SGLANG_USE_PICKLE_IPC"] = "0"

import logging  # noqa: E402
import sys  # noqa: E402

logger = logging.getLogger("smg.sglang_zmq_launcher")

_INPUT_FLAG = "--smg-input-ipc"
_OUTPUT_FLAG = "--smg-output-ipc"


def _extract_flag(argv: list[str], flag: str) -> tuple[str, list[str]]:
    """Pop ``flag VALUE`` or ``flag=VALUE`` from argv; return (value, rest).

    These are SMG-owned endpoints, not SGLang args, so they are stripped before
    the remainder is handed to SGLang's own parser.
    """
    for i, token in enumerate(argv):
        if token == flag:
            if i + 1 >= len(argv):
                raise SystemExit(f"{flag} requires a value")
            return argv[i + 1], argv[:i] + argv[i + 2 :]
        if token.startswith(flag + "="):
            return token.split("=", 1)[1], argv[:i] + argv[i + 1 :]
    raise SystemExit(f"{flag} is required")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    # scheduler_input_ipc_name: SMG PUSHes requests here, the scheduler PULLs.
    input_ipc, argv = _extract_flag(argv, _INPUT_FLAG)
    # tokenizer_ipc_name: under skip_tokenizer_init the scheduler PUSHes outputs
    # here, and SMG PULLs them.
    output_ipc, argv = _extract_flag(argv, _OUTPUT_FLAG)

    from sglang.srt.entrypoints.engine import (
        Engine,
        _set_envs_and_config,
        _set_gc,
    )
    from sglang.srt.managers.scheduler import run_scheduler_process
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import PortArgs, prepare_server_args
    from sglang.srt.utils import configure_logger

    # SMG tokenizes upstream and drives the tokenizer<->scheduler ZMQ wire
    # directly, so the scheduler must run without its own tokenizer. Pass this as
    # a CLI flag: SGLang resolves and freezes server_args in prepare_server_args,
    # so it can no longer be set afterward.
    if "--skip-tokenizer-init" not in argv:
        argv.append("--skip-tokenizer-init")
    server_args = prepare_server_args(argv)

    # Mirror the head of Engine._launch_subprocesses so the scheduler subprocess
    # inherits the same logging, env/config, plugins, and mp start method it
    # would under a normal SGLang launch.
    configure_logger(server_args)
    _set_envs_and_config(server_args)
    load_plugins()
    server_args.check_server_args()
    _set_gc(server_args)

    # Reuse SGLang's own port derivation (nccl port, instance id, the unused
    # detokenizer/rpc/metrics ipc names), then repoint the two SMG-owned sockets
    # at the endpoints SMG binds.
    port_args = PortArgs.init_new(server_args)
    port_args.scheduler_input_ipc_name = input_ipc
    port_args.tokenizer_ipc_name = output_ipc

    result, procs = Engine._launch_scheduler_processes(
        server_args, port_args, run_scheduler_process
    )
    # Block until every scheduler rank has finished loading; the SMG router
    # gates request admission on its own ZMQ readiness, so once the schedulers
    # are up this process just keeps them alive.
    result.wait_for_ready()
    logger.info(
        "SGLang scheduler(s) ready on input=%s output=%s; SMG owns the tokenizer ZMQ wire",
        input_ipc,
        output_ipc,
    )
    result.block_until_scheduler_exits()

    # A scheduler exiting means the engine is gone; propagate a non-zero status
    # (rather than a silent exit 0) so the parent launcher sees the failure
    # instead of leaving the router pushing to a dead socket.
    if any(proc.exitcode for proc in procs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
