"""
Serve command: two-pass CLI argument parsing with lazy backend import.

Launches backend worker(s) + gateway router via a single `smg serve` command.
"""

import argparse
from typing import List, Optional, Tuple

from smg.router_args import RouterArgs


def add_serve_args(parser: argparse.ArgumentParser) -> None:
    """Add serve-specific arguments (not from any backend)."""
    group = parser.add_argument_group("Serve Options")
    group.add_argument(
        "--backend",
        default="sglang",
        choices=["sglang", "vllm", "trtllm"],
        help="Inference backend to use (default: sglang)",
    )
    group.add_argument(
        "--dp-size",
        type=int,
        default=1,
        help="Data parallel size (number of worker replicas)",
    )
    group.add_argument(
        "--worker-host",
        default="127.0.0.1",
        help="Host for worker processes (default: 127.0.0.1)",
    )
    group.add_argument(
        "--worker-base-port",
        type=int,
        default=31000,
        help="Base port for workers (default: 31000)",
    )
    group.add_argument(
        "--worker-startup-timeout",
        type=int,
        default=300,
        help="Seconds to wait for workers to become healthy (default: 300)",
    )


def _import_backend_args(backend: str, parser: argparse.ArgumentParser) -> None:
    """Conditionally import and add backend-native args to parser."""
    match backend:
        case "sglang":
            try:
                from sglang.srt.server_args import ServerArgs

                ServerArgs.add_cli_args(parser)
            except ImportError:
                parser.error(
                    "sglang is not installed. Install it with: pip install sglang"
                )
        case "vllm":
            try:
                from vllm.engine.arg_utils import EngineArgs

                EngineArgs.add_cli_args(parser)
            except ImportError:
                parser.error(
                    "vllm is not installed. Install it with: pip install vllm"
                )
        case "trtllm":
            _add_trtllm_stub_args(parser)


def _add_trtllm_stub_args(parser: argparse.ArgumentParser) -> None:
    """Stub for TRT-LLM args until full integration."""
    group = parser.add_argument_group("TRT-LLM Options (stub)")
    group.add_argument("--model", type=str, help="Model path")


def parse_serve_args(
    argv: Optional[List[str]] = None,
) -> Tuple[str, argparse.Namespace]:
    """Two-pass argument parsing for serve command.

    Pass 1: Extract --backend with parse_known_args (no backend imports).
    Pass 2: Build full parser with backend-specific + router args.

    Returns:
        Tuple of (backend_name, parsed_namespace).
    """
    if argv is None:
        argv = []

    # Pass 1: extract --backend (lightweight, no backend imports)
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--backend", default="sglang", choices=["sglang", "vllm", "trtllm"]
    )
    pre_args, _ = pre_parser.parse_known_args(argv)
    backend = pre_args.backend

    # Pass 2: build full parser with backend-specific args
    parser = argparse.ArgumentParser(
        description=f"Launch {backend} worker(s) + gateway router"
    )
    add_serve_args(parser)
    _import_backend_args(backend, parser)
    RouterArgs.add_cli_args(parser, use_router_prefix=True, exclude_host_port=True)

    args = parser.parse_args(argv)
    return backend, args
