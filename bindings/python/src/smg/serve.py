"""
Serve command: two-pass CLI argument parsing with lazy backend import.

Launches backend worker(s) + gateway router via a single `smg serve` command.
"""

import argparse
import atexit
import logging
import os
import random
import signal
import socket
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from smg.launch_router import launch_router
from smg.router_args import RouterArgs

logger = logging.getLogger("smg.serve")


# ---------------------------------------------------------------------------
# WorkerLauncher ABC + backend implementations
# ---------------------------------------------------------------------------


class WorkerLauncher(ABC):
    """Abstract base class for backend worker launchers."""

    @abstractmethod
    def build_command(
        self, args: argparse.Namespace, host: str, port: int
    ) -> List[str]:
        """Build the CLI command list to launch a worker."""
        ...

    @abstractmethod
    def health_check(self, host: str, port: int, timeout: float) -> bool:
        """Return True when the worker at host:port is healthy."""
        ...

    @abstractmethod
    def worker_url(self, host: str, port: int) -> str:
        """Return the URL used by the router to reach this worker."""
        ...

    def launch(
        self,
        args: argparse.Namespace,
        host: str,
        port: int,
        env: Optional[dict] = None,
    ) -> subprocess.Popen:
        """Launch the worker subprocess."""
        cmd = self.build_command(args, host, port)
        return subprocess.Popen(
            cmd, start_new_session=True, env=env or os.environ.copy()
        )


class SglangWorkerLauncher(WorkerLauncher):
    """Launcher for sglang inference workers."""

    def build_command(
        self, args: argparse.Namespace, host: str, port: int
    ) -> List[str]:
        cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            getattr(args, "model_path", ""),
            "--host",
            host,
            "--port",
            str(port),
        ]
        if getattr(args, "grpc_mode", False):
            cmd.append("--grpc-mode")
        return cmd

    def health_check(self, host: str, port: int, timeout: float) -> bool:
        return _http_health_check(f"http://{host}:{port}/health", timeout)

    def worker_url(self, host: str, port: int) -> str:
        return f"http://{host}:{port}"


class VllmWorkerLauncher(WorkerLauncher):
    """Launcher for vLLM inference workers (gRPC mode)."""

    def build_command(
        self, args: argparse.Namespace, host: str, port: int
    ) -> List[str]:
        return [
            sys.executable,
            "-m",
            "vllm.entrypoints.grpc_server",
            "--model",
            getattr(args, "model", ""),
            "--host",
            host,
            "--port",
            str(port),
        ]

    def health_check(self, host: str, port: int, timeout: float) -> bool:
        return _grpc_health_check(host, port, timeout)

    def worker_url(self, host: str, port: int) -> str:
        return f"grpc://{host}:{port}"


class TrtllmWorkerLauncher(WorkerLauncher):
    """Launcher for TensorRT-LLM inference workers (gRPC mode).

    Uses ``python3 -m tensorrt_llm.commands.serve <model> --grpc ...``.
    See https://github.com/NVIDIA/TensorRT-LLM/pull/11037
    """

    def build_command(
        self, args: argparse.Namespace, host: str, port: int
    ) -> List[str]:
        return [
            sys.executable,
            "-m",
            "tensorrt_llm.commands.serve",
            getattr(args, "model", ""),
            "--grpc",
            "--host",
            host,
            "--port",
            str(port),
        ]

    def health_check(self, host: str, port: int, timeout: float) -> bool:
        return _grpc_health_check(host, port, timeout)

    def worker_url(self, host: str, port: int) -> str:
        return f"grpc://{host}:{port}"


BACKEND_LAUNCHERS = {
    "sglang": SglangWorkerLauncher,
    "vllm": VllmWorkerLauncher,
    "trtllm": TrtllmWorkerLauncher,
}


# ---------------------------------------------------------------------------
# Health check utilities
# ---------------------------------------------------------------------------


def _http_health_check(url: str, timeout: float) -> bool:
    """GET the URL and return True on HTTP 200."""
    try:
        import urllib.request

        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception as e:
        logger.debug("HTTP health check for %s failed: %s", url, e)
        return False


def _grpc_health_check(host: str, port: int, timeout: float) -> bool:
    """Standard gRPC health check with fallback to channel_ready for vLLM."""
    try:
        import grpc
        from grpc_health.v1 import health_pb2, health_pb2_grpc
    except ImportError:
        logger.debug("gRPC libraries not available for health check")
        return False

    try:
        channel = grpc.insecure_channel(f"{host}:{port}")
        try:
            stub = health_pb2_grpc.HealthStub(channel)
            request = health_pb2.HealthCheckRequest(service="")
            response = stub.Check(request, timeout=timeout)
            return response.status == health_pb2.HealthCheckResponse.SERVING
        finally:
            channel.close()
    except grpc.RpcError as e:
        # vLLM doesn't implement gRPC health service — fall back to channel ready
        if hasattr(e, "code") and e.code() == grpc.StatusCode.UNIMPLEMENTED:
            try:
                channel = grpc.insecure_channel(f"{host}:{port}")
                try:
                    grpc.channel_ready_future(channel).result(timeout=timeout)
                    return True
                finally:
                    channel.close()
            except Exception as e:
                logger.debug("gRPC channel_ready fallback for %s:%d failed: %s", host, port, e)
                return False
        logger.debug("gRPC health check for %s:%d failed: %s", host, port, e)
        return False
    except Exception as e:
        logger.debug("gRPC health check error for %s:%d: %s", host, port, e)
        return False


# ---------------------------------------------------------------------------
# Port discovery
# ---------------------------------------------------------------------------


def _find_available_ports(base_port: int, count: int) -> List[int]:
    """Find *count* available ports starting near *base_port*.

    Uses socket bind test (no sglang dependency).  Ports are spaced with a
    small random offset to reduce collisions across concurrent launches.
    """
    ports: List[int] = []
    candidate = base_port
    while len(ports) < count:
        if _is_port_available(candidate):
            ports.append(candidate)
            candidate += random.randint(1, 5)
        else:
            candidate += 1
        if candidate > 65535:
            raise RuntimeError(
                f"Could not find {count} available ports starting from {base_port}"
            )
    return ports


def _is_port_available(port: int) -> bool:
    """Return True if *port* is free on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


# ---------------------------------------------------------------------------
# Argument adders (backend-specific CLI arguments)
# ---------------------------------------------------------------------------


def _add_sglang_args(parser: argparse.ArgumentParser) -> None:
    """Add sglang-specific arguments."""
    try:
        from sglang.srt.server_args import ServerArgs

        ServerArgs.add_cli_args(parser)
    except ImportError:
        parser.error("sglang is not installed. Install it with: pip install sglang")


def _add_vllm_args(parser: argparse.ArgumentParser) -> None:
    """Add vllm-specific arguments."""
    try:
        from vllm.engine.arg_utils import EngineArgs

        EngineArgs.add_cli_args(parser)
    except ImportError:
        parser.error("vllm is not installed. Install it with: pip install vllm")


def _add_trtllm_stub_args(parser: argparse.ArgumentParser) -> None:
    """Stub for TRT-LLM args until full integration."""
    group = parser.add_argument_group("TRT-LLM Options (stub)")
    group.add_argument("--model", type=str, help="Model path")


BACKEND_ARG_ADDERS = {
    "sglang": _add_sglang_args,
    "vllm": _add_vllm_args,
    "trtllm": _add_trtllm_stub_args,
}

BACKEND_CHOICES = list(BACKEND_ARG_ADDERS.keys())
DEFAULT_BACKEND = "sglang"


# ---------------------------------------------------------------------------
# Serve argument parsing (two-pass)
# ---------------------------------------------------------------------------


def add_serve_args(parser: argparse.ArgumentParser) -> None:
    """Add serve-specific arguments (not from any backend)."""
    group = parser.add_argument_group("Serve Options")
    group.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        choices=BACKEND_CHOICES,
        help=f"Inference backend to use (default: {DEFAULT_BACKEND})",
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
    BACKEND_ARG_ADDERS[backend](parser)


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
        "--backend", default=DEFAULT_BACKEND, choices=BACKEND_CHOICES
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


# ---------------------------------------------------------------------------
# ServeOrchestrator
# ---------------------------------------------------------------------------

_WORKER_SHUTDOWN_TIMEOUT = 30


class ServeOrchestrator:
    """Coordinate worker launch, health checking, router startup, and shutdown."""

    def __init__(self, backend: str, args: argparse.Namespace):
        self.backend = backend
        self.args = args
        self.launcher: WorkerLauncher = BACKEND_LAUNCHERS[backend]()
        self.workers: List[Tuple[subprocess.Popen, int]] = []
        self._shutting_down = False

    # -- public API ---------------------------------------------------------

    def run(self) -> None:
        """Full lifecycle: launch workers → health check → start router."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self._cleanup_workers)
        try:
            self._launch_workers()
            self._wait_healthy()
            router_args = self._build_router_args()
            launch_router(router_args)
        finally:
            self._cleanup_workers()

    # -- internal -----------------------------------------------------------

    def _launch_workers(self) -> None:
        ports = _find_available_ports(self.args.worker_base_port, self.args.dp_size)
        host = self.args.worker_host
        for port in ports:
            env = os.environ.copy()
            proc = self.launcher.launch(self.args, host, port, env)
            self.workers.append((proc, port))
            logger.info("Launched %s worker on %s:%d (pid %d)", self.backend, host, port, proc.pid)

    def _wait_healthy(self) -> None:
        host = self.args.worker_host
        for proc, port in self.workers:
            deadline = time.monotonic() + self.args.worker_startup_timeout
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"Worker on port {port} exited with code {proc.returncode}"
                    )
                if self.launcher.health_check(host, port, timeout=5.0):
                    logger.info("Worker on %s:%d is healthy", host, port)
                    break
                time.sleep(2)
            else:
                raise TimeoutError(
                    f"Worker on port {port} not healthy within "
                    f"{self.args.worker_startup_timeout}s"
                )

    def _build_router_args(self) -> RouterArgs:
        worker_urls = [
            self.launcher.worker_url(self.args.worker_host, port)
            for _, port in self.workers
        ]
        router_args = RouterArgs.from_cli_args(self.args, use_router_prefix=True)
        router_args.worker_urls = worker_urls
        return router_args

    def _signal_handler(self, signum: int, frame: object) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        logger.info("Received signal %d, shutting down workers…", signum)
        self._cleanup_workers()
        sys.exit(128 + signum)

    def _cleanup_workers(self) -> None:
        """SIGTERM all worker process groups, wait, then SIGKILL stragglers."""
        if not self.workers:
            return

        # Send SIGTERM to each process group
        for proc, port in self.workers:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass

        # Wait up to _WORKER_SHUTDOWN_TIMEOUT seconds for graceful exit
        deadline = time.monotonic() + _WORKER_SHUTDOWN_TIMEOUT
        for proc, port in self.workers:
            remaining = max(0, deadline - time.monotonic())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def serve_main(argv: Optional[List[str]] = None) -> None:
    """Parse serve args, create orchestrator, and run."""
    backend, args = parse_serve_args(argv)
    orchestrator = ServeOrchestrator(backend, args)
    orchestrator.run()
