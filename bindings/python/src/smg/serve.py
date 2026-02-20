"""
Serve command: two-pass CLI argument parsing with lazy backend import.

Launches backend worker(s) + gateway router via a single `smg serve` command.
"""

from __future__ import annotations

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
        self, args: argparse.Namespace, backend_args: list[str], host: str, port: int
    ) -> list[str]:
        """Build the CLI command list to launch a worker."""
        ...

    def health_check(self, args: argparse.Namespace, host: str, port: int, timeout: float) -> bool:
        """Return True when the worker at host:port is healthy."""
        if getattr(args, "connection_mode", "grpc") == "grpc":
            return _grpc_health_check(host, port, timeout)
        return _http_health_check(f"http://{host}:{port}/health", timeout)

    def worker_url(self, args: argparse.Namespace, host: str, port: int) -> str:
        """Return the URL used by the router to reach this worker."""
        if getattr(args, "connection_mode", "grpc") == "grpc":
            return f"grpc://{host}:{port}"
        return f"http://{host}:{port}"

    def _get_tp_size(self, args: argparse.Namespace) -> int:
        """Return tensor-parallel size for GPU assignment. Default 1."""
        return 1

    def gpu_env(self, args: argparse.Namespace, dp_rank: int, env: dict | None = None) -> dict:
        """Build env dict with CUDA_VISIBLE_DEVICES for this worker's dp_rank."""
        env = dict(env) if env is not None else os.environ.copy()
        tp_size = self._get_tp_size(args)
        base_gpu = dp_rank * tp_size
        gpu_ids = range(base_gpu, base_gpu + tp_size)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
        env["PYTHONUNBUFFERED"] = "1"
        return env

    def _filter_backend_args(self, backend_args: list[str], filter_args: list[str]) -> list[str]:
        """Filter out args from backend_args that are already set by the launcher.

        Handles both ``--key value`` and ``--key=value`` syntax.
        """
        filtered = []
        skip_next = False
        for arg in backend_args:
            if skip_next:
                skip_next = False
                continue
            key = arg.split("=", 1)[0]
            if key in filter_args:
                if "=" not in arg:
                    skip_next = True  # value is the next token
                continue
            filtered.append(arg)
        return filtered

    def launch(
        self, args: argparse.Namespace, backend_args: list[str], host: str, port: int, env: dict
    ) -> subprocess.Popen:
        """Launch the worker subprocess."""
        cmd = self.build_command(args, backend_args, host, port)
        logger.info("Launching worker with command: %s", " ".join(cmd))

        return subprocess.Popen(
            cmd,
            start_new_session=True,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )


class SglangWorkerLauncher(WorkerLauncher):
    """Launcher for sglang inference workers."""

    def _get_tp_size(self, args: argparse.Namespace) -> int:
        return getattr(args, "tp_size", 1)

    def build_command(
        self, args: argparse.Namespace, backend_args: list[str], host: str, port: int
    ) -> list[str]:
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
        if getattr(args, "connection_mode", "grpc") == "grpc":
            cmd.append("--grpc-mode")

        cmd.extend(self._filter_backend_args(backend_args, ["--model-path", "--host", "--port"]))
        return cmd


class VllmWorkerLauncher(WorkerLauncher):
    """Launcher for vLLM inference workers."""

    def _get_tp_size(self, args: argparse.Namespace) -> int:
        return getattr(args, "tensor_parallel_size", 1)

    def build_command(
        self, args: argparse.Namespace, backend_args: list[str], host: str, port: int
    ) -> list[str]:
        vllm_entry_points = (
            "vllm.entrypoints.grpc_server"
            if getattr(args, "connection_mode", "grpc") == "grpc"
            else "vllm.entrypoints.openai.api_server"
        )

        cmd = [
            sys.executable,
            "-m",
            vllm_entry_points,
            "--model",
            getattr(args, "model", ""),
            "--host",
            host,
            "--port",
            str(port),
        ]
        cmd.extend(self._filter_backend_args(backend_args, ["--model", "--host", "--port"]))

        return cmd


class TrtllmWorkerLauncher(WorkerLauncher):
    """Launcher for TensorRT-LLM inference workers (gRPC mode only).

    Uses ``python3 -m tensorrt_llm.commands.serve <model> --grpc ...``.
    See https://github.com/NVIDIA/TensorRT-LLM/pull/11037
    """

    def _get_tp_size(self, args: argparse.Namespace) -> int:
        """Get tensor parallel size from args or config file.

        Priority: args.tp_size > args.tensor_parallel_size > config file > default(1)

        Raises:
            FileNotFoundError: If --config path does not exist.
            yaml.YAMLError: If --config file contains invalid YAML.
        """
        # Try --tp-size argument first
        tp_size = getattr(args, "tp_size", None)
        if tp_size is not None:
            return tp_size

        # Try --tensor-parallel-size (vLLM-style naming)
        tp_size = getattr(args, "tensor_parallel_size", None)
        if tp_size is not None:
            return tp_size

        # Try reading from config YAML file — let I/O and parse errors propagate
        # so misconfigurations are caught early instead of silently using tp=1.
        config_path = getattr(args, "config", None)
        if config_path:
            import yaml

            with open(config_path) as f:
                config = yaml.safe_load(f)
            if config and "tensor_parallel_size" in config:
                return int(config["tensor_parallel_size"])
            if config and "tp_size" in config:
                return int(config["tp_size"])
            logger.warning(
                "Config %s does not contain tensor_parallel_size or tp_size, defaulting to 1",
                config_path,
            )

        return 1

    def build_command(
        self, args: argparse.Namespace, backend_args: list[str], host: str, port: int
    ) -> list[str]:
        if getattr(args, "connection_mode", "grpc") != "grpc":
            raise ValueError("TensorRT-LLM backend only supports grpc connection mode")

        cmd = [
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

        # Add optional config file
        cmd.extend(self._filter_backend_args(backend_args, ["--model", "--host", "--port"]))

        return cmd


BACKEND_LAUNCHERS: dict[str, type[WorkerLauncher]] = {
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
            except Exception as fallback_err:
                logger.debug(
                    "gRPC channel_ready fallback for %s:%d failed: %s",
                    host,
                    port,
                    fallback_err,
                )
                return False
        logger.debug("gRPC health check for %s:%d failed: %s", host, port, e)
        return False
    except Exception as e:
        logger.debug("gRPC health check error for %s:%d: %s", host, port, e)
        return False


# ---------------------------------------------------------------------------
# Port discovery
# ---------------------------------------------------------------------------


def _find_available_ports(base_port: int, count: int) -> list[int]:
    """Find *count* available ports starting near *base_port*.

    Uses socket bind test (no sglang dependency).  Ports are spaced with a
    small random offset to reduce collisions across concurrent launches.
    """
    ports: list[int] = []
    candidate = base_port
    while len(ports) < count:
        if _is_port_available(candidate):
            ports.append(candidate)
            candidate += random.randint(1, 5)
        else:
            candidate += 1
        if candidate > 65535:
            raise RuntimeError(f"Could not find {count} available ports starting from {base_port}")
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
    """Add TensorRT-LLM specific arguments.

    Note: TensorRT-LLM doesn't provide a centralized argument manager like
    vLLM's EngineArgs. We manually add the most commonly used arguments.
    TP size is read from the config file, not passed as CLI argument.
    """
    group = parser.add_argument_group("TensorRT-LLM Options")
    group.add_argument("--model", type=str, help="Model path (HuggingFace ID or local path)")
    group.add_argument("--tp-size", type=int, help="Tensor parallel size (overrides config file)")
    group.add_argument(
        "--config",
        type=str,
        required=False,
        help="Config file path (YAML, optional - must contain tensor_parallel_size if provided)",
    )


BACKEND_ARG_ADDERS = {
    "sglang": _add_sglang_args,
    "vllm": _add_vllm_args,
    "trtllm": _add_trtllm_stub_args,
}

BACKEND_CHOICES = list(BACKEND_ARG_ADDERS.keys())
DEFAULT_BACKEND = os.getenv("SMG_DEFAULT_BACKEND", "sglang")


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
        "--connection-mode",
        default="grpc",
        choices=["grpc", "http"],
        help="Connection mode for workers (default: grpc). Note: trtllm only support grpc",
    )
    # Router host/port - may be overridden by backend (e.g. sglang)
    group.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for the router (default: 127.0.0.1)",
    )
    group.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the router (default: 8080)",
    )
    # Data parallel size - may be overridden by backend
    group.add_argument(
        "--data-parallel-size",
        "--dp-size",
        type=int,
        default=1,
        dest="data_parallel_size",
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
    argv: list[str] | None = None,
) -> tuple[str, argparse.Namespace, list[str]]:
    """Two-pass argument parsing for serve command.

    Pass 1: Parse with only serve + router args (parse_known_args). We get
    --backend and known flags; any remaining argv tokens are left as
    backend_args (no backend-specific parser loaded yet).
    Pass 2: Build full parser with serve + backend-specific + router args,
    then parse_args(argv) to get the full namespace. backend_args from pass 1
    are returned unchanged for the launcher to pass through to worker commands.

    Returns:
        Tuple of (backend_name, parsed_namespace, backend_args).
        backend_args: argv tokens not recognized in pass 1 (e.g. --config path).
    """
    if argv is None:
        argv = []

    # Pass 1: serve + router args only; unknown tokens become backend_args
    pre_parser = argparse.ArgumentParser(add_help=False)
    add_serve_args(pre_parser)
    RouterArgs.add_cli_args(pre_parser, use_router_prefix=True, exclude_host_port=True)
    serve_router_args, backend_args = pre_parser.parse_known_args(argv)
    backend = serve_router_args.backend

    # Pass 2: full parser with backend-specific args; resolve so backend can override
    parser = argparse.ArgumentParser(
        description=f"Launch {backend} worker(s) + gateway router",
        conflict_handler="resolve",
    )
    add_serve_args(parser)
    _import_backend_args(backend, parser)
    RouterArgs.add_cli_args(parser, use_router_prefix=True, exclude_host_port=True)

    args = parser.parse_args(argv)
    return backend, args, backend_args


# ---------------------------------------------------------------------------
# ServeOrchestrator
# ---------------------------------------------------------------------------

_WORKER_SHUTDOWN_TIMEOUT = 30


class ServeOrchestrator:
    """Coordinate worker launch, health checking, router startup, and shutdown."""

    def __init__(self, backend: str, args: argparse.Namespace, backend_args: list[str]):
        self.backend = backend
        self.args = args
        self.backend_args = backend_args
        self.launcher: WorkerLauncher = BACKEND_LAUNCHERS[backend]()
        self.workers: list[tuple[subprocess.Popen, int]] = []
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
        ports = _find_available_ports(self.args.worker_base_port, self.args.data_parallel_size)
        host = self.args.worker_host
        for dp_rank, port in enumerate(ports):
            env = self.launcher.gpu_env(self.args, dp_rank)
            proc = self.launcher.launch(self.args, self.backend_args, host, port, env)
            self.workers.append((proc, port))
            logger.info(
                "Launched %s worker on %s:%d (pid %d, GPUs: %s)",
                self.backend,
                host,
                port,
                proc.pid,
                env["CUDA_VISIBLE_DEVICES"],
            )

    def _wait_healthy(self) -> None:
        host = self.args.worker_host
        for proc, port in self.workers:
            deadline = time.monotonic() + self.args.worker_startup_timeout
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f"Worker on port {port} exited with code {proc.returncode}")
                if self.launcher.health_check(self.args, host, port, timeout=5.0):
                    logger.info("Worker on %s:%d is healthy", host, port)
                    break
                time.sleep(2)
            else:
                raise TimeoutError(
                    f"Worker on port {port} not healthy within {self.args.worker_startup_timeout}s"
                )

    def _build_router_args(self) -> RouterArgs:
        worker_urls = [
            self.launcher.worker_url(self.args, self.args.worker_host, port)
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


def serve_main(argv: list[str] | None = None) -> None:
    """Parse serve args, create orchestrator, and run."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    backend, args, backend_args = parse_serve_args(argv)
    orchestrator = ServeOrchestrator(backend, args, backend_args)
    orchestrator.run()
