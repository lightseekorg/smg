"""Simple worker lifecycle management. No pooling, no eviction, no sharing."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import IO, Any

from .constants import (
    DEFAULT_HOST,
    DEFAULT_STARTUP_TIMEOUT,
    ENV_SHOW_WORKER_LOGS,
    HEALTH_CHECK_INTERVAL,
    LAUNCH_STAGGER_DELAY,
    ConnectionMode,
    WorkerType,
    get_runtime,
)
from .model_specs import get_model_spec
from .process_utils import detect_ib_device, get_open_port, wait_for_health

logger = logging.getLogger(__name__)


@dataclass
class Worker:
    """A single inference worker process."""

    model_id: str
    engine: str  # "sglang", "vllm", or "trtllm"
    port: int
    gpu_ids: list[int]
    mode: ConnectionMode = ConnectionMode.HTTP
    worker_type: WorkerType = WorkerType.REGULAR
    bootstrap_port: int | None = None
    ib_device: str | None = None
    log_dir: str | None = None
    process: subprocess.Popen | None = field(default=None, repr=False)
    _log_file: IO[Any] | None = field(default=None, repr=False)

    @property
    def base_url(self) -> str:
        """Base URL for this worker."""
        if self.mode == ConnectionMode.GRPC:
            return f"grpc://{DEFAULT_HOST}:{self.port}"
        return f"http://{DEFAULT_HOST}:{self.port}"

    @property
    def worker_url(self) -> str:
        """Alias for base_url for Gateway compatibility."""
        return self.base_url

    @property
    def http_url(self) -> str:
        """HTTP URL (used for health checks even on gRPC workers)."""
        return f"http://{DEFAULT_HOST}:{self.port}"

    def start(self, timeout: int = DEFAULT_STARTUP_TIMEOUT) -> None:
        """Launch worker process and wait for health."""
        cmd = self._build_cmd()
        env = self._build_env()

        logger.info(
            "Starting %s %s worker for %s on GPUs %s port %d",
            self.engine,
            self.mode.value,
            self.model_id,
            self.gpu_ids,
            self.port,
        )
        logger.debug("Command: %s", " ".join(cmd))

        self.process = self._spawn_process(cmd, env)

        # Wait for health check
        if self.mode == ConnectionMode.GRPC:
            self._wait_grpc_healthy(timeout)
        else:
            wait_for_health(
                self.http_url,
                timeout=timeout,
                check_interval=HEALTH_CHECK_INTERVAL,
            )

        logger.info(
            "Worker %s healthy at %s (PID %d)",
            self.model_id,
            self.base_url,
            self.process.pid,
        )

    def stop(self) -> None:
        """Terminate worker process and all child processes."""
        if self.process is None or self.process.poll() is not None:
            return

        pid = self.process.pid
        logger.info("Stopping worker %s (PID %d)", self.model_id, pid)

        # Kill entire process group (workers run in their own session)
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            self.process.terminate()

        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                pgid = os.getpgid(pid)
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                self.process.kill()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.error("Worker PID %d did not die after SIGKILL", pid)

        # Clean up log file
        if self._log_file is not None:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

        # Release reserved ports
        from .process_utils import release_port

        release_port(self.port)
        if self.bootstrap_port is not None:
            release_port(self.bootstrap_port)

    def is_alive(self) -> bool:
        """Check if the worker process is still running."""
        return self.process is not None and self.process.poll() is None

    def _build_cmd(self) -> list[str]:
        """Build engine-specific launch command using model specs."""
        spec = get_model_spec(self.model_id)
        model_path = spec["model"]
        tp_size = spec.get("tp", 1)
        features = spec.get("features", [])

        if self.engine == "sglang":
            return self._build_sglang_cmd(model_path, tp_size, features, spec)
        elif self.engine == "vllm":
            if self.mode == ConnectionMode.GRPC:
                return self._build_vllm_grpc_cmd(model_path, tp_size, spec)
            else:
                return self._build_vllm_http_cmd(model_path, tp_size, spec)
        elif self.engine == "trtllm":
            return self._build_trtllm_cmd(model_path, tp_size, spec)
        else:
            raise ValueError(f"Unsupported engine: {self.engine}")

    def _build_sglang_cmd(
        self, model_path: str, tp_size: int, features: list[str], spec: dict
    ) -> list[str]:
        """Build SGLang launch command."""
        cmd = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model_path,
            "--host",
            DEFAULT_HOST,
            "--port",
            str(self.port),
            "--tp-size",
            str(tp_size),
            "--log-level",
            "warning",
        ]

        if self.mode == ConnectionMode.GRPC:
            cmd.append("--grpc-mode")

        if "embedding" in features:
            cmd.append("--is-embedding")

        # PD disaggregation arguments
        if self.worker_type == WorkerType.PREFILL:
            cmd.extend(["--disaggregation-mode", "prefill"])
            if self.bootstrap_port:
                cmd.extend(["--disaggregation-bootstrap-port", str(self.bootstrap_port)])
            if self.ib_device:
                cmd.extend(["--disaggregation-ib-device", self.ib_device])
        elif self.worker_type == WorkerType.DECODE:
            cmd.extend(["--disaggregation-mode", "decode"])
            cmd.extend(["--base-gpu-id", "0"])
            if self.ib_device:
                cmd.extend(["--disaggregation-ib-device", self.ib_device])

        # Additional worker args from model spec (e.g., --context-length)
        worker_args = spec.get("worker_args", [])
        if worker_args:
            cmd.extend(worker_args)

        return cmd

    def _build_vllm_grpc_cmd(self, model_path: str, tp_size: int, spec: dict) -> list[str]:
        """Build vLLM gRPC server command."""
        return self._build_vllm_base_cmd("vllm.entrypoints.grpc_server", model_path, tp_size, spec)

    def _build_vllm_http_cmd(self, model_path: str, tp_size: int, spec: dict) -> list[str]:
        """Build vLLM HTTP (OpenAI-compatible) server command."""
        return self._build_vllm_base_cmd(
            "vllm.entrypoints.openai.api_server", model_path, tp_size, spec
        )

    def _build_vllm_base_cmd(
        self, entrypoint: str, model_path: str, tp_size: int, spec: dict
    ) -> list[str]:
        """Build shared vLLM command base."""
        cmd = [
            "python3",
            "-m",
            entrypoint,
            "--model",
            model_path,
            "--host",
            DEFAULT_HOST,
            "--port",
            str(self.port),
            "--tensor-parallel-size",
            str(tp_size),
            "--max-model-len",
            "16384",
            "--gpu-memory-utilization",
            "0.9",
        ]
        extra = spec.get("vllm_args", [])
        if extra:
            cmd.extend(extra)
        return cmd

    def _build_trtllm_cmd(self, model_path: str, tp_size: int, spec: dict) -> list[str]:
        """Build TensorRT-LLM gRPC server command."""
        # Create config file to enable xgrammar guided decoding
        config_path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, prefix="trtllm_"
        )
        config_path.write("guided_decoding_backend: xgrammar\n")
        config_path.close()

        cmd = [
            "python3",
            "-m",
            "tensorrt_llm.commands.serve",
            "serve",
            model_path,
            "--grpc",
            "--host",
            DEFAULT_HOST,
            "--port",
            str(self.port),
            "--backend",
            "pytorch",
            "--tp_size",
            str(tp_size),
            "--extra_llm_api_options",
            config_path.name,
        ]
        extra = spec.get("trtllm_args", [])
        if extra:
            cmd.extend(extra)
        return cmd

    def _build_env(self) -> dict[str, str]:
        """Build environment variables for the worker process."""
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids))

        # TRT-LLM multi-GPU needs NCCL tuning for CI compatibility
        if self.engine == "trtllm" and len(self.gpu_ids) > 1:
            env["NCCL_DEBUG"] = "WARN"
            env["NCCL_IB_DISABLE"] = "1"
            env["NCCL_SHM_DISABLE"] = "1"
            env["TLLM_DISABLE_ALLREDUCE_AUTOTUNE"] = "1"

        return env

    def _spawn_process(self, cmd: list[str], env: dict[str, str]) -> subprocess.Popen:
        """Spawn the worker subprocess with output routing."""
        show_output = os.environ.get(ENV_SHOW_WORKER_LOGS, "0") == "1"

        stdout_target: int | IO[Any] | None = None
        stderr_target: int | IO[Any] | None = None

        if not show_output:
            if self.log_dir:
                os.makedirs(self.log_dir, exist_ok=True)
                safe_name = f"{self.model_id}_{self.engine}_{self.mode.value}_{self.port}".replace(
                    "/", "__"
                ).replace(":", "_")
                log_path = os.path.join(self.log_dir, f"worker-{safe_name}.log")
                self._log_file = open(log_path, "w", encoding="utf-8")
                stdout_target = self._log_file
                stderr_target = subprocess.STDOUT
            else:
                stdout_target = subprocess.DEVNULL
                stderr_target = subprocess.DEVNULL

        try:
            return subprocess.Popen(
                cmd,
                env=env,
                stdout=stdout_target,
                stderr=stderr_target,
                start_new_session=True,
            )
        except Exception:
            if self._log_file is not None:
                self._log_file.close()
                self._log_file = None
            raise

    def _wait_grpc_healthy(self, timeout: float) -> None:
        """Wait for a gRPC worker to become healthy."""
        try:
            import grpc
            from grpc_health.v1 import health_pb2, health_pb2_grpc
        except ImportError:
            raise RuntimeError(
                "gRPC health check requires grpcio and grpcio-health-checking packages"
            )

        start = time.perf_counter()
        channel = grpc.insecure_channel(f"{DEFAULT_HOST}:{self.port}")
        try:
            while time.perf_counter() - start < timeout:
                if not self.is_alive():
                    pid = self.process.pid if self.process else "unknown"
                    raise RuntimeError(f"Worker {self.model_id} (PID {pid}) died during startup")
                try:
                    stub = health_pb2_grpc.HealthStub(channel)
                    request = health_pb2.HealthCheckRequest(service="")
                    response = stub.Check(request, timeout=5.0)
                    if response.status == health_pb2.HealthCheckResponse.SERVING:
                        return
                except grpc.RpcError as e:
                    # UNIMPLEMENTED means server is up but doesn't have health service;
                    # fall back to channel connectivity check
                    if hasattr(e, "code") and e.code() == grpc.StatusCode.UNIMPLEMENTED:
                        try:
                            grpc.channel_ready_future(channel).result(timeout=5.0)
                            return
                        except Exception:
                            pass
                except Exception:
                    pass

                time.sleep(HEALTH_CHECK_INTERVAL)
        finally:
            channel.close()

        raise TimeoutError(
            f"gRPC worker {self.model_id} on port {self.port} "
            f"did not become healthy within {timeout}s"
        )


def start_workers(
    model_id: str,
    engine: str | None = None,
    mode: ConnectionMode = ConnectionMode.HTTP,
    count: int = 1,
    worker_type: WorkerType = WorkerType.REGULAR,
    timeout: int = DEFAULT_STARTUP_TIMEOUT,
    log_dir: str | None = None,
    gpu_offset: int = 0,
) -> list[Worker]:
    """Start N workers for a model. GPU IDs assigned sequentially.

    Args:
        model_id: Model identifier from MODEL_SPECS.
        engine: Runtime engine ("sglang", "vllm", "trtllm").
                If None, auto-detected from E2E_RUNTIME env var.
        mode: Connection mode (HTTP or GRPC).
        count: Number of workers to start.
        worker_type: Worker specialization (regular, prefill, decode).
        timeout: Seconds to wait for each worker to become healthy.
        log_dir: Directory to store worker log files.
        gpu_offset: Starting GPU index for worker assignment.

    Returns:
        List of started Worker instances.
    """
    if engine is None:
        engine = get_runtime()

    spec = get_model_spec(model_id)
    tp = spec.get("tp", 1)

    # Detect IB device for PD workers
    has_pd = worker_type in (WorkerType.PREFILL, WorkerType.DECODE)
    ib_device = detect_ib_device() if has_pd else None

    workers: list[Worker] = []

    try:
        for i in range(count):
            gpu_ids = list(range(gpu_offset, gpu_offset + tp))
            gpu_offset += tp
            port = get_open_port()
            bootstrap_port = get_open_port() if worker_type == WorkerType.PREFILL else None

            worker = Worker(
                model_id=model_id,
                engine=engine,
                port=port,
                gpu_ids=gpu_ids,
                mode=mode,
                worker_type=worker_type,
                bootstrap_port=bootstrap_port,
                ib_device=ib_device if has_pd else None,
                log_dir=log_dir,
            )

            # Stagger launches to avoid resource contention
            if i > 0 and LAUNCH_STAGGER_DELAY > 0:
                logger.info("Staggering launch by %ds", LAUNCH_STAGGER_DELAY)
                time.sleep(LAUNCH_STAGGER_DELAY)

            worker.start(timeout=timeout)
            workers.append(worker)
    except Exception:
        stop_workers(workers)
        raise

    return workers


def stop_workers(workers: list[Worker]) -> None:
    """Stop all workers in a list.

    Args:
        workers: List of Worker instances to stop.
    """
    for worker in workers:
        try:
            worker.stop()
        except Exception as e:
            logger.warning("Error stopping worker %s: %s", worker.model_id, e)
