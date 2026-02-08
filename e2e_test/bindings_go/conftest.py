"""Pytest fixtures for Go bindings E2E tests.

Provides fixtures to build and run the Go OAI server, then test it
with the OpenAI client. The Go OAI server connects directly to a gRPC
worker from the model pool.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import time
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from infra import ModelInstance, ModelPool

logger = logging.getLogger(__name__)

# Paths
_ROOT = Path(__file__).resolve().parents[2]  # smg/
_GO_BINDINGS = _ROOT / "bindings" / "golang"
_GO_OAI_SERVER = _GO_BINDINGS / "examples" / "oai_server"


def _find_free_port() -> int:
    """Find an available port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_server(host: str, port: int, timeout: float = 30.0) -> bool:
    """Wait for server to become available."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except (TimeoutError, ConnectionRefusedError, OSError):
            time.sleep(0.5)
    return False


@pytest.fixture(scope="session")
def go_ffi_library() -> Path:
    """Build the Go FFI library and return its directory path."""
    lib_dir = _GO_BINDINGS / "target" / "release"

    # Check for existing library
    if (lib_dir / "libsmg_go.so").exists() or (lib_dir / "libsmg_go.dylib").exists():
        logger.info(f"Go FFI library found at: {lib_dir}")
        return lib_dir

    # Build the library
    logger.info("Building Go FFI library...")
    result = subprocess.run(
        ["make", "build"],
        cwd=_GO_BINDINGS,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"Failed to build Go FFI library: {result.stderr}")

    # Verify the library was built
    if not (lib_dir / "libsmg_go.so").exists() and not (lib_dir / "libsmg_go.dylib").exists():
        pytest.fail("Go FFI library not found after build")

    logger.info(f"Go FFI library built at: {lib_dir}")
    return lib_dir


@pytest.fixture(scope="session")
def go_oai_binary(go_ffi_library: Path) -> Path:
    """Build the Go OAI server binary and return its path."""
    binary_path = _GO_OAI_SERVER / "oai_server"

    # Set up environment for CGO
    env = os.environ.copy()
    env["CGO_LDFLAGS"] = f"-L{go_ffi_library}"
    env["LD_LIBRARY_PATH"] = f"{go_ffi_library}:{env.get('LD_LIBRARY_PATH', '')}"

    # Build the binary
    # Use -buildvcs=false to avoid VCS stamping issues in CI environments
    logger.info("Building Go OAI server...")
    result = subprocess.run(
        ["go", "build", "-buildvcs=false", "-o", "oai_server", "."],
        cwd=_GO_OAI_SERVER,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        pytest.fail(f"Failed to build Go OAI server: {result.stderr}")

    if not binary_path.exists():
        pytest.fail(f"Go OAI server binary not found at {binary_path}")

    logger.info(f"Go OAI server binary: {binary_path}")
    return binary_path


@pytest.fixture(scope="class")
def grpc_worker(request, model_pool: ModelPool) -> Generator[ModelInstance, None, None]:
    """Get a gRPC worker from the model pool.

    Uses the @pytest.mark.model marker to determine which model to use.
    """
    from fixtures.markers import get_marker_value
    from infra import DEFAULT_MODEL, ENV_MODEL, ConnectionMode

    # Get model from marker or env var or default
    model_id = get_marker_value(request, "model")
    if model_id is None:
        model_id = os.environ.get(ENV_MODEL, DEFAULT_MODEL)

    logger.info(f"Getting gRPC worker for model: {model_id}")

    try:
        # get() auto-acquires the returned instance
        instance = model_pool.get(model_id, ConnectionMode.GRPC)
    except (KeyError, RuntimeError) as e:
        pytest.fail(f"Failed to get gRPC worker for {model_id}: {e}")

    logger.info(f"Got gRPC worker at port {instance.port}")

    try:
        yield instance
    finally:
        instance.release()


@pytest.fixture(scope="class")
def grpc_workers(request, model_pool: ModelPool) -> Generator[list[ModelInstance], None, None]:
    """Get multiple gRPC workers from the model pool.

    Uses markers to determine configuration:
    - @pytest.mark.model("model-id"): Which model to use
    - @pytest.mark.workers(count=N): How many workers (default 1)
    """
    from fixtures.markers import get_marker_kwargs, get_marker_value
    from infra import DEFAULT_MODEL, ENV_MODEL, ConnectionMode, WorkerIdentity, WorkerType

    # Get model from marker or env var or default
    model_id = get_marker_value(request, "model")
    if model_id is None:
        model_id = os.environ.get(ENV_MODEL, DEFAULT_MODEL)

    # Get worker count from marker
    workers_config = get_marker_kwargs(request, "workers", defaults={"count": 1})
    num_workers = workers_config.get("count") or 1

    logger.info(f"Getting {num_workers} gRPC workers for model: {model_id}")

    instances: list = []

    try:
        if num_workers > 1:
            # Get existing workers of this mode
            all_existing = model_pool.get_workers_by_type(model_id, WorkerType.REGULAR)
            existing_for_mode = [w for w in all_existing if w.mode == ConnectionMode.GRPC]

            # Release workers with wrong mode
            for w in all_existing:
                if w not in existing_for_mode:
                    w.release()

            if len(existing_for_mode) >= num_workers:
                instances = existing_for_mode[:num_workers]
                # Release excess workers
                for w in existing_for_mode[num_workers:]:
                    w.release()
            else:
                # Need to launch more workers
                missing = num_workers - len(existing_for_mode)
                workers_to_launch = [
                    WorkerIdentity(
                        model_id,
                        ConnectionMode.GRPC,
                        WorkerType.REGULAR,
                        len(existing_for_mode) + i,
                    )
                    for i in range(missing)
                ]
                new_instances = model_pool.launch_workers(workers_to_launch, startup_timeout=300)
                # Acquire newly launched instances
                for inst in new_instances:
                    inst.acquire()
                instances = existing_for_mode + new_instances

            if not instances:
                pytest.fail(f"Failed to get {num_workers} gRPC workers for {model_id}")
            if len(instances) < num_workers:
                pytest.fail(
                    f"Expected {num_workers} gRPC workers but only got {len(instances)} for {model_id}. "
                    f"Available workers may be insufficient."
                )
        else:
            # Single worker - use simple get()
            instance = model_pool.get(model_id, ConnectionMode.GRPC)
            instances = [instance]

        logger.info(
            f"Got {len(instances)} gRPC workers at ports: {[inst.port for inst in instances]}"
        )
        assert len(instances) == num_workers, (
            f"Worker count mismatch: got {len(instances)}, expected {num_workers}"
        )

        yield instances

    except (KeyError, RuntimeError) as e:
        pytest.fail(f"Failed to get gRPC workers for {model_id}: {e}")

    finally:
        for inst in instances:
            inst.release()


@pytest.fixture(scope="class")
def go_oai_server(
    request,
    grpc_worker: ModelInstance,
    go_oai_binary: Path,
    go_ffi_library: Path,
) -> Generator[tuple[str, int, str], None, None]:
    """Start the Go OAI server connected to a single gRPC worker.

    Yields:
        Tuple of (host, port, model_path) for the Go OAI server.
    """
    # Get the gRPC endpoint from the worker
    grpc_endpoint = f"grpc://localhost:{grpc_worker.port}"

    # Find a free port for the Go OAI server
    oai_port = _find_free_port()

    # Set up environment - the Go OAI server uses env vars for config
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{go_ffi_library}:{env.get('LD_LIBRARY_PATH', '')}"
    env["DYLD_LIBRARY_PATH"] = f"{go_ffi_library}:{env.get('DYLD_LIBRARY_PATH', '')}"

    # Configuration via environment variables (server uses these, not CLI args)
    env["SGL_GRPC_ENDPOINT"] = grpc_endpoint
    env["SGL_TOKENIZER_PATH"] = grpc_worker.model_path  # model dir contains tokenizer
    env["PORT"] = str(oai_port)

    # Start the Go OAI server
    logger.info(
        f"Starting Go OAI server on port {oai_port}, connecting to gRPC worker at {grpc_endpoint}"
    )
    logger.info(f"Tokenizer path: {grpc_worker.model_path}")

    cmd = [str(go_oai_binary)]

    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # Wait for server to start
        if not _wait_for_server("localhost", oai_port, timeout=30.0):
            stdout, stderr = process.communicate(timeout=5)
            pytest.fail(
                f"Go OAI server failed to start.\n"
                f"Command: {' '.join(cmd)}\n"
                f"stdout: {stdout.decode()}\n"
                f"stderr: {stderr.decode()}"
            )

        logger.info(f"Go OAI server started on port {oai_port}")
        yield ("localhost", oai_port, grpc_worker.model_path)

    finally:
        # Shutdown the server
        logger.info("Shutting down Go OAI server...")
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


@pytest.fixture(scope="class")
def go_oai_server_multi(
    request,
    grpc_workers: list[ModelInstance],
    go_oai_binary: Path,
    go_ffi_library: Path,
) -> Generator[tuple[str, int, str], None, None]:
    """Start the Go OAI server connected to multiple gRPC workers with load balancing.

    Uses @pytest.mark.workers(count=N) to determine worker count.
    Uses @pytest.mark.gateway(policy="round_robin") to determine policy (default: round_robin).

    Yields:
        Tuple of (host, port, model_path) for the Go OAI server.
    """
    from fixtures.markers import get_marker_kwargs

    # Get policy from gateway marker
    gateway_config = get_marker_kwargs(request, "gateway", defaults={"policy": "round_robin"})
    policy_name = gateway_config.get("policy", "round_robin")

    # Build comma-separated endpoints
    grpc_endpoints = ",".join(f"grpc://localhost:{w.port}" for w in grpc_workers)

    # Find a free port for the Go OAI server
    oai_port = _find_free_port()

    # Set up environment
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{go_ffi_library}:{env.get('LD_LIBRARY_PATH', '')}"
    env["DYLD_LIBRARY_PATH"] = f"{go_ffi_library}:{env.get('DYLD_LIBRARY_PATH', '')}"

    # Configuration via environment variables
    # Use SGL_GRPC_ENDPOINTS (plural) for multi-worker support
    env["SGL_GRPC_ENDPOINTS"] = grpc_endpoints
    env["SGL_TOKENIZER_PATH"] = grpc_workers[0].model_path
    env["SGL_POLICY_NAME"] = policy_name
    env["PORT"] = str(oai_port)

    # Verify we got the expected number of workers
    workers_config = get_marker_kwargs(request, "workers", defaults={"count": 1})
    expected_workers = workers_config.get("count") or 1
    if len(grpc_workers) != expected_workers:
        pytest.fail(
            f"Expected {expected_workers} gRPC workers but got {len(grpc_workers)}. "
            f"Check that the model pool has enough resources."
        )

    logger.info(
        f"Starting Go OAI server on port {oai_port}, connecting to {len(grpc_workers)} gRPC workers "
        f"with policy={policy_name}"
    )
    logger.info(f"gRPC endpoints: {grpc_endpoints}")
    logger.info(f"Tokenizer path: {grpc_workers[0].model_path}")

    cmd = [str(go_oai_binary)]

    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # Wait for server to start
        if not _wait_for_server("localhost", oai_port, timeout=60.0):
            stdout, stderr = process.communicate(timeout=5)
            pytest.fail(
                f"Go OAI server failed to start.\n"
                f"Command: {' '.join(cmd)}\n"
                f"stdout: {stdout.decode()}\n"
                f"stderr: {stderr.decode()}"
            )

        logger.info(
            f"Go OAI server started on port {oai_port} with {len(grpc_workers)} workers "
            f"and policy={policy_name}"
        )
        yield ("localhost", oai_port, grpc_workers[0].model_path)

    finally:
        logger.info("Shutting down Go OAI server...")
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


@pytest.fixture(scope="class")
def go_openai_client(go_oai_server: tuple[str, int, str]):
    """Create an OpenAI client connected to the Go OAI server."""
    import openai

    host, port, _ = go_oai_server
    client = openai.OpenAI(
        base_url=f"http://{host}:{port}/v1",
        api_key="not-needed",
    )
    return client
