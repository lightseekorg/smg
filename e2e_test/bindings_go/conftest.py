"""Pytest fixtures for Go bindings E2E tests.

Provides fixtures to build and run the Go OAI server, then test it
with the OpenAI client. The Go OAI server connects directly to a gRPC
worker launched via start_workers().
"""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Generator
from pathlib import Path

import pytest
from infra import ConnectionMode, get_open_port, get_runtime, release_port, terminate_process
from infra.model_specs import get_model_spec
from infra.process_utils import wait_for_health
from infra.worker import Worker, start_workers, stop_workers

logger = logging.getLogger(__name__)

# Paths
_ROOT = Path(__file__).resolve().parents[2]  # smg/
_GO_BINDINGS = _ROOT / "bindings" / "golang"
_GO_OAI_SERVER = _GO_BINDINGS / "examples" / "oai_server"


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
def grpc_worker(request) -> Generator[Worker, None, None]:
    """Launch a single gRPC worker.

    Uses the @pytest.mark.model marker to determine which model to use.
    """
    from fixtures.markers import get_marker_value
    from infra import DEFAULT_MODEL, ENV_MODEL

    # Get model from marker or env var or default
    model_id = get_marker_value(request, "model")
    if model_id is None:
        model_id = os.environ.get(ENV_MODEL, DEFAULT_MODEL)

    engine = get_runtime()
    logger.info(f"Starting gRPC worker for model: {model_id} (engine={engine})")

    try:
        workers = start_workers(model_id, engine=engine, mode=ConnectionMode.GRPC, count=1)
    except (KeyError, RuntimeError) as e:
        pytest.fail(f"Failed to start gRPC worker for {model_id}: {e}")

    worker = workers[0]
    logger.info(f"Started gRPC worker at port {worker.port}")

    yield worker

    stop_workers(workers)


@pytest.fixture(scope="class")
def grpc_workers(request) -> Generator[list[Worker], None, None]:
    """Launch multiple gRPC workers.

    Uses markers to determine configuration:
    - @pytest.mark.model("model-id"): Which model to use
    - @pytest.mark.workers(count=N): How many workers (default 1)
    """
    from fixtures.markers import get_marker_kwargs, get_marker_value
    from infra import DEFAULT_MODEL, ENV_MODEL

    # Get model from marker or env var or default
    model_id = get_marker_value(request, "model")
    if model_id is None:
        model_id = os.environ.get(ENV_MODEL, DEFAULT_MODEL)

    # Get worker count from marker
    workers_config = get_marker_kwargs(request, "workers", defaults={"count": 1})
    num_workers = workers_config.get("count") or 1

    engine = get_runtime()
    logger.info(f"Starting {num_workers} gRPC workers for model: {model_id} (engine={engine})")

    try:
        workers = start_workers(
            model_id,
            engine=engine,
            mode=ConnectionMode.GRPC,
            count=num_workers,
        )
    except (KeyError, RuntimeError) as e:
        pytest.fail(f"Failed to start gRPC workers for {model_id}: {e}")

    logger.info(f"Started {len(workers)} gRPC workers at ports: {[w.port for w in workers]}")
    assert len(workers) == num_workers, (
        f"Worker count mismatch: got {len(workers)}, expected {num_workers}"
    )

    yield workers

    stop_workers(workers)


@pytest.fixture(scope="class")
def go_oai_server(
    request,
    grpc_worker: Worker,
    go_oai_binary: Path,
    go_ffi_library: Path,
) -> Generator[tuple[str, int, str], None, None]:
    """Start the Go OAI server connected to a single gRPC worker.

    Yields:
        Tuple of (host, port, model_path) for the Go OAI server.
    """
    # Get the gRPC endpoint from the worker
    grpc_endpoint = f"grpc://localhost:{grpc_worker.port}"
    model_path = get_model_spec(grpc_worker.model_id)["model"]

    # Find a free port for the Go OAI server
    oai_port = get_open_port()

    # Set up environment - the Go OAI server uses env vars for config
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{go_ffi_library}:{env.get('LD_LIBRARY_PATH', '')}"
    env["DYLD_LIBRARY_PATH"] = f"{go_ffi_library}:{env.get('DYLD_LIBRARY_PATH', '')}"

    # Configuration via environment variables (server uses these, not CLI args)
    env["SGL_GRPC_ENDPOINT"] = grpc_endpoint
    env["SGL_TOKENIZER_PATH"] = model_path  # model dir contains tokenizer
    env["PORT"] = str(oai_port)

    # Start the Go OAI server
    logger.info(
        f"Starting Go OAI server on port {oai_port}, connecting to gRPC worker at {grpc_endpoint}"
    )
    logger.info(f"Tokenizer path: {model_path}")

    cmd = [str(go_oai_binary)]

    try:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for server to start
        try:
            wait_for_health(f"http://localhost:{oai_port}", timeout=30.0, check_interval=0.5)
        except TimeoutError:
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                terminate_process(process, timeout=10)
                pytest.fail(
                    f"Go OAI server failed to start and did not exit cleanly.\n"
                    f"Command: {' '.join(cmd)}"
                )
            pytest.fail(
                f"Go OAI server failed to start.\n"
                f"Command: {' '.join(cmd)}\n"
                f"stdout: {stdout.decode()}\n"
                f"stderr: {stderr.decode()}"
            )

        logger.info(f"Go OAI server started on port {oai_port}")
        yield ("localhost", oai_port, model_path)

    finally:
        logger.info("Shutting down Go OAI server...")
        terminate_process(process, timeout=10)
        release_port(oai_port)


@pytest.fixture(scope="class")
def go_oai_server_multi(
    request,
    grpc_workers: list[Worker],
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
    model_path = get_model_spec(grpc_workers[0].model_id)["model"]

    # Find a free port for the Go OAI server
    oai_port = get_open_port()

    # Set up environment
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{go_ffi_library}:{env.get('LD_LIBRARY_PATH', '')}"
    env["DYLD_LIBRARY_PATH"] = f"{go_ffi_library}:{env.get('DYLD_LIBRARY_PATH', '')}"

    # Configuration via environment variables
    # Use SGL_GRPC_ENDPOINTS (plural) for multi-worker support
    env["SGL_GRPC_ENDPOINTS"] = grpc_endpoints
    env["SGL_TOKENIZER_PATH"] = model_path
    env["SGL_POLICY_NAME"] = policy_name
    env["PORT"] = str(oai_port)

    # Verify we got the expected number of workers
    workers_config = get_marker_kwargs(request, "workers", defaults={"count": 1})
    expected_workers = workers_config.get("count") or 1
    if len(grpc_workers) != expected_workers:
        pytest.fail(
            f"Expected {expected_workers} gRPC workers but got {len(grpc_workers)}. "
            f"Check that enough GPUs are available."
        )

    logger.info(
        f"Starting Go OAI server on port {oai_port}, connecting to {len(grpc_workers)} gRPC workers "
        f"with policy={policy_name}"
    )
    logger.info(f"gRPC endpoints: {grpc_endpoints}")
    logger.info(f"Tokenizer path: {model_path}")

    cmd = [str(go_oai_binary)]

    try:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for server to start
        try:
            wait_for_health(f"http://localhost:{oai_port}", timeout=60.0, check_interval=0.5)
        except TimeoutError:
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                terminate_process(process, timeout=10)
                pytest.fail(
                    f"Go OAI server failed to start and did not exit cleanly.\n"
                    f"Command: {' '.join(cmd)}"
                )
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
        yield ("localhost", oai_port, model_path)

    finally:
        logger.info("Shutting down Go OAI server...")
        terminate_process(process, timeout=10)
        release_port(oai_port)


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
