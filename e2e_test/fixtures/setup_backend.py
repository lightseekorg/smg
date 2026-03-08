"""Backend setup fixtures for E2E tests.

Simplified backend lifecycle -- one set of workers and gateway per test class.
No model pool, no thread-local caching. Direct worker management via
start_workers/stop_workers.
"""

from __future__ import annotations

import logging
import os

import anthropic
import openai
import pytest
from infra import (
    DEFAULT_MODEL,
    DEFAULT_ROUTER_TIMEOUT,
    ENV_MODEL,
    ENV_SKIP_BACKEND_SETUP,
    RUNTIME_LABELS,
    THIRD_PARTY_MODELS,
    ConnectionMode,
    Gateway,
    WorkerType,
    get_runtime,
    launch_cloud_gateway,
)
from infra.model_specs import get_model_spec
from infra.worker import start_workers, stop_workers

from .markers import get_marker_kwargs, get_marker_value

logger = logging.getLogger(__name__)

_GW_DEFAULTS = {
    "policy": "round_robin",
    "timeout": DEFAULT_ROUTER_TIMEOUT,
    "extra_args": None,
    "log_level": None,
    "log_dir": None,
}

_WORKER_DEFAULTS = {"count": 1, "prefill": None, "decode": None}


def _start_gateway(gateway: Gateway, gateway_config: dict, **mode_kwargs) -> None:
    """Start gateway with mode-specific kwargs and shared config."""
    gateway.start(
        **mode_kwargs,
        policy=gateway_config["policy"],
        timeout=gateway_config["timeout"],
        extra_args=gateway_config["extra_args"],
        log_level=gateway_config.get("log_level"),
        log_dir=gateway_config.get("log_dir"),
    )


def _make_openai_client(gateway: Gateway) -> openai.OpenAI:
    return openai.OpenAI(base_url=f"{gateway.base_url}/v1", api_key="not-used")


# ---------------------------------------------------------------------------
# Main fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def setup_backend(request: pytest.FixtureRequest):
    """Class-scoped fixture that launches workers + gateway for each test class.

    Backend type is determined by parametrize value via ``request.param``:
      - ``"http"``, ``"grpc"``: Local workers (SGLang, vLLM, or TRT-LLM)
      - ``"pd_http"``, ``"pd_grpc"``: PD disaggregation workers
      - ``"openai"``, ``"xai"``, ``"anthropic"``: Cloud backends (no workers)

    Configuration via markers:
      - ``@pytest.mark.model("model-id")``: Override default model
      - ``@pytest.mark.workers(count=1)``: Number of regular workers
      - ``@pytest.mark.workers(prefill=1, decode=1)``: PD worker counts
      - ``@pytest.mark.gateway(policy=..., timeout=..., extra_args=...)``: Gateway config

    Returns:
        Tuple of ``(backend_name, model_path, client, gateway)``
    """
    backend_name: str = request.param

    if os.environ.get(ENV_SKIP_BACKEND_SETUP, "").lower() in ("1", "true", "yes"):
        pytest.skip(f"{ENV_SKIP_BACKEND_SETUP} is set")

    model_id = get_marker_value(request, "model")
    if model_id is None:
        model_id = os.environ.get(ENV_MODEL, DEFAULT_MODEL)

    gateway_config = get_marker_kwargs(request, "gateway", defaults=_GW_DEFAULTS)

    # Cloud backends (no local workers)
    if backend_name in THIRD_PARTY_MODELS:
        yield from _setup_cloud(backend_name, request, gateway_config)
        return

    # Local backends
    is_pd = backend_name.startswith("pd_")
    protocol = backend_name.replace("pd_", "")
    connection_mode = ConnectionMode(protocol)
    engine = get_runtime()
    model_path = get_model_spec(model_id)["model"]
    workers_config = get_marker_kwargs(request, "workers", defaults=_WORKER_DEFAULTS)
    log_dir = gateway_config.get("log_dir")

    gateway = Gateway()
    try:
        if is_pd:
            yield from _setup_pd(
                model_id,
                model_path,
                engine,
                connection_mode,
                workers_config,
                gateway_config,
                gateway,
                log_dir,
            )
        else:
            yield from _setup_local(
                model_id,
                model_path,
                engine,
                connection_mode,
                workers_config,
                gateway_config,
                gateway,
                backend_name,
                log_dir,
            )
    except Exception:
        gateway.shutdown()
        raise


# ---------------------------------------------------------------------------
# Local (non-PD) backend
# ---------------------------------------------------------------------------


def _setup_local(
    model_id,
    model_path,
    engine,
    connection_mode,
    workers_config,
    gateway_config,
    gateway,
    backend_name,
    log_dir,
):
    """Launch regular workers + gateway, yield result tuple, tear down."""
    num_workers = workers_config.get("count") or 1
    logger.info("Starting %s backend: model=%s, workers=%d", backend_name, model_id, num_workers)

    workers = start_workers(
        model_id,
        engine=engine,
        mode=connection_mode,
        count=num_workers,
        log_dir=log_dir,
    )
    try:
        _start_gateway(
            gateway,
            gateway_config,
            worker_urls=[w.base_url for w in workers],
            model_path=model_path,
        )
        logger.info("%s backend ready at %s", backend_name, gateway.base_url)
        yield backend_name, model_path, _make_openai_client(gateway), gateway
    finally:
        logger.info("Tearing down %s backend", backend_name)
        gateway.shutdown()
        stop_workers(workers)


# ---------------------------------------------------------------------------
# PD disaggregation backend
# ---------------------------------------------------------------------------


def _setup_pd(
    model_id,
    model_path,
    engine,
    connection_mode,
    workers_config,
    gateway_config,
    gateway,
    log_dir,
):
    """Launch prefill + decode workers + PD gateway, yield, tear down."""
    spec = get_model_spec(model_id)
    num_prefill = workers_config.get("prefill") or 1
    num_decode = workers_config.get("decode") or 1
    backend_name = f"pd_{connection_mode.value}"
    runtime_label = RUNTIME_LABELS.get(engine, engine)

    logger.info(
        "Starting %s PD backend: model=%s, %d prefill + %d decode",
        runtime_label,
        model_id,
        num_prefill,
        num_decode,
    )

    all_workers: list = []
    try:
        prefill_workers = start_workers(
            model_id,
            engine=engine,
            mode=connection_mode,
            count=num_prefill,
            worker_type=WorkerType.PREFILL,
            log_dir=log_dir,
        )
        all_workers.extend(prefill_workers)

        # Decode workers start on GPUs after prefill workers
        decode_gpu_offset = num_prefill * spec.get("tp", 1)
        decode_workers = start_workers(
            model_id,
            engine=engine,
            mode=connection_mode,
            count=num_decode,
            worker_type=WorkerType.DECODE,
            log_dir=log_dir,
            gpu_offset=decode_gpu_offset,
        )
        all_workers.extend(decode_workers)

        _start_gateway(
            gateway,
            gateway_config,
            prefill_workers=prefill_workers,
            decode_workers=decode_workers,
        )
        logger.info("%s PD backend ready at %s", runtime_label, gateway.base_url)
        yield backend_name, model_path, _make_openai_client(gateway), gateway
    finally:
        logger.info("Tearing down %s PD backend", runtime_label)
        gateway.shutdown()
        stop_workers(all_workers)


# ---------------------------------------------------------------------------
# Cloud backend
# ---------------------------------------------------------------------------


def _setup_cloud(backend_name, request, gateway_config):
    """Launch cloud gateway (no local workers), yield result tuple, tear down."""
    cfg = THIRD_PARTY_MODELS[backend_name]
    api_key_env = cfg.get("api_key_env")

    if api_key_env and not os.environ.get(api_key_env):
        pytest.fail(f"{api_key_env} not set for {backend_name} tests")

    storage_backend = get_marker_value(request, "storage", default="memory")

    logger.info("Launching cloud backend: %s (storage=%s)", backend_name, storage_backend)
    gateway = launch_cloud_gateway(
        backend_name,
        history_backend=storage_backend,
        extra_args=gateway_config.get("extra_args"),
    )

    api_key = os.environ.get(api_key_env) if api_key_env else "not-used"
    model_path = cfg["model"]

    client: openai.OpenAI | anthropic.Anthropic
    if cfg.get("client_type") == "anthropic":
        client = anthropic.Anthropic(base_url=gateway.base_url, api_key=api_key)
    else:
        client = openai.OpenAI(base_url=f"{gateway.base_url}/v1", api_key=api_key)

    try:
        yield backend_name, model_path, client, gateway
    finally:
        logger.info("Tearing down cloud backend: %s", backend_name)
        gateway.shutdown()


# ---------------------------------------------------------------------------
# Per-test gateway fixture (isolated router state)
# ---------------------------------------------------------------------------


@pytest.fixture
def backend_router(request: pytest.FixtureRequest):
    """Function-scoped fixture that launches a fresh gateway per test.

    Starts a single worker and a new gateway for each test function.
    Use when tests need isolated router state.

    Usage::

        @pytest.mark.parametrize("backend_router", ["grpc", "http"], indirect=True)
        def test_router_state(backend_router):
            gateway = backend_router
    """
    backend_name = request.param
    model_id = os.environ.get(ENV_MODEL, DEFAULT_MODEL)
    connection_mode = ConnectionMode(backend_name)
    model_path = get_model_spec(model_id)["model"]

    workers = start_workers(model_id, engine=get_runtime(), mode=connection_mode, count=1)
    gateway = Gateway()
    try:
        gateway.start(worker_urls=[w.base_url for w in workers], model_path=model_path)
        yield gateway
    finally:
        gateway.shutdown()
        stop_workers(workers)
