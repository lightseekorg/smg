"""Backend setup fixtures for E2E tests.

This module provides fixtures for launching gateways/routers for different backends.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from infra import ModelPool

from infra import RUNTIME_LABELS, get_runtime, is_trtllm, is_vllm

from .markers import get_marker_kwargs, get_marker_value

logger = logging.getLogger(__name__)


@pytest.fixture(scope="class")
def setup_backend(request: pytest.FixtureRequest, model_pool: "ModelPool"):
    """Class-scoped fixture that launches a router for each test class.

    Routers are cheap to start (~1-2s) compared to workers (~30-60s), so we
    launch a fresh router per test class for isolation while reusing the
    expensive workers from model_pool.

    Backend types:
    - "http", "grpc": Gets existing worker from model_pool, launches router
    - "pd_http", "pd_grpc": Launches prefill/decode workers via model_pool, launches PD router
    - "openai", "xai", etc.: Launches cloud router (no local workers)

    Configuration via markers:
    - @pytest.mark.model("model-id"): Override default model
    - @pytest.mark.workers(count=1): Number of regular workers behind router
    - @pytest.mark.workers(prefill=1, decode=1): PD worker configuration
    - @pytest.mark.gateway(policy="round_robin", timeout=60): Gateway configuration

    Returns:
        Tuple of (backend_name, model_path, openai_client, gateway)

    Usage:
        @pytest.mark.parametrize("setup_backend", ["http"], indirect=True)
        class TestBasic:
            def test_chat(self, setup_backend):
                backend, model, client, gateway = setup_backend
    """
    import openai
    from infra import (
        DEFAULT_MODEL,
        DEFAULT_ROUTER_TIMEOUT,
        ENV_MODEL,
        ENV_SKIP_BACKEND_SETUP,
        LOCAL_MODES,
        ConnectionMode,
        Gateway,
        WorkerIdentity,
        WorkerType,
    )

    backend_name = request.param

    # Skip if requested
    if os.environ.get(ENV_SKIP_BACKEND_SETUP, "").lower() in ("1", "true", "yes"):
        pytest.skip(f"{ENV_SKIP_BACKEND_SETUP} is set")

    # Get model from marker or env var or default
    model_id = get_marker_value(request, "model")
    if model_id is None:
        model_id = os.environ.get(ENV_MODEL, DEFAULT_MODEL)

    # Get worker configuration from marker
    workers_config = get_marker_kwargs(
        request, "workers", defaults={"count": 1, "prefill": None, "decode": None}
    )

    # Get gateway configuration from marker
    gateway_config = get_marker_kwargs(
        request,
        "gateway",
        defaults={
            "policy": "round_robin",
            "timeout": DEFAULT_ROUTER_TIMEOUT,
            "extra_args": None,
        },
    )

    # PD disaggregation backends - explicit connection modes
    if backend_name == "pd_http":
        yield from _setup_pd_http_backend(
            request, model_pool, model_id, workers_config, gateway_config
        )
        return

    if backend_name == "pd_grpc":
        # gRPC mode PD
        yield from _setup_pd_grpc_backend(
            request, model_pool, model_id, workers_config, gateway_config
        )
        return

    # Check if this is a local backend (grpc, http)
    try:
        connection_mode = ConnectionMode(backend_name)
        is_local = connection_mode in LOCAL_MODES
    except ValueError:
        is_local = False
        connection_mode = None

    # Local backends: check runtime environment variable for gRPC mode
    if is_local:
        # For gRPC mode, check E2E_RUNTIME environment variable
        if connection_mode == ConnectionMode.GRPC:
            runtime = get_runtime()
            runtime_label = RUNTIME_LABELS.get(runtime, "SGLang")
            logger.info(
                "gRPC backend detected: E2E_RUNTIME=%s, routing to %s backend",
                runtime,
                runtime_label,
            )

            # Route to runtime-specific gRPC backend (vLLM, TRT-LLM)
            if is_vllm() or is_trtllm():
                yield from _setup_grpc_backend(
                    request, model_pool, model_id, workers_config, gateway_config
                )
                return

        # Otherwise use regular local backend (sglang grpc or http)
        yield from _setup_local_backend(
            request,
            model_pool,
            backend_name,
            model_id,
            connection_mode,
            workers_config,
            gateway_config,
        )
        return

    # Get storage backend from marker (default: memory)
    storage_backend = get_marker_value(request, "storage", default="memory")

    # Cloud backends: launch cloud router
    yield from _setup_cloud_backend(backend_name, storage_backend, gateway_config)


def _setup_pd_http_backend(
    request: pytest.FixtureRequest,
    model_pool: "ModelPool",
    model_id: str,
    workers_config: dict,
    gateway_config: dict,
):
    """Setup SGLang PD disaggregation backend (HTTP mode with bootstrap)."""
    from infra import ConnectionMode

    yield from _setup_pd_backend_common(
        model_pool=model_pool,
        model_id=model_id,
        workers_config=workers_config,
        gateway_config=gateway_config,
        connection_mode=ConnectionMode.HTTP,
        backend_name="pd_http",
    )


def _setup_pd_grpc_backend(
    request: pytest.FixtureRequest,
    model_pool: "ModelPool",
    model_id: str,
    workers_config: dict,
    gateway_config: dict,
):
    """Setup PD disaggregation backend with gRPC mode."""
    from infra import ConnectionMode

    yield from _setup_pd_backend_common(
        model_pool=model_pool,
        model_id=model_id,
        workers_config=workers_config,
        gateway_config=gateway_config,
        connection_mode=ConnectionMode.GRPC,
        backend_name="pd_grpc",
    )


def _setup_pd_backend_common(
    model_pool: "ModelPool",
    model_id: str,
    workers_config: dict,
    gateway_config: dict,
    connection_mode,
    backend_name: str,
):
    """Common setup for PD disaggregation backends.

    Args:
        model_pool: The model pool instance.
        model_id: Model identifier.
        workers_config: Worker configuration from markers.
        gateway_config: Gateway configuration from markers.
        connection_mode: ConnectionMode.HTTP for SGLang, ConnectionMode.GRPC for vLLM.
        backend_name: Backend name to yield ("pd_http" or "pd_grpc").
    """
    import openai
    from infra import Gateway, WorkerIdentity, WorkerType

    runtime = get_runtime()
    runtime_label = RUNTIME_LABELS.get(runtime, "SGLang")
    logger.info("Setting up %s PD backend for model %s", runtime_label, model_id)

    num_prefill = workers_config.get("prefill") or 1
    num_decode = workers_config.get("decode") or 1
    logger.info(
        "%s PD config: %d prefill, %d decode workers",
        runtime_label,
        num_prefill,
        num_decode,
    )

    # get_workers_by_type auto-acquires all returned workers
    # Filter by connection_mode to ensure we use the right worker type (HTTP vs gRPC)
    all_prefills = model_pool.get_workers_by_type(model_id, WorkerType.PREFILL)
    all_decodes = model_pool.get_workers_by_type(model_id, WorkerType.DECODE)

    # Filter by connection mode and release workers we won't use
    existing_prefills = [w for w in all_prefills if w.mode == connection_mode]
    existing_decodes = [w for w in all_decodes if w.mode == connection_mode]

    # Release workers that don't match the requested connection mode
    for w in all_prefills:
        if w not in existing_prefills:
            w.release()
    for w in all_decodes:
        if w not in existing_decodes:
            w.release()

    missing_prefill = max(0, num_prefill - len(existing_prefills))
    missing_decode = max(0, num_decode - len(existing_decodes))

    if missing_prefill == 0 and missing_decode == 0:
        prefills = existing_prefills[:num_prefill]
        decodes = existing_decodes[:num_decode]
        for w in existing_prefills[num_prefill:]:
            w.release()
        for w in existing_decodes[num_decode:]:
            w.release()
        logger.info(
            "Using pre-launched %s PD workers: %d prefill, %d decode",
            runtime_label,
            len(prefills),
            len(decodes),
        )
    else:
        workers_to_launch: list[WorkerIdentity] = []
        for i in range(missing_prefill):
            workers_to_launch.append(
                WorkerIdentity(
                    model_id,
                    connection_mode,
                    WorkerType.PREFILL,
                    len(existing_prefills) + i,
                )
            )
        for i in range(missing_decode):
            workers_to_launch.append(
                WorkerIdentity(
                    model_id,
                    connection_mode,
                    WorkerType.DECODE,
                    len(existing_decodes) + i,
                )
            )

        logger.info(
            "Have %d/%d prefill, %d/%d decode. Launching %d more workers",
            len(existing_prefills),
            num_prefill,
            len(existing_decodes),
            num_decode,
            len(workers_to_launch),
        )
        new_instances = model_pool.launch_workers(
            workers_to_launch, startup_timeout=300
        )

        if not new_instances:
            for w in existing_prefills + existing_decodes:
                w.release()
            pytest.fail(
                f"Failed to launch {runtime_label} PD workers: needed "
                f"{len(workers_to_launch)} workers but could not allocate GPUs"
            )

        for inst in new_instances:
            inst.acquire()

        new_prefills = [w for w in new_instances if w.worker_type == WorkerType.PREFILL]
        new_decodes = [w for w in new_instances if w.worker_type == WorkerType.DECODE]
        prefills = existing_prefills + new_prefills
        decodes = existing_decodes + new_decodes

    if not prefills or not decodes:
        for w in prefills + decodes:
            w.release()
        pytest.fail(
            f"{runtime_label} PD setup incomplete: have {len(prefills)} prefill, "
            f"{len(decodes)} decode (need {num_prefill} prefill, {num_decode} decode)"
        )

    model_path = prefills[0].model_path

    gateway = Gateway()
    gateway.start(
        prefill_workers=prefills,
        decode_workers=decodes,
        policy=gateway_config["policy"],
        timeout=gateway_config["timeout"],
        extra_args=gateway_config["extra_args"],
    )

    client = openai.OpenAI(
        base_url=f"{gateway.base_url}/v1",
        api_key="not-used",
    )

    logger.info(
        "Setup %s PD backend: model=%s, %d prefill + %d decode workers, "
        "gateway=%s, policy=%s",
        runtime_label,
        model_id,
        len(prefills),
        len(decodes),
        gateway.base_url,
        gateway_config["policy"],
    )

    try:
        yield backend_name, model_path, client, gateway
    finally:
        logger.info("Tearing down %s PD gateway", runtime_label)
        gateway.shutdown()
        for worker in prefills + decodes:
            worker.release()


def _setup_grpc_backend(
    request: pytest.FixtureRequest,
    model_pool: "ModelPool",
    model_id: str,
    workers_config: dict,
    gateway_config: dict,
):
    """Setup a runtime-specific gRPC backend (vLLM or TensorRT-LLM)."""
    import openai
    from infra import Gateway

    runtime = get_runtime()
    runtime_label = RUNTIME_LABELS.get(runtime, runtime)

    logger.info("Setting up %s gRPC backend for model %s", runtime_label, model_id)

    try:
        instance = model_pool.get_grpc_worker(model_id)
    except RuntimeError as e:
        pytest.fail(str(e))

    model_path = instance.model_path
    worker_urls = [instance.worker_url]

    gateway = Gateway()
    gateway.start(
        worker_urls=worker_urls,
        model_path=model_path,
        policy=gateway_config["policy"],
        timeout=gateway_config["timeout"],
        extra_args=gateway_config["extra_args"],
    )

    client = openai.OpenAI(
        base_url=f"{gateway.base_url}/v1",
        api_key="not-used",
    )

    logger.info(
        "Setup %s gRPC backend: model=%s, worker=%s, gateway=%s, policy=%s",
        runtime_label,
        model_id,
        instance.worker_url,
        gateway.base_url,
        gateway_config["policy"],
    )

    try:
        yield "grpc", model_path, client, gateway
    finally:
        logger.info("Tearing down %s gRPC gateway", runtime_label)
        gateway.shutdown()
        instance.release()


def _setup_local_backend(
    request: pytest.FixtureRequest,
    model_pool: "ModelPool",
    backend_name: str,
    model_id: str,
    connection_mode,
    workers_config: dict,
    gateway_config: dict,
):
    """Setup local backend (grpc, http)."""
    import openai
    from infra import Gateway, WorkerIdentity, WorkerType

    num_workers = workers_config.get("count") or 1
    instances: list = []  # Track instances for reference counting

    try:
        if num_workers > 1:
            # get_workers_by_type auto-acquires all returned workers
            all_existing = model_pool.get_workers_by_type(model_id, WorkerType.REGULAR)
            existing_for_mode = [w for w in all_existing if w.mode == connection_mode]

            # Release workers we won't use (wrong mode)
            for w in all_existing:
                if w not in existing_for_mode:
                    w.release()

            if len(existing_for_mode) >= num_workers:
                instances = existing_for_mode[:num_workers]
                # Release excess workers we won't use
                for w in existing_for_mode[num_workers:]:
                    w.release()
            else:
                missing = num_workers - len(existing_for_mode)
                workers_to_launch = [
                    WorkerIdentity(
                        model_id,
                        connection_mode,
                        WorkerType.REGULAR,
                        len(existing_for_mode) + i,
                    )
                    for i in range(missing)
                ]
                new_instances = model_pool.launch_workers(
                    workers_to_launch, startup_timeout=300
                )
                # Acquire newly launched instances
                for inst in new_instances:
                    inst.acquire()
                instances = existing_for_mode + new_instances

            if not instances:
                pytest.fail(f"Failed to get {num_workers} workers for {model_id}")
            worker_urls = [inst.worker_url for inst in instances]
            model_path = instances[0].model_path
        else:
            # get() auto-acquires the returned instance
            instance = model_pool.get(model_id, connection_mode)
            instances = [instance]
            worker_urls = [instance.worker_url]
            model_path = instance.model_path
    except RuntimeError as e:
        pytest.fail(str(e))

    # Launch gateway
    gateway = Gateway()
    gateway.start(
        worker_urls=worker_urls,
        model_path=model_path,
        policy=gateway_config["policy"],
        timeout=gateway_config["timeout"],
        extra_args=gateway_config["extra_args"],
    )

    client = openai.OpenAI(
        base_url=f"{gateway.base_url}/v1",
        api_key="not-used",
    )

    logger.info(
        "Setup %s backend: model=%s, workers=%d, gateway=%s, policy=%s",
        backend_name,
        model_id,
        num_workers,
        gateway.base_url,
        gateway_config["policy"],
    )

    try:
        yield backend_name, model_path, client, gateway
    finally:
        logger.info("Tearing down gateway for %s backend", backend_name)
        gateway.shutdown()
        # Release references to allow eviction
        for inst in instances:
            inst.release()


def _setup_cloud_backend(
    backend_name: str,
    storage_backend: str = "memory",
    gateway_config: dict | None = None,
):
    """Setup cloud backend (openai, xai, etc.).

    Args:
        backend_name: Cloud backend name (openai, xai).
        storage_backend: History storage backend (memory, oracle).
        gateway_config: Gateway configuration from marker.
    """
    import openai
    from infra import THIRD_PARTY_MODELS, launch_cloud_gateway

    if backend_name not in THIRD_PARTY_MODELS:
        pytest.fail(f"Unknown cloud runtime: {backend_name}")

    cfg = THIRD_PARTY_MODELS[backend_name]
    api_key_env = cfg.get("api_key_env")

    if api_key_env and not os.environ.get(api_key_env):
        pytest.skip(f"{api_key_env} not set, skipping {backend_name} tests")

    extra_args = gateway_config.get("extra_args") if gateway_config else None

    logger.info(
        "Launching cloud backend: %s with storage=%s", backend_name, storage_backend
    )
    gateway = launch_cloud_gateway(
        backend_name,
        history_backend=storage_backend,
        extra_args=extra_args,
    )

    api_key = os.environ.get(api_key_env) if api_key_env else "not-used"
    client = openai.OpenAI(
        base_url=f"{gateway.base_url}/v1",
        api_key=api_key,
    )

    try:
        yield backend_name, cfg["model"], client, gateway
    finally:
        logger.info("Tearing down cloud backend: %s", backend_name)
        gateway.shutdown()


@pytest.fixture
def backend_router(request: pytest.FixtureRequest, model_pool: "ModelPool"):
    """Function-scoped fixture for launching a fresh router per test.

    This launches a new Gateway for each test, pointing to workers from the pool.
    Use for tests that need isolated router state.

    Usage:
        @pytest.mark.parametrize("backend_router", ["grpc", "http"], indirect=True)
        def test_router_state(backend_router):
            gateway = backend_router
    """
    from infra import DEFAULT_MODEL, ENV_MODEL, ConnectionMode, Gateway

    backend_name = request.param
    model_id = os.environ.get(ENV_MODEL, DEFAULT_MODEL)

    connection_mode = ConnectionMode(backend_name)

    try:
        # get() auto-acquires the returned instance
        instance = model_pool.get(model_id, connection_mode)
    except KeyError:
        pytest.skip(f"Model {model_id}:{backend_name} not available in pool")
    except RuntimeError as e:
        pytest.fail(str(e))

    gateway = Gateway()
    gateway.start(
        worker_urls=[instance.worker_url],
        model_path=instance.model_path,
    )

    try:
        yield gateway
    finally:
        gateway.shutdown()
        # Release reference to allow eviction
        instance.release()
