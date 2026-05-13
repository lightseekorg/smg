"""Backend setup fixtures for E2E tests.

Backend lifecycle: one gateway per test class, but the underlying workers
are pooled across classes that share ``(model_id, engine, mode, count)``.

Workers are the expensive part (cold start is 1-3 minutes); gateways start
in seconds. Decoupling worker lifetime from class scope lets consecutive
same-config classes reuse warm workers and just spin a fresh gateway.

Pool behaviour:
- Keyed on ``(model_id, engine, mode, count)``; PD and cloud backends are
  never pooled.
- LRU with a single-entry default (``E2E_WORKER_POOL_SIZE``) so GPU memory
  stays bounded.
- Flushed at session end via ``shutdown_worker_pool``; on eviction the
  evicted workers are stopped immediately.
- Opt out with ``E2E_DISABLE_WORKER_POOL=1`` (falls back to per-class
  start/stop semantics).
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from typing import NamedTuple

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

# Track worker startup failures — fail fast after repeated failures
_worker_start_failures: dict[str, int] = {}  # engine -> count
_MAX_WORKER_START_FAILURES = 3  # fail fast after this many failures (matches --reruns 2)


def _start_workers_tracked(**kwargs) -> list:
    """Start workers and track failures by engine for fail-fast."""
    engine = kwargs.get("engine") or get_runtime()
    try:
        return start_workers(**kwargs)
    except (TimeoutError, RuntimeError):
        _worker_start_failures[engine] = _worker_start_failures.get(engine, 0) + 1
        raise


def _require_exact_worker_count(*, role: str, requested: int, workers: list) -> None:
    """Fail the run if we did not acquire exactly the requested worker count."""
    got = len(workers) if workers is not None else 0
    if got != requested:
        pytest.fail(
            f"E2E worker acquisition failed: expected exactly {requested} {role} "
            f"worker(s), obtained {got}"
        )


# ---------------------------------------------------------------------------
# Session-scoped worker pool
# ---------------------------------------------------------------------------


class _WorkerKey(NamedTuple):
    model_id: str
    engine: str
    mode: str  # "http" | "grpc"
    worker_count: int


_worker_pool: OrderedDict[_WorkerKey, list] = OrderedDict()

_raw_pool_size = os.environ.get("E2E_WORKER_POOL_SIZE", "1") or "1"
try:
    _parsed_pool_size = int(_raw_pool_size)
except (TypeError, ValueError):
    logger.warning("Invalid E2E_WORKER_POOL_SIZE=%r, using 1", _raw_pool_size)
    _parsed_pool_size = 1
_POOL_MAX = max(1, _parsed_pool_size)

_POOL_DISABLED = os.environ.get("E2E_DISABLE_WORKER_POOL", "").lower() in ("1", "true", "yes")


def _pool_enabled() -> bool:
    return not _POOL_DISABLED


def _pool_get(key: _WorkerKey) -> list | None:
    """Return cached workers for ``key`` and mark them most-recently-used."""
    workers = _worker_pool.get(key)
    if workers is not None:
        _worker_pool.move_to_end(key)
    return workers


def _pool_put(key: _WorkerKey, workers: list) -> None:
    """Insert ``workers`` into the pool, evicting LRU entries past the cap.

    Eviction stops the evicted workers eagerly so GPU memory is reclaimed
    before the next cold start.
    """
    _worker_pool[key] = workers
    _worker_pool.move_to_end(key)
    while len(_worker_pool) > _POOL_MAX:
        evict_key = next(iter(_worker_pool))
        if evict_key == key:
            # Should not happen with _POOL_MAX >= 1, but guard anyway.
            break
        evict_workers = _worker_pool.pop(evict_key)
        logger.info("Evicting cached workers for %s", evict_key)
        stop_workers(evict_workers)


def _pool_make_room_for(key: _WorkerKey) -> None:
    """Evict LRU pool entries to free GPUs before launching ``key`` fresh.

    Pooled workers occupy GPUs 0..tp*count-1; a brand-new worker set for a
    different config would collide with them. Drop everything that does not
    match ``key`` until the pool has strictly less than ``_POOL_MAX`` entries,
    leaving room for the new entry. Called only on cache-miss so a true cache
    hit never evicts anyone.
    """
    while len(_worker_pool) >= _POOL_MAX:
        evict_key = next(iter(_worker_pool))
        if evict_key == key:
            # Pool already has our key (shouldn't happen on a miss); bail out.
            break
        evict_workers = _worker_pool.pop(evict_key)
        logger.info("Evicting cached workers for %s to free GPUs", evict_key)
        stop_workers(evict_workers)


def _pool_drop(key: _WorkerKey) -> list | None:
    """Remove a key from the pool without stopping its workers."""
    return _worker_pool.pop(key, None)


def shutdown_worker_pool() -> None:
    """Stop and forget every pooled worker. Called at session end."""
    while _worker_pool:
        key, workers = _worker_pool.popitem(last=False)
        logger.info("Stopping pooled workers for %s at session end", key)
        stop_workers(workers)


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
    log_dir = os.environ.get("E2E_LOG_DIR") or gateway_config.get("log_dir")

    fail_count = _worker_start_failures.get(engine, 0)
    if fail_count >= _MAX_WORKER_START_FAILURES:
        pytest.exit(
            f"Engine {engine} failed to start workers {fail_count} times — aborting test session",
            returncode=1,
        )

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
    """Launch (or reuse) regular workers + a fresh gateway, yield, tear down.

    Workers are pooled across test classes that share
    ``(model_id, engine, connection_mode, count)`` so the 1-3 min cold start
    is paid once per distinct config per session instead of per class. The
    gateway is always fresh — its startup is cheap and isolates router state.
    """
    num_workers = workers_config.get("count") or 1
    key = _WorkerKey(model_id, engine, connection_mode.value, num_workers)
    use_pool = _pool_enabled()

    cached = _pool_get(key) if use_pool else None
    if cached is not None:
        logger.info(
            "Reusing pooled workers for %s backend: model=%s, workers=%d",
            backend_name,
            model_id,
            num_workers,
        )
        workers = cached
        is_fresh = False
    else:
        if use_pool:
            # Free up GPUs occupied by stale pool entries before launching.
            _pool_make_room_for(key)
        logger.info(
            "Starting %s backend: model=%s, workers=%d", backend_name, model_id, num_workers
        )
        workers = _start_workers_tracked(
            model_id=model_id,
            engine=engine,
            mode=connection_mode,
            count=num_workers,
            log_dir=log_dir,
        )
        is_fresh = True

    _require_exact_worker_count(
        role=f"{backend_name} ({model_id})",
        requested=num_workers,
        workers=workers,
    )

    gateway_started = False
    try:
        _start_gateway(
            gateway,
            gateway_config,
            worker_urls=[w.base_url for w in workers],
            model_path=model_path,
        )
        gateway_started = True
        logger.info("%s backend ready at %s", backend_name, gateway.base_url)
        yield backend_name, model_path, _make_openai_client(gateway), gateway
    finally:
        logger.info("Tearing down %s backend", backend_name)
        gateway.shutdown()
        if not use_pool:
            if is_fresh:
                stop_workers(workers)
        elif gateway_started:
            # Gateway came up healthy → workers are good to keep warm.
            if is_fresh:
                _pool_put(key, workers)
            # else: cached workers stayed in pool throughout (never popped).
        else:
            # Gateway never came up: treat these workers as suspect.
            if is_fresh:
                logger.info("Discarding workers after gateway start failure")
                stop_workers(workers)
            else:
                evicted = _pool_drop(key)
                if evicted is not None:
                    logger.info("Evicting cached workers after gateway start failure")
                    stop_workers(evicted)


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

    # PD workers assign GPUs starting at 0, which would collide with any
    # regular (non-PD) workers kept warm in the pool. Drain the pool first
    # so the GPUs are free.
    if _worker_pool:
        logger.info("Draining %d pooled worker set(s) before PD setup", len(_worker_pool))
        shutdown_worker_pool()

    all_workers: list = []
    try:
        prefill_workers = _start_workers_tracked(
            model_id=model_id,
            engine=engine,
            mode=connection_mode,
            count=num_prefill,
            worker_type=WorkerType.PREFILL,
            log_dir=log_dir,
        )
        _require_exact_worker_count(
            role=f"PD prefill ({model_id})",
            requested=num_prefill,
            workers=prefill_workers,
        )
        all_workers.extend(prefill_workers)

        # Decode workers start on GPUs after prefill workers
        decode_gpu_offset = num_prefill * spec.get("tp", 1)
        decode_workers = _start_workers_tracked(
            model_id=model_id,
            engine=engine,
            mode=connection_mode,
            count=num_decode,
            worker_type=WorkerType.DECODE,
            log_dir=log_dir,
            gpu_offset=decode_gpu_offset,
        )
        _require_exact_worker_count(
            role=f"PD decode ({model_id})",
            requested=num_decode,
            workers=decode_workers,
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
    _require_exact_worker_count(
        role=f"backend_router {backend_name} ({model_id})",
        requested=1,
        workers=workers,
    )
    gateway = Gateway()
    try:
        gateway.start(worker_urls=[w.base_url for w in workers], model_path=model_path)
        yield gateway
    finally:
        gateway.shutdown()
        stop_workers(workers)
