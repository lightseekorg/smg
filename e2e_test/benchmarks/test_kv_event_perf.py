"""Cache-aware KV event benchmark coverage for vLLM gRPC workers."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass

import pytest

from infra import ConnectionMode, Gateway, get_runtime
from infra.constants import ENV_VLLM_ENABLE_KV_EVENTS
from infra.model_specs import get_model_spec
from infra.worker import start_workers, stop_workers

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
_WORKER_COUNT = 4
_NIGHTLY_ONLY = pytest.mark.skipif(
    os.environ.get("E2E_NIGHTLY") != "1",
    reason="extreme KV event benchmark only runs in nightly mode",
)
_TEST_MODE = os.environ.get("BENCH_TEST_MODE", "0") == "1"


@dataclass(frozen=True)
class BenchmarkProfile:
    """Single benchmark profile for a routing policy run."""

    name: str
    num_concurrency: int
    traffic_scenario: str
    max_requests_per_run: int
    max_time_per_run: int


_STANDARD_PROFILE = BenchmarkProfile(
    name="regular",
    num_concurrency=8 if _TEST_MODE else 32,
    traffic_scenario="D(100,100)" if _TEST_MODE else "D(4000,100)",
    max_requests_per_run=16 if _TEST_MODE else 200,
    max_time_per_run=3,
)
_EXTREME_PROFILE = BenchmarkProfile(
    name="extreme",
    num_concurrency=16 if _TEST_MODE else 128,
    traffic_scenario="D(2000,200)" if _TEST_MODE else "D(7800,200)",
    max_requests_per_run=32 if _TEST_MODE else 512,
    max_time_per_run=3 if _TEST_MODE else 10,
)


@contextmanager
def _with_vllm_kv_events(enabled: bool):
    """Temporarily toggle vLLM KV event streaming for worker launch."""
    old_value = os.environ.get(ENV_VLLM_ENABLE_KV_EVENTS)
    if enabled:
        os.environ[ENV_VLLM_ENABLE_KV_EVENTS] = "1"
    else:
        os.environ.pop(ENV_VLLM_ENABLE_KV_EVENTS, None)

    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(ENV_VLLM_ENABLE_KV_EVENTS, None)
        else:
            os.environ[ENV_VLLM_ENABLE_KV_EVENTS] = old_value


@contextmanager
def _running_gateway(model_id: str, policy: str, enable_kv_events: bool):
    """Launch a fresh gRPC worker set and gateway for a single benchmark run."""
    runtime = get_runtime()
    model_path = get_model_spec(model_id)["model"]
    log_dir = os.environ.get("E2E_LOG_DIR")
    workers = []
    gateway = Gateway()

    with _with_vllm_kv_events(enable_kv_events):
        try:
            workers = start_workers(
                model_id=model_id,
                engine=runtime,
                mode=ConnectionMode.GRPC,
                count=_WORKER_COUNT,
                log_dir=log_dir,
            )
            gateway.start(
                worker_urls=[worker.base_url for worker in workers],
                model_path=model_path,
                policy=policy,
                log_level="debug" if os.environ.get("E2E_NIGHTLY") == "1" else None,
                log_dir=log_dir,
            )
            yield model_path, gateway
        finally:
            gateway.shutdown()
            stop_workers(workers)


def _run_policy_benchmark(
    genai_bench_runner,
    *,
    model_id: str,
    policy: str,
    enable_kv_events: bool,
    profile: BenchmarkProfile,
) -> None:
    """Run one benchmark scenario with a clean worker/gateway stack."""
    if get_runtime() != "vllm":
        pytest.skip("KV event benchmark requires E2E_RUNTIME=vllm")

    with _running_gateway(model_id, policy, enable_kv_events) as (model_path, gateway):
        kv_events_mode = "kv_events_on" if enable_kv_events else "kv_events_off"
        experiment_folder = f"benchmark_{policy}_{kv_events_mode}_{profile.name}_grpc"
        logger.info(
            "Running %s benchmark for %s: policy=%s kv_events=%s",
            profile.name,
            model_id,
            policy,
            enable_kv_events,
        )
        genai_bench_runner(
            router_url=gateway.base_url,
            model_path=model_path,
            experiment_folder=experiment_folder,
            num_concurrency=profile.num_concurrency,
            traffic_scenario=profile.traffic_scenario,
            max_requests_per_run=profile.max_requests_per_run,
            max_time_per_run=profile.max_time_per_run,
        )


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.skip_for_runtime("sglang", "trtllm", reason="KV event benchmark targets vLLM gRPC")
class TestKvEventPerf:
    """Baseline and cache-aware benchmark coverage for the KV event path."""

    def test_round_robin_baseline(self, genai_bench_runner):
        _run_policy_benchmark(
            genai_bench_runner,
            model_id=_DEFAULT_MODEL_ID,
            policy="round_robin",
            enable_kv_events=False,
            profile=_STANDARD_PROFILE,
        )

    def test_cache_aware_with_kv_events(self, genai_bench_runner):
        _run_policy_benchmark(
            genai_bench_runner,
            model_id=_DEFAULT_MODEL_ID,
            policy="cache_aware",
            enable_kv_events=True,
            profile=_STANDARD_PROFILE,
        )

    @pytest.mark.nightly
    @_NIGHTLY_ONLY
    def test_round_robin_baseline_extreme(self, genai_bench_runner):
        _run_policy_benchmark(
            genai_bench_runner,
            model_id=_DEFAULT_MODEL_ID,
            policy="round_robin",
            enable_kv_events=False,
            profile=_EXTREME_PROFILE,
        )

    @pytest.mark.nightly
    @_NIGHTLY_ONLY
    def test_cache_aware_with_kv_events_extreme(self, genai_bench_runner):
        _run_policy_benchmark(
            genai_bench_runner,
            model_id=_DEFAULT_MODEL_ID,
            policy="cache_aware",
            enable_kv_events=True,
            profile=_EXTREME_PROFILE,
        )
