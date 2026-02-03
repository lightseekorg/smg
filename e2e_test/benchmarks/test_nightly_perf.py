"""Nightly comprehensive benchmark tests.

Runs all models on an 8-GPU H200 node using genai-bench default scenarios
and concurrency levels. No performance thresholds — results are uploaded
as artifacts for tracking over time.

Each model has Single (1 worker) and Multi (N workers) classes, both
parametrized with http and grpc backends. The workflow matrix crosses
model × variant (single/multi × sglang/vllm), filtering vllm to grpc-only.

genai-bench defaults (omitted flags):
  - Concurrency: [1, 2, 4, 8, 16, 32, 64, 128, 256]
  - Text scenarios: N(480,240)/(300,150), D(100,100), D(100,1000),
                     D(2000,200), D(7800,200)
  - Embedding scenarios: E(64), E(128), E(256), E(512), E(1024)
"""

import os

import pytest
from infra import get_runtime


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_MAX_REQUESTS = 300
_MAX_TIME_PER_RUN = 10  # seconds per scenario×concurrency combo
_TIMEOUT_SEC = 10800  # 3 hours per model

# TODO: revert before merge — fast settings for PR testing
_TEST_MODE = True
_TEST_NUM_CONCURRENCY = 1
_TEST_TRAFFIC_SCENARIO = "D(100,100)"
_TEST_MAX_REQUESTS = 10


def _run_nightly(setup_backend, genai_bench_runner, model_id, **kwargs):
    """Run nightly benchmark for a model with genai-bench defaults."""
    backend, model_path, client, gateway = setup_backend

    # Get runtime and GPU info for metadata
    runtime = get_runtime()  # sglang or vllm from E2E_RUNTIME env var
    gpu_type = os.environ.get("GPU_TYPE", "H200")
    gpu_count = int(os.environ.get("GPU_COUNT", "8"))

    # Include runtime in folder name to distinguish sglang vs vllm results
    experiment_folder = f"nightly_{model_id}_{backend}_{runtime}"

    if _TEST_MODE:
        genai_bench_runner(
            router_url=gateway.base_url,
            model_path=model_path,
            experiment_folder=experiment_folder,
            num_concurrency=_TEST_NUM_CONCURRENCY,
            traffic_scenario=_TEST_TRAFFIC_SCENARIO,
            max_requests_per_run=_TEST_MAX_REQUESTS,
            timeout_sec=300,
            server_engine=runtime,
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            **kwargs,
        )
        return

    genai_bench_runner(
        router_url=gateway.base_url,
        model_path=model_path,
        experiment_folder=experiment_folder,
        num_concurrency=None,      # use genai-bench defaults
        traffic_scenario=None,     # use genai-bench defaults
        max_requests_per_run=_MAX_REQUESTS,
        max_time_per_run=_MAX_TIME_PER_RUN,
        timeout_sec=_TIMEOUT_SEC,
        server_engine=runtime,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Model configurations: (model_id, class_name_fragment, multi_workers, backends, extra_kwargs)
# backends: list of backends to test (default: ["http", "grpc"])
# ---------------------------------------------------------------------------

_NIGHTLY_MODELS = [
    ("llama-8b",    "Llama8b",    8, ["http", "grpc"], {}),
    # ("llama-1b",    "Llama1b",    8, ["http", "grpc"], {}),
    ("qwen-7b",     "Qwen7b",     8, ["http", "grpc"], {}),
    # ("qwen-14b",    "Qwen14b",    4, ["http", "grpc"], {}),
    # ("deepseek-7b", "Deepseek7b", 8, ["http", "grpc"], {}),
    # ("qwen-30b",    "Qwen30b",    2, ["http", "grpc"], {}),
    # ("mistral-7b",  "Mistral7b",  8, ["http", "grpc"], {}),
    ("gpt-oss",     "GptOss",     4, ["http", "grpc"], {}),  # Skip gRPC (vLLM not supported)
    # ("llama-4-maverick-17b", "Llama4Maverick", 1, ["http", "grpc"], {}),  # 1 worker uses all 8 GPUs (tp=8)
]


# ---------------------------------------------------------------------------
# Dynamic test class generation
# ---------------------------------------------------------------------------


def _make_test_class(model_id, worker_count, backends, extra_kwargs):
    """Create a nightly benchmark test class for a model/worker configuration."""

    @pytest.mark.nightly
    @pytest.mark.e2e
    @pytest.mark.model(model_id)
    @pytest.mark.workers(count=worker_count)
    @pytest.mark.gateway(policy="round_robin")
    @pytest.mark.parametrize("setup_backend", backends, indirect=True)
    class _NightlyTest:
        def test_nightly_perf(self, setup_backend, genai_bench_runner):
            _run_nightly(setup_backend, genai_bench_runner, model_id, **extra_kwargs)

    return _NightlyTest


for _model_id, _name, _multi_workers, _backends, _extra in _NIGHTLY_MODELS:
    for _suffix, _count in [("Single", 1), ("Multi", _multi_workers)]:
        _cls_name = f"TestNightly{_name}{_suffix}"
        _cls = _make_test_class(_model_id, _count, _backends, _extra)
        _cls.__name__ = _cls_name
        _cls.__qualname__ = _cls_name
        globals()[_cls_name] = _cls

# Clean up loop variables from module namespace
del _model_id, _name, _multi_workers, _backends, _extra, _suffix, _count, _cls_name, _cls
