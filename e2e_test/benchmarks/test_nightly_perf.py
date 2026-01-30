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

import pytest


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

    if _TEST_MODE:
        genai_bench_runner(
            router_url=gateway.base_url,
            model_path=model_path,
            experiment_folder=f"nightly_{model_id}_{backend}",
            num_concurrency=_TEST_NUM_CONCURRENCY,
            traffic_scenario=_TEST_TRAFFIC_SCENARIO,
            max_requests_per_run=_TEST_MAX_REQUESTS,
            timeout_sec=300,
            **kwargs,
        )
        return

    genai_bench_runner(
        router_url=gateway.base_url,
        model_path=model_path,
        experiment_folder=f"nightly_{model_id}_{backend}",
        num_concurrency=None,      # use genai-bench defaults
        traffic_scenario=None,     # use genai-bench defaults
        max_requests_per_run=_MAX_REQUESTS,
        max_time_per_run=_MAX_TIME_PER_RUN,
        timeout_sec=_TIMEOUT_SEC,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Shared markers
# ---------------------------------------------------------------------------

_nightly = pytest.mark.nightly
_e2e = pytest.mark.e2e
_backends = pytest.mark.parametrize("setup_backend", ["http", "grpc"], indirect=True)
_gateway = pytest.mark.gateway(policy="round_robin")


# ---------------------------------------------------------------------------
# llama-8b (tp=1 → single=1, multi=8)
# ---------------------------------------------------------------------------


@_nightly
@_e2e
@pytest.mark.model("llama-8b")
@pytest.mark.workers(count=1)
@_gateway
@_backends
class TestNightlyLlama8bSingle:
    def test_nightly_perf(self, setup_backend, genai_bench_runner):
        _run_nightly(setup_backend, genai_bench_runner, "llama-8b")


@_nightly
@_e2e
@pytest.mark.model("llama-8b")
@pytest.mark.workers(count=8)
@_gateway
@_backends
class TestNightlyLlama8bMulti:
    def test_nightly_perf(self, setup_backend, genai_bench_runner):
        _run_nightly(setup_backend, genai_bench_runner, "llama-8b")


# ---------------------------------------------------------------------------
# llama-1b (tp=1 → single=1, multi=8)
# ---------------------------------------------------------------------------


@_nightly
@_e2e
@pytest.mark.model("llama-1b")
@pytest.mark.workers(count=1)
@_gateway
@_backends
class TestNightlyLlama1bSingle:
    def test_nightly_perf(self, setup_backend, genai_bench_runner):
        _run_nightly(setup_backend, genai_bench_runner, "llama-1b")


@_nightly
@_e2e
@pytest.mark.model("llama-1b")
@pytest.mark.workers(count=8)
@_gateway
@_backends
class TestNightlyLlama1bMulti:
    def test_nightly_perf(self, setup_backend, genai_bench_runner):
        _run_nightly(setup_backend, genai_bench_runner, "llama-1b")


# ---------------------------------------------------------------------------
# qwen-7b (tp=1 → single=1, multi=8)
# ---------------------------------------------------------------------------


@_nightly
@_e2e
@pytest.mark.model("qwen-7b")
@pytest.mark.workers(count=1)
@_gateway
@_backends
class TestNightlyQwen7bSingle:
    def test_nightly_perf(self, setup_backend, genai_bench_runner):
        _run_nightly(setup_backend, genai_bench_runner, "qwen-7b")


@_nightly
@_e2e
@pytest.mark.model("qwen-7b")
@pytest.mark.workers(count=8)
@_gateway
@_backends
class TestNightlyQwen7bMulti:
    def test_nightly_perf(self, setup_backend, genai_bench_runner):
        _run_nightly(setup_backend, genai_bench_runner, "qwen-7b")


# ---------------------------------------------------------------------------
# qwen-14b (tp=2 → single=1, multi=4)
# ---------------------------------------------------------------------------


@_nightly
@_e2e
@pytest.mark.model("qwen-14b")
@pytest.mark.workers(count=1)
@_gateway
@_backends
class TestNightlyQwen14bSingle:
    def test_nightly_perf(self, setup_backend, genai_bench_runner):
        _run_nightly(setup_backend, genai_bench_runner, "qwen-14b")


@_nightly
@_e2e
@pytest.mark.model("qwen-14b")
@pytest.mark.workers(count=4)
@_gateway
@_backends
class TestNightlyQwen14bMulti:
    def test_nightly_perf(self, setup_backend, genai_bench_runner):
        _run_nightly(setup_backend, genai_bench_runner, "qwen-14b")


# ---------------------------------------------------------------------------
# deepseek-7b (tp=1 → single=1, multi=8)
# ---------------------------------------------------------------------------


@_nightly
@_e2e
@pytest.mark.model("deepseek-7b")
@pytest.mark.workers(count=1)
@_gateway
@_backends
class TestNightlyDeepseek7bSingle:
    def test_nightly_perf(self, setup_backend, genai_bench_runner):
        _run_nightly(setup_backend, genai_bench_runner, "deepseek-7b")


@_nightly
@_e2e
@pytest.mark.model("deepseek-7b")
@pytest.mark.workers(count=8)
@_gateway
@_backends
class TestNightlyDeepseek7bMulti:
    def test_nightly_perf(self, setup_backend, genai_bench_runner):
        _run_nightly(setup_backend, genai_bench_runner, "deepseek-7b")


# ---------------------------------------------------------------------------
# qwen-30b (tp=4 → single=1, multi=2)
# ---------------------------------------------------------------------------


@_nightly
@_e2e
@pytest.mark.model("qwen-30b")
@pytest.mark.workers(count=1)
@_gateway
@_backends
class TestNightlyQwen30bSingle:
    def test_nightly_perf(self, setup_backend, genai_bench_runner):
        _run_nightly(setup_backend, genai_bench_runner, "qwen-30b")


@_nightly
@_e2e
@pytest.mark.model("qwen-30b")
@pytest.mark.workers(count=2)
@_gateway
@_backends
class TestNightlyQwen30bMulti:
    def test_nightly_perf(self, setup_backend, genai_bench_runner):
        _run_nightly(setup_backend, genai_bench_runner, "qwen-30b")


# ---------------------------------------------------------------------------
# mistral-7b (tp=1 → single=1, multi=8)
# ---------------------------------------------------------------------------


@_nightly
@_e2e
@pytest.mark.model("mistral-7b")
@pytest.mark.workers(count=1)
@_gateway
@_backends
class TestNightlyMistral7bSingle:
    def test_nightly_perf(self, setup_backend, genai_bench_runner):
        _run_nightly(setup_backend, genai_bench_runner, "mistral-7b")


@_nightly
@_e2e
@pytest.mark.model("mistral-7b")
@pytest.mark.workers(count=8)
@_gateway
@_backends
class TestNightlyMistral7bMulti:
    def test_nightly_perf(self, setup_backend, genai_bench_runner):
        _run_nightly(setup_backend, genai_bench_runner, "mistral-7b")


# ---------------------------------------------------------------------------
# embedding (tp=1 → single=1, multi=8)
# ---------------------------------------------------------------------------


@_nightly
@_e2e
@pytest.mark.model("embedding")
@pytest.mark.workers(count=1)
@_gateway
@_backends
class TestNightlyEmbeddingSingle:
    def test_nightly_perf(self, setup_backend, genai_bench_runner):
        _run_nightly(
            setup_backend, genai_bench_runner, "embedding",
            task="text-to-embeddings",
        )


@_nightly
@_e2e
@pytest.mark.model("embedding")
@pytest.mark.workers(count=8)
@_gateway
@_backends
class TestNightlyEmbeddingMulti:
    def test_nightly_perf(self, setup_backend, genai_bench_runner):
        _run_nightly(
            setup_backend, genai_bench_runner, "embedding",
            task="text-to-embeddings",
        )


# ---------------------------------------------------------------------------
# gpt-oss (tp=2 → single=1, multi=4)
# ---------------------------------------------------------------------------


@_nightly
@_e2e
@pytest.mark.model("gpt-oss")
@pytest.mark.workers(count=1)
@_gateway
@_backends
class TestNightlyGptOssSingle:
    def test_nightly_perf(self, setup_backend, genai_bench_runner):
        _run_nightly(setup_backend, genai_bench_runner, "gpt-oss")


@_nightly
@_e2e
@pytest.mark.model("gpt-oss")
@pytest.mark.workers(count=4)
@_gateway
@_backends
class TestNightlyGptOssMulti:
    def test_nightly_perf(self, setup_backend, genai_bench_runner):
        _run_nightly(setup_backend, genai_bench_runner, "gpt-oss")
