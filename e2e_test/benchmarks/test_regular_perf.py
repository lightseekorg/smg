"""Regular router performance benchmark test."""

import pytest


@pytest.mark.e2e
@pytest.mark.workers(count=4)
@pytest.mark.gateway(policy="round_robin")
@pytest.mark.parametrize("setup_backend", ["http", "grpc"], indirect=True)
class TestRegularPerf:
    """Performance benchmark for regular (non-PD) router."""

    def test_regular_perf(self, setup_backend, genai_bench_runner):
        """Run genai-bench against regular router and validate metrics."""
        backend, model_path, client, gateway = setup_backend
        genai_bench_runner(
            router_url=gateway.base_url,
            model_path=model_path,
            experiment_folder=f"benchmark_round_robin_regular_{backend}",
            # Increase max_requests to ensure benchmark runs long enough for
            # accurate GPU utilization sampling (at least 30+ seconds)
            max_requests_per_run=200,
            thresholds={
                "ttft_mean_max": 0.8,
                "e2e_latency_mean_max": 14,
                "input_throughput_mean_min": 800,
                "output_throughput_mean_min": 12,
                "gpu_util_mean_min": 30,
            },
        )
