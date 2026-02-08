"""Go bindings OAI server performance benchmark test.

Tests the performance of the Go OAI server which connects directly to gRPC workers.
This benchmark uses the exact same configuration as the regular gateway benchmark
for a fair comparison.

NOTE: This test requires Go toolchain and FFI library to be available.
It should only be run by the go-bindings-benchmark CI job, not the regular benchmarks job.
"""

import pytest

# Import Go bindings fixtures for this specific test file
pytest_plugins = ["e2e_test.bindings_go.conftest"]


@pytest.mark.e2e
@pytest.mark.workers(count=4)
@pytest.mark.gateway(policy="round_robin")
@pytest.mark.model("meta-llama/Llama-3.2-1B-Instruct")
class TestGoBindingsPerf:
    """Performance benchmark for Go OAI server.

    Uses the exact same configuration as TestRegularPerf:
    - 4 workers
    - round_robin policy
    - 32 concurrency, D(4000,100) traffic scenario
    - Same thresholds and default max_requests (160)
    """

    def test_go_oai_server_perf(self, go_oai_server_multi, genai_bench_runner):
        """Run genai-bench against Go OAI server with 4 workers and validate metrics."""
        host, port, model_path = go_oai_server_multi

        genai_bench_runner(
            router_url=f"http://{host}:{port}",
            model_path=model_path,
            experiment_folder="benchmark_go_bindings",
            thresholds={
                # Same thresholds as regular benchmark
                "ttft_mean_max": 6,
                "e2e_latency_mean_max": 14,
                "input_throughput_mean_min": 800,
                "output_throughput_mean_min": 12,
                "gpu_util_p50_min": 50,
            },
        )
