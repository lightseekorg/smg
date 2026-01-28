"""Go bindings OAI server performance benchmark test.

Tests the performance of the Go OAI server which connects directly to gRPC workers.
This benchmark measures the overhead introduced by the Go FFI layer and HTTP server
compared to the Rust gateway.

The Go OAI server uses the same load balancing policies as the Rust gateway
(round_robin, cache_aware, random) to distribute requests across multiple workers.

NOTE: This test requires Go toolchain and FFI library to be available.
It should only be run by the go-bindings-benchmark CI job, not the regular benchmarks job.
"""

import pytest

# Import Go bindings fixtures for this specific test file
pytest_plugins = ["e2e_test.bindings_go.conftest"]


@pytest.mark.e2e
@pytest.mark.workers(count=4)
@pytest.mark.gateway(policy="cache_aware")
@pytest.mark.model("llama-1b")
class TestGoBindingsPerf:
    """Performance benchmark for Go OAI server.

    The Go OAI server provides an OpenAI-compatible HTTP API that uses FFI
    to communicate with the underlying gRPC workers. It uses the gateway's
    load balancing policies to distribute requests across multiple workers.

    This benchmark uses the same configuration as regular gateway benchmarks:
    - 4 gRPC workers
    - cache_aware load balancing policy
    - 32 concurrent requests
    - D(4000,100) traffic scenario

    This allows for direct comparison between Go OAI server and Rust gateway.
    """

    def test_go_oai_server_perf(self, go_oai_server_multi, genai_bench_runner):
        """Run genai-bench against Go OAI server with 4 workers and validate metrics."""
        host, port, model_path = go_oai_server_multi

        genai_bench_runner(
            router_url=f"http://{host}:{port}",
            model_path=model_path,
            experiment_folder="benchmark_go_bindings",
            # Match regular benchmark parameters for fair comparison
            num_concurrency=32,
            traffic_scenario="D(4000,100)",
            max_requests_per_run=160,  # 32 * 5 = 160 (same as regular benchmark default)
            thresholds={
                # Thresholds matching regular benchmark
                # Slightly relaxed due to FFI overhead
                "ttft_mean_max": 8,  # vs 6 for Rust gateway
                "e2e_latency_mean_max": 16,  # vs 14 for Rust gateway
                "input_throughput_mean_min": 600,  # vs 800 for Rust gateway
                "output_throughput_mean_min": 10,  # vs 12 for Rust gateway
                "gpu_util_p50_min": 95,  # Same as Rust gateway
            },
        )
