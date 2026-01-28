"""Go bindings OAI server performance benchmark test.

Tests the performance of the Go OAI server which connects directly to a gRPC worker.
This benchmark measures the overhead introduced by the Go FFI layer and HTTP server
compared to direct gRPC communication.

NOTE: This test requires Go toolchain and FFI library to be available.
It should only be run by the go-bindings-benchmark CI job, not the regular benchmarks job.
"""

import pytest

# Import Go bindings fixtures for this specific test file
pytest_plugins = ["e2e_test.bindings_go.conftest"]


@pytest.mark.e2e
@pytest.mark.workers(count=4)
@pytest.mark.model("llama-1b")
class TestGoBindingsPerf:
    """Performance benchmark for Go OAI server.

    The Go OAI server provides an OpenAI-compatible HTTP API that uses FFI
    to communicate with the underlying gRPC worker. This benchmark measures:
    - Time to first token (TTFT)
    - End-to-end latency
    - Input/output throughput
    - GPU utilization

    Thresholds are slightly relaxed compared to direct gRPC/HTTP backends
    to account for FFI and HTTP layer overhead.
    """

    def test_go_oai_server_perf(self, go_oai_server, genai_bench_runner):
        """Run genai-bench against Go OAI server and validate metrics."""
        host, port, model_path = go_oai_server

        genai_bench_runner(
            router_url=f"http://{host}:{port}/v1",
            model_path=model_path,
            experiment_folder="benchmark_go_bindings",
            thresholds={
                # Slightly relaxed thresholds compared to direct gRPC (6s, 14s, 800, 12, 99)
                # to account for FFI overhead and HTTP layer
                "ttft_mean_max": 8,
                "e2e_latency_mean_max": 16,
                "input_throughput_mean_min": 600,
                "output_throughput_mean_min": 10,
                "gpu_util_p50_min": 95,
            },
        )
