# MLX Direct HTTP vs Router + gRPC Benchmark

Compares throughput and latency of two paths into the same model:

1. **Direct HTTP** — `mlx_lm.server` exposing the OpenAI-compatible API
2. **Router + gRPC** — SMG router fronting the SMG MLX gRPC servicer

Both paths land at the same MLX inference engine; the difference is the
transport and the work the router does (Rust tokenization, chat
templating, JSON ↔ proto conversion).

## What it measures

For each (scenario, concurrency) cell:

- Request throughput (req/s)
- TTFT mean and p90 (ms)
- End-to-end latency mean (ms)
- Output token throughput (tokens/s)

Defaults: 4 concurrencies × 2 scenarios × 2 setups × 10 min/cell ≈ 2.7
hours total.

## Running locally

Apple Silicon Mac with Docker running:

```bash
# Build smg in release mode (much faster than the ci profile)
cargo build --release --bin smg

# Install Python deps
pip install -e ./crates/grpc_client/python
pip install -e "./grpc_servicer[mlx]"

# Pull the bench image
docker pull ghcr.io/moirai-internal/genai-bench:0.0.4

# Run (takes ~2.7 hours by default)
./benchmarks/mlx_grpc_vs_http/run.sh

# Build the summary table
python3 benchmarks/mlx_grpc_vs_http/aggregate.py
```

## Tunables

All via env vars (see `run.sh`):

| Var | Default | Notes |
|---|---|---|
| `MODEL` | `mlx-community/gemma-3-4b-it-qat-4bit` | ~3 GB, fits 16 GB Mac |
| `CONCURRENCIES` | `1 4 16 64` | space-separated |
| `SCENARIOS` | `D(100,256) D(2000,128)` | genai-bench `D(in,out)` |
| `DURATION_MIN` | `10` | minutes per cell |

## CI

`.github/workflows/nightly-mlx-bench.yml` runs this on `macos-latest`
on a nightly schedule and via manual trigger (`workflow_dispatch`) with
overridable inputs.

## Why this benchmark exists

There's a real question of how much overhead the SMG router adds vs
talking to mlx-lm's own HTTP server directly. Three pieces of upstream
work the router does in Rust (tokenization, chat templating, JSON
parsing) move the heavy CPU off Python's hot path; gRPC + protobuf is
cheaper than HTTP/JSON for streaming numeric payloads. The expectation:

- **Single user**: roughly comparable, router slightly ahead because of
  Rust tokenization.
- **High concurrency**: router noticeably ahead because mlx-lm's
  thread-per-request HTTP server hits Python GIL contention.
- **Long inputs**: router more ahead because tokenization cost grows
  linearly with prompt size.

The benchmark validates or falsifies these expectations.
