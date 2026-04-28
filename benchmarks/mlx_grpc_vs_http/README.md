# MLX Direct HTTP vs Router + gRPC Benchmark

Compares throughput and latency of two paths into the same MLX model:

1. **Direct HTTP** — `mlx_lm.server` exposing the OpenAI-compatible API
2. **Router + gRPC** — SMG router fronting the SMG MLX gRPC servicer

Both paths land at the same MLX `BatchGenerator`; the difference is the
transport, where tokenization happens (Python vs Rust), and what the
router does on top (chat templating, JSON ↔ proto conversion).

## What it measures

For each (scenario, concurrency) cell:
- Request throughput (req/s)
- Output token throughput (tokens/s)
- TTFT mean / p50 / p95 / p99 (ms)
- TPOT mean / p50 / p95 / p99 (ms)
- Errors / total requests

## Scenarios

Real prompt distributions (no synthetic D(in,out) lengths):

| Scenario | Dataset | Why |
|---|---|---|
| `chat` | `anon8231489123/ShareGPT_Vicuna_unfiltered` | Typical chat traffic — short user prompts, exercises TTFT and decode throughput. |
| `agent` | `vdaita/edit_10k_char` | Local coding-agent workload — ~10k character (~2.5k token) code-context prompts. Models the realistic Mac use case (Cursor/Continue/Cline editing files). Stresses prefill bandwidth and tokenization cost — where Rust router should win big over Python. |

## Why a custom bench client (not vllm bench or genai-bench)

- **vllm bench** requires `pip install vllm` which is Linux/CUDA-only on PyPI. No macOS arm64 wheels. Can't install on the runner we need.
- **genai-bench** only supports synthetic length distributions (`D(in,out)`), not real datasets. ShareGPT and InstructCoder-style workloads are exactly the comparison the user-facing pitch needs.
- **`bench_client.py`** is ~250 LOC of asyncio + httpx + HF datasets. Output JSON is shaped like `vllm bench --save-result` so we can switch to vllm bench later if mac support arrives.

## Running locally

Apple Silicon Mac:

```bash
# Build smg in release mode
cargo build --release --bin smg

# Install Python deps
pip install -e ./crates/grpc_client/python
pip install -e "./grpc_servicer[mlx]"
pip install "mlx-lm>=0.22.0"
pip install "httpx[http2]>=0.27" "datasets>=3.0"

# Full run (~30 min)
./benchmarks/mlx_grpc_vs_http/run.sh

# Build the summary table
python3 benchmarks/mlx_grpc_vs_http/aggregate.py
```

For a quick local smoke test:

```bash
CONCURRENCIES="1" SCENARIOS="chat" \
  ./benchmarks/mlx_grpc_vs_http/run.sh
```

## Tunables (env vars)

| Var | Default | Notes |
|---|---|---|
| `PHASE` | `all` | `http`, `grpc`, or `all` |
| `MODEL` | `mlx-community/gemma-3-4b-it-qat-4bit` | ~3 GB |
| `CONCURRENCIES` | `1 4 16 64` | space-separated |
| `SCENARIOS` | `chat agent` | space-separated |
| `RESULTS_DIR` | `bench-results` | output dir |

Per-cell prompt counts scale with concurrency (defined in `run.sh:prompts_for`):
- `chat`: 50 prompts at low conc, `4×concurrency` at high conc
- `agent`: 30 prompts at low conc, `2×concurrency` at high conc (longer prefill → slower)

## Robustness

- **Per-cell failures tolerated**: if a server falls over (e.g. mlx-lm.server at c=64 returns all 503s), the cell's bench client exits non-zero, run.sh writes a `.failed` marker, and continues to the next cell.
- **Phase-level isolation**: workflow runs phases as two separate steps with `if: always()`. Phase 2 (gRPC) runs even if phase 1 (HTTP) crashed entirely.

## CI

`.github/workflows/nightly-mlx-bench.yml` runs this on `macos-latest` (Apple Silicon) on a nightly schedule and via manual trigger (`workflow_dispatch`) with overridable inputs.
