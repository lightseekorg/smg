# MLX Three-Way Benchmark

Compares throughput and latency of three ways to serve the same MLX model
on Apple Silicon:

1. **`mlx-lm.server`** — direct HTTP via the `mlx-lm` package
2. **`smg → mlx-grpc`** — SMG router fronting the SMG MLX gRPC servicer (PR #1099)
3. **`vllm-metal`** — `vllm-project/vllm-metal` `vllm serve`

Driver: [`genai-bench`](https://github.com/moirai-internal/genai-bench),
the same harness the rest of the SMG nightly benchmark suite uses.

## What it measures

For each (scenario, concurrency, backend) cell:

- Requests per second
- Output token throughput (tok/s)
- TTFT mean / p99 (ms)
- TPOT mean (ms)
- Completed request count

## Scenarios

Synthetic deterministic token shapes — content sampled from genai-bench's
built-in `sonnet.txt` to match each shape:

| Scenario | Shape | Why |
|---|---|---|
| `chat` | `D(100,256)` | Short prompt + medium output. Typical chat turn — exercises TTFT. |
| `agent` | `D(2500,256)` | ~2.5k token context + medium output. Models RAG / code-edit / Cursor-style local agent traffic — stresses prefill bandwidth. |

## Running locally

Apple Silicon Mac. Build smg, install Python deps, install vllm-metal in
its own venv, then run.

```bash
# Build smg
cargo build --release --bin smg

# Python deps in the main env
pip install -e ./crates/grpc_client/python
pip install -e "./grpc_servicer[mlx]"
pip install "mlx-lm>=0.22.0" genai-bench

# vllm-metal (separate venv to avoid dependency conflicts)
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash

# Full sweep (~1–3 hours depending on matrix)
./benchmarks/mlx_grpc_vs_http/run.sh

# Build the summary table
python3 benchmarks/mlx_grpc_vs_http/aggregate.py --results-dir bench-results
```

Quick local smoke test (~5 min):

```bash
PHASES="mlx grpc" CONCURRENCIES="1" SCENARIOS="chat" DURATION_MIN=1 \
  ./benchmarks/mlx_grpc_vs_http/run.sh
```

## Tunables (env vars)

| Var | Default | Notes |
|---|---|---|
| `PHASES` | `mlx grpc vllm` | space-separated subset |
| `MODEL` | `mlx-community/gemma-3-4b-it-qat-4bit` | ~3 GB |
| `CONCURRENCIES` | `1 4 16 64` | space-separated |
| `SCENARIOS` | `chat agent` | space-separated |
| `DURATION_MIN` | `5` | minutes per cell |
| `MAX_REQUESTS` | `100000` | hard cap (duration is the real limit) |
| `RESULTS_DIR` | `bench-results` | output dir |
| `VLLM_VENV` | `~/.venv-vllm-metal` | vllm-metal venv path |

## Robustness

- **Per-cell failures tolerated**: if genai-bench fails for one cell,
  `run.sh` writes a `.failed` marker and continues. `aggregate.py` renders
  missing cells as `—`.
- **Phase-level isolation**: the workflow runs each phase as its own step
  with `if: always()`, so a crash in one backend doesn't block the others.
- **vllm-metal optional**: if `$VLLM_VENV` doesn't exist, the vllm phase
  is skipped with a log message rather than failing the whole run.

## CI

`.github/workflows/nightly-mlx-bench.yml` runs this on `macos-latest`
(Apple Silicon) on a nightly schedule and via manual `workflow_dispatch`
with overridable inputs.
