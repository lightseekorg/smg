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

## Reading the results

Cells render four meaningful columns (RPS, output tok/s, TTFT, TPOT)
across all three backends.

**High-concurrency agent cells may be empty (`—`)** if the
`DURATION_MIN` budget runs out before any request completes a 2.5k
token prefill. Bump `DURATION_MIN` (or drop to lower concurrency) if
you need numbers there.

### Why we patch genai-bench

`mlx-lm.server` is built on Python's `BaseHTTPRequestHandler`, which
uses HTTP/1.0 by default and emits no `Content-Length` or
`Transfer-Encoding: chunked` for streaming responses. Wire-level probes
confirm mlx-lm DOES stream per-token chunks at ~18 ms intervals — the
issue is purely client-side: genai-bench calls
`response.iter_lines(chunk_size=None)` (in `genai_bench/user/openai_user.py`),
which under the hood does `urllib3.HTTPResponse.read(amt=None)` and
buffers to EOF when the response has neither `Content-Length` nor a
`chunked` envelope. Result without the patch: every per-token chunk
arrives at the bench client in one bulk read at end-of-stream, which
collapses measured TPOT to ~0 and inflates TTFT to e2e latency.

The `Patch genai-bench streaming chunk size` workflow step changes
`chunk_size=None` to `chunk_size=1`, which makes urllib3 yield per OS
read — restoring correct per-token timing. The patch is a no-op for
the SMG router (Rust hyper, HTTP/1.1 + chunked) and vllm-metal (uvicorn,
HTTP/1.1 + chunked); both stream correctly with either chunk_size.

When running locally, apply the same patch by hand once:
```bash
python3 -c "import genai_bench, os; print(os.path.dirname(genai_bench.__file__) + '/user/openai_user.py')" \
  | xargs sed -i.bak 's/response.iter_lines(chunk_size=None)/response.iter_lines(chunk_size=1)/g'
```

## Robustness

- **Per-cell failures tolerated**: if genai-bench fails for one cell,
  `run.sh` writes a `.failed` marker and continues. `aggregate.py` renders
  missing cells as `—`.
- **Phase-level isolation**: the workflow runs each phase as its own step
  with `if: always()`, so a crash in one backend doesn't block the others.
- **vllm-metal optional**: if `$VLLM_VENV` doesn't exist, the vllm phase
  is skipped with a log message rather than failing the whole run.

## CI vs local model defaults

| Environment | `MODEL` default | Why |
|---|---|---|
| GitHub Actions (`nightly-mlx-bench.yml`) | `mlx-community/gemma-3-1b-it-qat-4bit` (~0.7 GB) | `macos-latest` runners expose ~5 GB Metal memory total. A 4 B-class model (3 GB) plus vllm-metal's ~1.7 GB overhead leaves a negative KV-cache budget — lowering `--gpu-memory-utilization` makes that worse, not better, since the flag gates how much Metal vllm-metal can use. **QAT format also matters**: vllm-metal's Metal path only picks up `*-qat-4bit` checkpoints; regular 4-bit MLX checkpoints silently fall back to a PyTorch CPU/fp32 path that needs ~6 GB and OOMs at startup. |
| Local Mac (`run.sh`) | `mlx-community/gemma-3-4b-it-qat-4bit` (~3 GB) | M-series Pro/Max with 16+ GB unified memory has plenty of headroom; the larger model produces more representative absolute throughput / latency numbers. |

The bench compares all three backends on the *same* model — switching
to the 1 B variant for CI keeps the **relative** router/streaming
comparison meaningful (TTFT shape, RPS scaling with concurrency, TPOT
correctness). For production-shaped *absolute* numbers at 4 B or
larger, use the local `run.sh` defaults or override `MODEL` via
`workflow_dispatch` on a runner with more RAM (e.g. self-hosted M-Pro,
or `macos-latest-xlarge` with 14 GB).

## CI

`.github/workflows/nightly-mlx-bench.yml` runs this on `macos-latest`
(Apple Silicon) on a nightly schedule and via manual `workflow_dispatch`
with overridable inputs.
