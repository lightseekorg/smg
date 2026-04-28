#!/usr/bin/env bash
# Three-way MLX inference comparison on Apple Silicon.
#
# Phases (each opt-in via PHASES env var):
#   1. mlx   — mlx-lm.server (HTTP, port 8001)
#   2. grpc  — SMG router → MLX gRPC servicer (router on 30000, servicer on 50051)
#   3. vllm  — vllm-metal `vllm serve` (HTTP, port 8002)
#
# Driver: `genai-bench` (pip-installed, same CLI as the rest of the SMG
# nightly benchmark suite). genai-bench writes one JSON per cell into
# $RESULTS_DIR/<label>_<scenario>_c<concurrency>/, which aggregate.py
# then reduces into a 3-way comparison markdown table.
#
# Usage:
#   ./run.sh                          # all three phases
#   PHASES=mlx ./run.sh               # only mlx-lm.server
#   PHASES="mlx grpc" ./run.sh        # skip vllm-metal
#   CONCURRENCIES="1 4" ./run.sh      # quick local sweep
#
# Tunables (env):
#   PHASES                space-sep subset of "mlx grpc vllm" (default: all three)
#   MODEL                 default mlx-community/gemma-3-4b-it-qat-4bit
#   CONCURRENCIES         default "1 4 16 64"
#   SCENARIOS             space-sep subset of "chat agent" (default: both)
#   DURATION_MIN          minutes per cell (default: 5)
#   MAX_REQUESTS          hard cap per cell (default: 100000, effectively
#                         unbounded; duration is the real limit)
#   RESULTS_DIR           output dir (default: bench-results)
#   SMG_BIN               path to smg binary (default: target/release/smg)
#   VLLM_VENV             path to vllm-metal venv (default: ~/.venv-vllm-metal)
#   MLX_PORT/GRPC_PORT/ROUTER_PORT/VLLM_PORT  port overrides

set -euo pipefail

PHASES="${PHASES:-mlx grpc vllm}"
MODEL="${MODEL:-mlx-community/gemma-3-4b-it-qat-4bit}"
CONCURRENCIES="${CONCURRENCIES:-1 4 16 64}"
SCENARIOS="${SCENARIOS:-chat agent}"
DURATION_MIN="${DURATION_MIN:-5}"
MAX_REQUESTS="${MAX_REQUESTS:-100000}"
RESULTS_DIR="${RESULTS_DIR:-bench-results}"
SMG_BIN="${SMG_BIN:-target/release/smg}"
VLLM_VENV="${VLLM_VENV:-$HOME/.venv-vllm-metal}"

MLX_PORT="${MLX_PORT:-8001}"
GRPC_PORT="${GRPC_PORT:-50051}"
ROUTER_PORT="${ROUTER_PORT:-30000}"
VLLM_PORT="${VLLM_PORT:-8002}"

# Map our friendly scenario names to genai-bench traffic-scenario strings.
# chat:  short prompt + medium output (typical chatbot turn).
# agent: ~2.5k token context + medium output (RAG / code-edit / Cursor-style
#        local agent traffic where prefill bandwidth dominates).
scenario_traffic() {
    case "$1" in
        chat)  echo "D(100,256)" ;;
        agent) echo "D(2500,256)" ;;
        *)     echo "" ;;
    esac
}

mkdir -p "$RESULTS_DIR"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

PIDS=()
cleanup() {
    log "Cleaning up child processes..."
    for pid in "${PIDS[@]:-}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            sleep 1
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    PIDS=()
}
trap cleanup EXIT INT TERM

wait_for_port() {
    local port="$1"
    local timeout="${2:-300}"
    local start=$SECONDS
    while ! nc -z 127.0.0.1 "$port" 2>/dev/null; do
        if (( SECONDS - start > timeout )); then
            log "Timeout waiting for port $port"
            return 1
        fi
        sleep 1
    done
}

wait_for_openai() {
    local base="$1"
    local timeout="${2:-300}"
    local start=$SECONDS
    while ! curl -fsS "$base/v1/models" >/dev/null 2>&1; do
        if (( SECONDS - start > timeout )); then
            log "Timeout waiting for $base/v1/models"
            return 1
        fi
        sleep 2
    done
}

# Run one genai-bench cell against a backend. Tolerates per-cell failure:
# writes a .failed marker and returns 0 so the loop continues.
run_bench_cell() {
    local label="$1"          # mlx | grpc | vllm
    local base_url="$2"
    local scenario="$3"       # chat | agent
    local concurrency="$4"

    local traffic
    traffic="$(scenario_traffic "$scenario")"
    if [ -z "$traffic" ]; then
        log "Unknown scenario: $scenario"
        return 1
    fi

    local exp_name="${label}_${scenario}_c${concurrency}"
    local exp_dir="$RESULTS_DIR/$exp_name"
    mkdir -p "$exp_dir"

    log "[$exp_name] genai-bench scenario=$traffic c=$concurrency duration=${DURATION_MIN}m"

    if ! genai-bench benchmark \
        --api-backend openai \
        --api-base "$base_url" \
        --api-key dummy-token \
        --api-model-name "$MODEL" \
        --model-tokenizer "$MODEL" \
        --task text-to-text \
        --num-concurrency "$concurrency" \
        --traffic-scenario "$traffic" \
        --max-requests-per-run "$MAX_REQUESTS" \
        --max-time-per-run "$DURATION_MIN" \
        --experiment-folder-name "$exp_name" \
        --experiment-base-dir "$RESULTS_DIR"
    then
        log "[$exp_name] FAILED — recording marker, continuing"
        date -u +"%Y-%m-%dT%H:%M:%SZ" >"$exp_dir/.failed"
    fi
}

run_phase_mlx() {
    log "=== Phase: mlx-lm.server (HTTP) ==="

    mlx_lm.server --model "$MODEL" --host 127.0.0.1 --port "$MLX_PORT" \
        >"$RESULTS_DIR/mlx-lm.log" 2>&1 &
    local pid=$!
    PIDS+=("$pid")
    wait_for_openai "http://127.0.0.1:$MLX_PORT" 300
    log "mlx-lm.server up on :$MLX_PORT (pid=$pid)"

    for scenario in $SCENARIOS; do
        for c in $CONCURRENCIES; do
            run_bench_cell "mlx" "http://127.0.0.1:$MLX_PORT" "$scenario" "$c"
        done
    done

    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    PIDS=()
    sleep 3
}

run_phase_grpc() {
    log "=== Phase: SMG router + MLX gRPC servicer ==="

    python3 -m smg_grpc_servicer.mlx.server --model "$MODEL" \
        --host 127.0.0.1 --port "$GRPC_PORT" \
        >"$RESULTS_DIR/mlx-grpc.log" 2>&1 &
    local grpc_pid=$!
    PIDS+=("$grpc_pid")
    wait_for_port "$GRPC_PORT" 300
    log "MLX gRPC servicer up on :$GRPC_PORT (pid=$grpc_pid)"
    sleep 5  # let warmup finish before the router connects

    "$SMG_BIN" launch \
        --host 127.0.0.1 --port "$ROUTER_PORT" \
        --worker-urls "grpc://127.0.0.1:$GRPC_PORT" \
        >"$RESULTS_DIR/smg-router.log" 2>&1 &
    local router_pid=$!
    PIDS+=("$router_pid")
    wait_for_openai "http://127.0.0.1:$ROUTER_PORT" 60
    log "SMG router up on :$ROUTER_PORT (pid=$router_pid)"

    for scenario in $SCENARIOS; do
        for c in $CONCURRENCIES; do
            run_bench_cell "grpc" "http://127.0.0.1:$ROUTER_PORT" "$scenario" "$c"
        done
    done

    kill "$router_pid" "$grpc_pid" 2>/dev/null || true
    wait "$router_pid" 2>/dev/null || true
    wait "$grpc_pid" 2>/dev/null || true
    PIDS=()
    sleep 3
}

run_phase_vllm() {
    log "=== Phase: vllm-metal serve ==="

    if [ ! -f "$VLLM_VENV/bin/activate" ]; then
        log "vllm-metal venv not found at $VLLM_VENV — skipping phase"
        return 0
    fi

    # shellcheck disable=SC1091
    source "$VLLM_VENV/bin/activate"
    # Cap context: agent prompts are ~2.5k tokens + 256 output. 8192 is
    # plenty and avoids vllm-metal's max-position-embeddings-derived budget
    # (which can be huge for Gemma).
    vllm serve "$MODEL" --host 127.0.0.1 --port "$VLLM_PORT" \
        --max-model-len 8192 \
        >"$RESULTS_DIR/vllm-metal.log" 2>&1 &
    local pid=$!
    PIDS+=("$pid")
    deactivate || true

    wait_for_openai "http://127.0.0.1:$VLLM_PORT" 600
    log "vllm-metal up on :$VLLM_PORT (pid=$pid)"

    for scenario in $SCENARIOS; do
        for c in $CONCURRENCIES; do
            run_bench_cell "vllm" "http://127.0.0.1:$VLLM_PORT" "$scenario" "$c"
        done
    done

    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    PIDS=()
    sleep 3
}

for phase in $PHASES; do
    case "$phase" in
        mlx)  run_phase_mlx ;;
        grpc) run_phase_grpc ;;
        vllm) run_phase_vllm ;;
        *)    log "Unknown phase: $phase"; exit 1 ;;
    esac
done

log "Done. Results in $RESULTS_DIR"
log "Run aggregator: python3 $(dirname "$0")/aggregate.py --results-dir $RESULTS_DIR"
