#!/usr/bin/env bash
# Compare MLX inference paths on Apple Silicon via genai-bench.
#
# Phases (opt-in via PHASES):
#   mlx   — mlx-lm.server (HTTP, :8001)
#   grpc  — SMG router → MLX gRPC servicer (:30000 → :50051)
#
# Each cell writes one JSON into $RESULTS_DIR/<label>_<scenario>_c<concurrency>/,
# which mlx_bench_aggregate.py reduces into a markdown table.
#
# Usage:
#   ./scripts/mlx_bench.sh                          # both phases
#   PHASES=mlx ./scripts/mlx_bench.sh               # only mlx-lm.server
#   CONCURRENCIES="1 4" ./scripts/mlx_bench.sh      # quick sweep

set -euo pipefail

PHASES="${PHASES:-mlx grpc}"
MODEL="${MODEL:-mlx-community/gemma-3-4b-it-qat-4bit}"
CONCURRENCIES="${CONCURRENCIES:-1 4 16 64}"
SCENARIOS="${SCENARIOS:-chat agent}"
DURATION_MIN="${DURATION_MIN:-5}"
MAX_REQUESTS="${MAX_REQUESTS:-100000}"
RESULTS_DIR="${RESULTS_DIR:-bench-results}"
SMG_BIN="${SMG_BIN:-target/release/smg}"

MLX_PORT="${MLX_PORT:-8001}"
GRPC_PORT="${GRPC_PORT:-50051}"
ROUTER_PORT="${ROUTER_PORT:-30000}"

# chat:  short prompt + medium output (typical chat turn).
# agent: ~2.5k token context — RAG / code-edit traffic where prefill dominates.
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

# Readiness gate for processes that bind a port before warmup finishes.
wait_for_log_line() {
    local file="$1"
    local pattern="$2"
    local timeout="${3:-300}"
    local start=$SECONDS
    while ! grep -q "$pattern" "$file" 2>/dev/null; do
        if (( SECONDS - start > timeout )); then
            log "Timeout waiting for '$pattern' in $file"
            return 1
        fi
        sleep 1
    done
}

# Run one cell. Writes a .failed marker and continues on cell failure.
run_bench_cell() {
    local label="$1"
    local base_url="$2"
    local scenario="$3"
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
    # Port binds before warmup; the listening log line is the real ready signal.
    wait_for_log_line "$RESULTS_DIR/mlx-grpc.log" "gRPC server listening on" 300
    log "MLX gRPC servicer up on :$GRPC_PORT (pid=$grpc_pid)"

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

for phase in $PHASES; do
    case "$phase" in
        mlx)  run_phase_mlx ;;
        grpc) run_phase_grpc ;;
        *)    log "Unknown phase: $phase"; exit 1 ;;
    esac
done

log "Done. Results in $RESULTS_DIR"
log "Aggregate: python3 $(dirname "$0")/mlx_bench_aggregate.py --results-dir $RESULTS_DIR"
