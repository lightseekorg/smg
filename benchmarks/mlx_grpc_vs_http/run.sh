#!/usr/bin/env bash
# MLX direct-HTTP vs router+gRPC benchmark.
#
# Drives a custom OpenAI-compatible bench client (bench_client.py) against
# both targets:
#   - Phase 1: mlx-lm.server (direct HTTP)
#   - Phase 2: SMG router → MLX gRPC servicer
#
# Two real-world prompt scenarios:
#   - chat:  ShareGPT — typical chat traffic, short prompts
#   - agent: vdaita/edit_10k_char — local coding-agent traffic with
#            ~2.5k token code-context prompts (Cursor/Continue style)
#
# Phases can run independently via PHASE env (default "all"). The CI workflow
# splits them into separate steps so a phase-1 failure doesn't block phase 2.
#
# Per-cell failures inside a phase are tolerated — bench_client returns
# non-zero only if every request failed; we capture that as a .failed marker
# and continue. aggregate.py treats missing JSON as "—".
#
# Tunables via env:
#   PHASE                http | grpc | all (default: all)
#   MODEL                model id (default: mlx-community/gemma-3-4b-it-qat-4bit)
#   CONCURRENCIES        space-separated (default: "1 4 16 64")
#   SCENARIOS            space-separated (default: "chat agent")
#   RESULTS_DIR          output dir (default: bench-results)
#   HTTP_PORT            mlx-lm http port (default: 8001)
#   GRPC_PORT            mlx grpc servicer port (default: 50051)
#   ROUTER_PORT          smg router port (default: 30000)
#   SMG_BIN              path to smg binary (default: target/release/smg)

set -euo pipefail

PHASE="${PHASE:-all}"
MODEL="${MODEL:-mlx-community/gemma-3-4b-it-qat-4bit}"
CONCURRENCIES="${CONCURRENCIES:-1 4 16 64}"
SCENARIOS="${SCENARIOS:-chat agent}"
RESULTS_DIR="${RESULTS_DIR:-bench-results}"
HTTP_PORT="${HTTP_PORT:-8001}"
GRPC_PORT="${GRPC_PORT:-50051}"
ROUTER_PORT="${ROUTER_PORT:-30000}"
SMG_BIN="${SMG_BIN:-target/release/smg}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_PY="$SCRIPT_DIR/bench_client.py"

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
    local timeout="${2:-180}"
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
    local timeout="${2:-180}"
    local start=$SECONDS
    while ! curl -fsS "$base/v1/models" >/dev/null 2>&1; do
        if (( SECONDS - start > timeout )); then
            log "Timeout waiting for $base/v1/models"
            return 1
        fi
        sleep 2
    done
}

# Number of prompts per cell — scales with concurrency so each cell has
# a meaningful number of completed requests at every level. Chat is fast
# (short prompts → short prefill), agent is slow (2.5k token prefill
# dominates), so agent uses fewer prompts to keep total runtime reasonable.
prompts_for() {
    local scenario="$1"
    local concurrency="$2"
    case "$scenario" in
        chat)  echo $(( concurrency < 16 ? 50 : concurrency * 4 )) ;;
        agent) echo $(( concurrency < 16 ? 30 : concurrency * 2 )) ;;
        *)     echo 50 ;;
    esac
}

run_bench() {
    local label="$1"           # http or grpc
    local base_url="$2"
    local scenario="$3"
    local concurrency="$4"

    local exp_name="${label}_${scenario}_c${concurrency}"
    local exp_dir="$RESULTS_DIR/$exp_name"
    local out_json="$exp_dir/result.json"
    mkdir -p "$exp_dir"

    local num_prompts
    num_prompts="$(prompts_for "$scenario" "$concurrency")"

    log "[$exp_name] scenario=$scenario concurrency=$concurrency num_prompts=$num_prompts"

    if ! python3 "$BENCH_PY" \
        --base-url "$base_url" \
        --model "$MODEL" \
        --label "$label" \
        --scenario "$scenario" \
        --concurrency "$concurrency" \
        --num-prompts "$num_prompts" \
        --out "$out_json"
    then
        log "[$exp_name] FAILED — recording marker, continuing"
        date -u +"%Y-%m-%dT%H:%M:%SZ" >"$exp_dir/.failed"
        return 0
    fi
}

run_phase_http() {
    log "=== Phase 1: direct HTTP (mlx-lm.server) ==="

    mlx_lm.server --model "$MODEL" --host 127.0.0.1 --port "$HTTP_PORT" \
        >"$RESULTS_DIR/mlx-lm.log" 2>&1 &
    local mlx_pid=$!
    PIDS+=("$mlx_pid")
    wait_for_openai "http://127.0.0.1:$HTTP_PORT" 300
    log "mlx-lm.server up on :$HTTP_PORT (pid=$mlx_pid)"

    for scenario in $SCENARIOS; do
        for c in $CONCURRENCIES; do
            run_bench "http" "http://127.0.0.1:$HTTP_PORT" "$scenario" "$c"
        done
    done

    log "Stopping mlx-lm.server..."
    kill "$mlx_pid" 2>/dev/null || true
    wait "$mlx_pid" 2>/dev/null || true
    PIDS=()
}

run_phase_grpc() {
    log "=== Phase 2: SMG router + MLX gRPC servicer ==="

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
            run_bench "grpc" "http://127.0.0.1:$ROUTER_PORT" "$scenario" "$c"
        done
    done

    log "Stopping gRPC servers..."
    kill "$router_pid" 2>/dev/null || true
    kill "$grpc_pid" 2>/dev/null || true
    wait "$router_pid" 2>/dev/null || true
    wait "$grpc_pid" 2>/dev/null || true
    PIDS=()
}

case "$PHASE" in
    http) run_phase_http ;;
    grpc) run_phase_grpc ;;
    all)  run_phase_http; run_phase_grpc ;;
    *)    log "Unknown PHASE=$PHASE (expected http|grpc|all)"; exit 1 ;;
esac

log "Done. Results in $RESULTS_DIR"
