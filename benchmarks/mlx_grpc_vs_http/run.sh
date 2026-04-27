#!/usr/bin/env bash
# MLX direct-HTTP vs router+gRPC benchmark.
#
# Spins up mlx-lm's HTTP server, runs genai-bench against it, then spins up
# the SMG MLX servicer + SMG router and runs the same benchmark matrix.
# Results land in $RESULTS_DIR for aggregate.py to summarize.
#
# Tunables via env (overridable from the workflow):
#   MODEL                model id (default: mlx-community/gemma-3-4b-it-qat-4bit)
#   CONCURRENCIES        space-separated (default: "1 4 16 64")
#   SCENARIOS            space-separated genai-bench scenarios
#                        (default: "D(100,256) D(2000,128)")
#   DURATION_MIN         minutes per cell (default: 10)
#   MAX_REQUESTS         hard cap per cell (default: 100000, effectively
#                        unbounded; duration is the real limit)
#   RESULTS_DIR          output dir (default: bench-results)
#   GENAI_BENCH_IMAGE    docker image (default: ghcr.io/moirai-internal/
#                        genai-bench:0.0.4)
#   HTTP_PORT            mlx-lm http port (default: 8001)
#   GRPC_PORT            mlx grpc servicer port (default: 50051)
#   ROUTER_PORT          smg router port (default: 30000)
#   SMG_BIN              path to smg binary (default: target/release/smg)

set -euo pipefail

MODEL="${MODEL:-mlx-community/gemma-3-4b-it-qat-4bit}"
CONCURRENCIES="${CONCURRENCIES:-1 4 16 64}"
SCENARIOS="${SCENARIOS:-D(100,256) D(2000,128)}"
DURATION_MIN="${DURATION_MIN:-10}"
MAX_REQUESTS="${MAX_REQUESTS:-100000}"
RESULTS_DIR="${RESULTS_DIR:-bench-results}"
GENAI_BENCH_IMAGE="${GENAI_BENCH_IMAGE:-ghcr.io/moirai-internal/genai-bench:0.0.4}"
HTTP_PORT="${HTTP_PORT:-8001}"
GRPC_PORT="${GRPC_PORT:-50051}"
ROUTER_PORT="${ROUTER_PORT:-30000}"
SMG_BIN="${SMG_BIN:-target/release/smg}"

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

# OpenAI-compatible endpoint health check (waits for /v1/models).
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

run_bench() {
    local label="$1"          # "http" or "grpc"
    local base_url="$2"       # e.g. http://127.0.0.1:8001
    local scenario="$3"       # e.g. D(100,256)
    local concurrency="$4"

    # Sanitize for filenames: D(100,256) -> D-100-256
    local scenario_slug
    scenario_slug="$(echo "$scenario" | tr '(),' '-' | tr -s '-' | sed 's/-$//')"
    local exp_name="${label}_${scenario_slug}_c${concurrency}"
    local exp_dir="$RESULTS_DIR/$exp_name"
    mkdir -p "$exp_dir"

    log "[$exp_name] genai-bench scenario=$scenario concurrency=$concurrency duration=${DURATION_MIN}m"

    # Mount HF cache so the bench reuses tokenizers we already have on disk.
    local hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
    local hf_mount=()
    if [ -d "$hf_home" ]; then
        hf_mount+=( -v "$hf_home:$hf_home" -e "HF_HOME=$hf_home" )
    fi

    # genai-bench reaches the host via host.docker.internal because Docker
    # for Mac (and Colima) run a Linux VM — `--network host` would attach
    # to the VM's network, not the macOS host's. The base URL is rewritten
    # accordingly: 127.0.0.1 → host.docker.internal.
    local container_url="${base_url/127.0.0.1/host.docker.internal}"
    container_url="${container_url/localhost/host.docker.internal}"

    # genai-bench's --max-time-per-run is in SECONDS (despite the name).
    local duration_s=$(( DURATION_MIN * 60 ))

    docker run --rm \
        --add-host=host.docker.internal:host-gateway \
        -v "$(pwd):$(pwd)" \
        -w "$(pwd)" \
        "${hf_mount[@]}" \
        "$GENAI_BENCH_IMAGE" \
        benchmark \
        --api-backend openai \
        --api-base "$container_url" \
        --api-key dummy-token \
        --api-model-name "$MODEL" \
        --model-tokenizer "$MODEL" \
        --task text-to-text \
        --num-concurrency "$concurrency" \
        --traffic-scenario "$scenario" \
        --max-requests-per-run "$MAX_REQUESTS" \
        --max-time-per-run "$duration_s" \
        --experiment-folder-name "$exp_name" \
        --experiment-base-dir "$RESULTS_DIR"
}

# ──────────────────────────────────────────────────────────────────────────
# Phase 1: direct mlx-lm HTTP server
# ──────────────────────────────────────────────────────────────────────────
log "=== Phase 1: direct HTTP (mlx-lm.server) ==="

mlx_lm.server --model "$MODEL" --host 127.0.0.1 --port "$HTTP_PORT" \
    >"$RESULTS_DIR/mlx-lm.log" 2>&1 &
MLX_HTTP_PID=$!
PIDS+=("$MLX_HTTP_PID")
wait_for_openai "http://127.0.0.1:$HTTP_PORT" 300
log "mlx-lm.server up on :$HTTP_PORT (pid=$MLX_HTTP_PID)"

for scenario in $SCENARIOS; do
    for c in $CONCURRENCIES; do
        run_bench "http" "http://127.0.0.1:$HTTP_PORT" "$scenario" "$c"
    done
done

log "Stopping mlx-lm.server..."
kill "$MLX_HTTP_PID" 2>/dev/null || true
wait "$MLX_HTTP_PID" 2>/dev/null || true
# Rebuild PIDS without the stopped pid (pattern substitution with /
# would leave an empty-string ghost entry, not actually remove it).
new_pids=()
for pid in "${PIDS[@]}"; do
    [ "$pid" = "$MLX_HTTP_PID" ] || new_pids+=("$pid")
done
PIDS=("${new_pids[@]}")

# ──────────────────────────────────────────────────────────────────────────
# Phase 2: SMG router + MLX gRPC servicer
# ──────────────────────────────────────────────────────────────────────────
log "=== Phase 2: SMG router + MLX gRPC servicer ==="

python3 -m smg_grpc_servicer.mlx.server --model "$MODEL" \
    --host 127.0.0.1 --port "$GRPC_PORT" \
    >"$RESULTS_DIR/mlx-grpc.log" 2>&1 &
GRPC_PID=$!
PIDS+=("$GRPC_PID")
wait_for_port "$GRPC_PORT" 300
log "MLX gRPC servicer up on :$GRPC_PORT (pid=$GRPC_PID)"
# Give the gen thread a beat to finish warmup before starting the router.
sleep 5

"$SMG_BIN" launch \
    --host 127.0.0.1 --port "$ROUTER_PORT" \
    --worker-urls "grpc://127.0.0.1:$GRPC_PORT" \
    >"$RESULTS_DIR/smg-router.log" 2>&1 &
ROUTER_PID=$!
PIDS+=("$ROUTER_PID")
wait_for_openai "http://127.0.0.1:$ROUTER_PORT" 60
log "SMG router up on :$ROUTER_PORT (pid=$ROUTER_PID)"

for scenario in $SCENARIOS; do
    for c in $CONCURRENCIES; do
        run_bench "grpc" "http://127.0.0.1:$ROUTER_PORT" "$scenario" "$c"
    done
done

log "Done. Results in $RESULTS_DIR"
