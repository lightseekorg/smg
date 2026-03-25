#!/bin/bash
# Local mesh profiling test
#
# Spins up mock workers + SMG gateway replicas with mesh enabled.
# Then use the load generator to build up the radix tree and observe
# mesh sync behavior under load.
#
# Prerequisites:
#   cargo build --release -p smg
#   pip install aiohttp  (for load generator)
#
# Usage:
#   ./scripts/mesh-profile-local.sh start       # Start workers + gateways
#   ./scripts/mesh-profile-local.sh load         # Run load generator (200 req/s, 60s)
#   ./scripts/mesh-profile-local.sh status       # Show health + mesh metrics
#   ./scripts/mesh-profile-local.sh stop         # Stop everything
#
# Typical flow:
#   1. ./scripts/mesh-profile-local.sh start
#   2. ./scripts/mesh-profile-local.sh load
#   3. # In another terminal: tail -f target/mesh-profile/gateway-0.log | grep "mesh sync"
#   4. # Or: curl http://127.0.0.1:29000/metrics | grep router_mesh
#   5. ./scripts/mesh-profile-local.sh stop

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="${REPO_ROOT}/target/mesh-profile"
# Resolve cargo target directory (may be a shared location)
CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$(cargo metadata --no-deps --format-version 1 2>/dev/null | python3 -c 'import sys,json; print(json.load(sys.stdin)["target_directory"])' 2>/dev/null || echo "${REPO_ROOT}/target")}"
SMG_BIN="${CARGO_TARGET_DIR}/release/smg"
NUM_WORKERS=20
NUM_GATEWAYS=3
WORKER_BASE_PORT=9000
GATEWAY_BASE_PORT=30000
MESH_BASE_PORT=39500
METRICS_BASE_PORT=29000

mkdir -p "$LOG_DIR"

start_mock_workers() {
    : > "$LOG_DIR/worker_pids.txt"
    echo "Starting $NUM_WORKERS mock workers..."
    for i in $(seq 0 $((NUM_WORKERS - 1))); do
        port=$((WORKER_BASE_PORT + i))
        python3 "$SCRIPT_DIR/mock_worker.py" "$port" &
        echo $! >> "$LOG_DIR/worker_pids.txt"
    done
    echo "Mock workers started on ports $WORKER_BASE_PORT-$((WORKER_BASE_PORT + NUM_WORKERS - 1))"
}

start_gateways() {
    if [ ! -f "$SMG_BIN" ]; then
        echo "ERROR: $SMG_BIN not found. Run: cargo build --release -p smg"
        exit 1
    fi

    : > "$LOG_DIR/gateway_pids.txt"
    echo "Starting $NUM_GATEWAYS SMG gateway replicas with mesh..."

    for i in $(seq 0 $((NUM_GATEWAYS - 1))); do
        port=$((GATEWAY_BASE_PORT + i))
        mesh_port=$((MESH_BASE_PORT + i))
        metrics_port=$((METRICS_BASE_PORT + i))

        # Build mesh peer list (all others)
        MESH_PEERS=""
        for j in $(seq 0 $((NUM_GATEWAYS - 1))); do
            if [ "$i" != "$j" ]; then
                MESH_PEERS="$MESH_PEERS 127.0.0.1:$((MESH_BASE_PORT + j))"
            fi
        done

        echo "  Gateway $i: port=$port mesh=$mesh_port metrics=$metrics_port"

        # No --worker-urls: workers are registered via API so they flow through mesh
        "$SMG_BIN" \
            --host 127.0.0.1 \
            --port "$port" \
            --policy cache_aware \
            --enable-mesh \
            --mesh-host 127.0.0.1 \
            --mesh-port "$mesh_port" \
            --mesh-server-name "gateway-$i" \
            --mesh-peer-urls $MESH_PEERS \
            --prometheus-port "$metrics_port" \
            --prometheus-host 127.0.0.1 \
            --log-level debug \
            > "$LOG_DIR/gateway-$i.log" 2>&1 &

        echo $! >> "$LOG_DIR/gateway_pids.txt"
    done
    echo "Gateways started. Logs: $LOG_DIR/gateway-*.log"
}

register_workers() {
    # Register workers via the admin API on gateway-0.
    # The mesh syncs worker state to the other gateways.
    local gw_port=$GATEWAY_BASE_PORT
    echo "Registering $NUM_WORKERS workers via API on gateway-0 (port $gw_port)..."

    local registered=0
    for i in $(seq 0 $((NUM_WORKERS - 1))); do
        local worker_port=$((WORKER_BASE_PORT + i))
        local resp
        resp=$(curl -sf -X POST "http://127.0.0.1:$gw_port/workers" \
            -H "Content-Type: application/json" \
            -d "{\"url\": \"http://127.0.0.1:$worker_port\"}" 2>&1) && registered=$((registered + 1))
    done
    echo "Registered $registered/$NUM_WORKERS workers"
    if [ "$registered" -ne "$NUM_WORKERS" ]; then
        echo "WARNING: only $registered of $NUM_WORKERS workers registered — some may have failed to start"
    fi

    # Verify workers synced to other gateways
    echo "Waiting 3s for mesh sync..."
    sleep 3
    for i in $(seq 0 $((NUM_GATEWAYS - 1))); do
        local port=$((GATEWAY_BASE_PORT + i))
        local count
        count=$(curl -sf "http://127.0.0.1:$port/workers" 2>/dev/null | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('data',[])))" 2>/dev/null || echo "?")
        echo "  Gateway $i: $count workers"
    done
}

run_load() {
    local rps="${1:-200}"
    local duration="${2:-60}"
    local ports=""
    for i in $(seq 0 $((NUM_GATEWAYS - 1))); do
        [ -n "$ports" ] && ports="$ports,"
        ports="$ports$((GATEWAY_BASE_PORT + i))"
    done
    echo "Running load: ${rps} req/s for ${duration}s across $NUM_GATEWAYS gateways"
    python3 "$SCRIPT_DIR/mesh_load_gen.py" \
        --rps "$rps" \
        --duration "$duration" \
        --gateway-ports "$ports"
}

stop_all() {
    echo "Stopping all processes..."
    for pidfile in "$LOG_DIR/worker_pids.txt" "$LOG_DIR/gateway_pids.txt"; do
        if [ -f "$pidfile" ]; then
            while read -r pid; do
                kill "$pid" 2>/dev/null || true
            done < "$pidfile"
            rm "$pidfile"
        fi
    done
    echo "Stopped."
}

show_status() {
    echo "=== Gateways ==="
    for i in $(seq 0 $((NUM_GATEWAYS - 1))); do
        port=$((GATEWAY_BASE_PORT + i))
        metrics_port=$((METRICS_BASE_PORT + i))
        echo -n "  Gateway $i (port $port): "
        if curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
            echo "UP"
        else
            echo "DOWN"
        fi

        # Mesh sync round duration
        curl -sf "http://127.0.0.1:$metrics_port/metrics" 2>/dev/null \
            | grep -E 'router_mesh_sync_round_duration|router_mesh_sync_batch_bytes|router_mesh_store_' \
            | grep -v '^#' \
            | sed 's/^/    /' || true
        echo ""
    done

    echo "=== Workers ==="
    up=0; down=0
    for i in $(seq 0 $((NUM_WORKERS - 1))); do
        port=$((WORKER_BASE_PORT + i))
        if curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
            up=$((up + 1))
        else
            down=$((down + 1))
        fi
    done
    echo "  $up UP, $down DOWN (of $NUM_WORKERS)"
}

case "${1:-help}" in
    start)
        start_mock_workers
        sleep 1
        start_gateways
        echo ""
        echo "Waiting 5s for mesh to converge..."
        sleep 5
        register_workers
        echo ""
        show_status
        echo ""
        echo "Next: $0 load [rps] [duration_secs]"
        ;;
    load)
        run_load "${2:-200}" "${3:-60}"
        ;;
    stop)
        stop_all
        ;;
    status)
        show_status
        ;;
    *)
        echo "Usage: $0 {start|load [rps] [duration]|status|stop}"
        echo ""
        echo "  start              Start 20 mock workers + 3 mesh gateways"
        echo "  load [rps] [dur]   Send load (default: 200 req/s for 60s)"
        echo "  status             Show health + mesh metrics"
        echo "  stop               Stop everything"
        ;;
esac
