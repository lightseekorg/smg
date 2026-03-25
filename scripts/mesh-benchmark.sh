#!/bin/bash
# Mesh benchmark suite
#
# Runs multiple configurations and collects metrics for comparison.
#
# Usage: ./scripts/mesh-benchmark.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$(cargo metadata --no-deps --format-version 1 2>/dev/null | python3 -c 'import sys,json; print(json.load(sys.stdin)["target_directory"])' 2>/dev/null || echo "${REPO_ROOT}/target")}"
SMG_BIN="${CARGO_TARGET_DIR}/release/smg"
LOG_DIR="${REPO_ROOT}/target/mesh-benchmark"
RESULTS_FILE="${LOG_DIR}/results.md"

WORKER_BASE_PORT=9000
GATEWAY_BASE_PORT=30000
MESH_BASE_PORT=39500
METRICS_BASE_PORT=29000
NUM_GATEWAYS=3

mkdir -p "$LOG_DIR"

cleanup() {
    # Kill all mock workers and gateways
    pkill -f mock_worker.py 2>/dev/null || true
    pkill -f "smg.*mesh" 2>/dev/null || true
    sleep 2
    # Clean PID files
    rm -f "$LOG_DIR"/*.pid
}

start_workers() {
    local count=$1
    echo "  Starting $count mock workers..."
    for i in $(seq 0 $((count - 1))); do
        local port=$((WORKER_BASE_PORT + i))
        python3 "$SCRIPT_DIR/mock_worker.py" "$port" &
        echo $! >> "$LOG_DIR/worker.pid"
    done
    sleep 1
}

start_gateways() {
    local num_workers=$1
    echo "  Starting $NUM_GATEWAYS gateways (mesh enabled)..."

    for i in $(seq 0 $((NUM_GATEWAYS - 1))); do
        local port=$((GATEWAY_BASE_PORT + i))
        local mesh_port=$((MESH_BASE_PORT + i))
        local metrics_port=$((METRICS_BASE_PORT + i))

        local MESH_PEERS=""
        for j in $(seq 0 $((NUM_GATEWAYS - 1))); do
            if [ "$i" != "$j" ]; then
                MESH_PEERS="$MESH_PEERS 127.0.0.1:$((MESH_BASE_PORT + j))"
            fi
        done

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
            --log-level warn \
            > "$LOG_DIR/gateway-$i.log" 2>&1 &

        echo $! >> "$LOG_DIR/gateway.pid"
    done
    sleep 5
}

register_workers() {
    local count=$1
    local registered=0
    for i in $(seq 0 $((count - 1))); do
        local port=$((WORKER_BASE_PORT + i))
        curl -sf -X POST "http://127.0.0.1:$GATEWAY_BASE_PORT/workers" \
            -H "Content-Type: application/json" \
            -d "{\"url\": \"http://127.0.0.1:$port\"}" > /dev/null 2>&1 && registered=$((registered + 1))
    done
    echo "  Registered $registered/$count workers"
    # Wait for workers to be processed
    sleep 20
}

collect_metrics() {
    local label=$1
    local metrics_port=$METRICS_BASE_PORT

    local policy_bytes
    policy_bytes=$(curl -sf "http://127.0.0.1:$metrics_port/metrics" 2>/dev/null \
        | grep 'router_mesh_sync_batch_bytes_sum.*policy' \
        | awk '{sum += $2} END {print sum+0}')

    local policy_count
    policy_count=$(curl -sf "http://127.0.0.1:$metrics_port/metrics" 2>/dev/null \
        | grep 'router_mesh_sync_batch_bytes_count.*policy' \
        | awk '{sum += $2} END {print sum+0}')

    local round_sum
    round_sum=$(curl -sf "http://127.0.0.1:$metrics_port/metrics" 2>/dev/null \
        | grep 'router_mesh_sync_round_duration_seconds_sum' \
        | awk '{sum += $2} END {print sum+0}')

    local round_count
    round_count=$(curl -sf "http://127.0.0.1:$metrics_port/metrics" 2>/dev/null \
        | grep 'router_mesh_sync_round_duration_seconds_count' \
        | awk '{sum += $2} END {print sum+0}')

    local worker_bytes
    worker_bytes=$(curl -sf "http://127.0.0.1:$metrics_port/metrics" 2>/dev/null \
        | grep 'router_mesh_sync_batch_bytes_sum.*worker' \
        | awk '{sum += $2} END {print sum+0}')

    local avg_batch="0"
    if [ "$policy_count" -gt 0 ] 2>/dev/null; then
        avg_batch=$(echo "$policy_bytes $policy_count" | awk '{printf "%.1f", $1/$2/1024}')
    fi

    local avg_round="0"
    if [ "$round_count" -gt 0 ] 2>/dev/null; then
        avg_round=$(echo "$round_sum $round_count" | awk '{printf "%.2f", $1/$2*1000}')
    fi

    local total_kb
    total_kb=$(echo "$policy_bytes" | awk '{printf "%.1f", $1/1024}')

    local worker_kb
    worker_kb=$(echo "$worker_bytes" | awk '{printf "%.1f", $1/1024}')

    echo "| $label | ${total_kb} KB | $policy_count | ${avg_batch} KB | ${avg_round} ms | ${worker_kb} KB |"
}

run_load() {
    local rps=$1
    local duration=$2
    local ports=""
    for i in $(seq 0 $((NUM_GATEWAYS - 1))); do
        [ -n "$ports" ] && ports="$ports,"
        ports="$ports$((GATEWAY_BASE_PORT + i))"
    done
    python3 "$SCRIPT_DIR/mesh_load_gen.py" \
        --rps "$rps" \
        --duration "$duration" \
        --gateway-ports "$ports" 2>&1 | tail -3
}

run_scenario() {
    local workers=$1
    local rps=$2
    local duration=$3
    local label="${workers}w / ${rps}rps / ${duration}s"

    echo ""
    echo "=== Scenario: $label ==="
    cleanup
    start_workers "$workers"
    start_gateways "$workers"
    register_workers "$workers"

    echo "  Running load: ${rps} req/s for ${duration}s"
    run_load "$rps" "$duration"

    # Collect and print metrics
    collect_metrics "$label"
}

# Main
echo "# Mesh Delta Encoding Benchmark"
echo ""
echo "Configuration: $NUM_GATEWAYS gateways, cache_aware policy"
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# Header
echo "| Scenario | Policy Total | Batches | Avg Batch | Avg Round | Worker Total |"
echo "|----------|-------------|---------|-----------|-----------|-------------|"

# Run scenarios
{
    # Vary workers
    run_scenario 10 100 30
    cleanup

    run_scenario 20 100 30
    cleanup

    run_scenario 40 100 30
    cleanup

    # Vary RPS with 20 workers
    run_scenario 20 50 30
    cleanup

    run_scenario 20 200 30
    cleanup

    run_scenario 20 500 30
    cleanup

    # Sustained load
    run_scenario 20 200 120
    cleanup

} 2>&1 | tee "$LOG_DIR/benchmark.log"

cleanup
echo ""
echo "Full log: $LOG_DIR/benchmark.log"
