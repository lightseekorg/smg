#!/bin/bash
# Run a single benchmark scenario
# Usage: ./scripts/mesh-bench-one.sh <num_workers> <rps> <duration_secs>
set -euo pipefail

WORKERS=${1:-20}
RPS=${2:-100}
DURATION=${3:-30}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$(cargo metadata --no-deps --format-version 1 2>/dev/null | python3 -c 'import sys,json; print(json.load(sys.stdin)["target_directory"])' 2>/dev/null || echo "${REPO_ROOT}/target")}"
SMG_BIN="${CARGO_TARGET_DIR}/release/smg"
LOG_DIR="${REPO_ROOT}/target/mesh-bench"
NUM_GATEWAYS=3
WORKER_BASE=9000
GW_BASE=30000
MESH_BASE=39500
METRICS_BASE=29000

mkdir -p "$LOG_DIR"

# Cleanup
pkill -f mock_worker.py 2>/dev/null || true
pkill -f 'smg.*mesh' 2>/dev/null || true
sleep 2

# Start workers
for i in $(seq 0 $((WORKERS - 1))); do
    python3 "$SCRIPT_DIR/mock_worker.py" $((WORKER_BASE + i)) &
done
sleep 1

# Start gateways
for i in $(seq 0 $((NUM_GATEWAYS - 1))); do
    PEERS=""
    for j in $(seq 0 $((NUM_GATEWAYS - 1))); do
        [ "$i" != "$j" ] && PEERS="$PEERS 127.0.0.1:$((MESH_BASE + j))"
    done
    "$SMG_BIN" --host 127.0.0.1 --port $((GW_BASE + i)) --policy cache_aware \
        --enable-mesh --mesh-host 127.0.0.1 --mesh-port $((MESH_BASE + i)) \
        --mesh-server-name "gw-$i" --mesh-peer-urls $PEERS \
        --prometheus-port $((METRICS_BASE + i)) --prometheus-host 127.0.0.1 \
        --log-level warn > "$LOG_DIR/gw-$i.log" 2>&1 &
done
sleep 5

# Register workers
REG=0
for i in $(seq 0 $((WORKERS - 1))); do
    curl -sf -X POST "http://127.0.0.1:$GW_BASE/workers" \
        -H "Content-Type: application/json" \
        -d "{\"url\":\"http://127.0.0.1:$((WORKER_BASE + i))\"}" >/dev/null 2>&1 && REG=$((REG + 1))
done
sleep 20

# Run load
PORTS="$GW_BASE"
for i in $(seq 1 $((NUM_GATEWAYS - 1))); do PORTS="$PORTS,$((GW_BASE + i))"; done
python3 "$SCRIPT_DIR/mesh_load_gen.py" --rps "$RPS" --duration "$DURATION" --gateway-ports "$PORTS" 2>&1 | grep -E 'Done|Avg'

# Collect metrics from gateway 0
M=$METRICS_BASE
POL_BYTES=$(curl -sf "http://127.0.0.1:$M/metrics" | grep 'router_mesh_sync_batch_bytes_sum.*policy' | awk '{s+=$2}END{print s+0}')
POL_COUNT=$(curl -sf "http://127.0.0.1:$M/metrics" | grep 'router_mesh_sync_batch_bytes_count.*policy' | awk '{s+=$2}END{print s+0}')
RND_SUM=$(curl -sf "http://127.0.0.1:$M/metrics" | grep 'router_mesh_sync_round_duration_seconds_sum' | awk '{s+=$2}END{print s+0}')
RND_COUNT=$(curl -sf "http://127.0.0.1:$M/metrics" | grep 'router_mesh_sync_round_duration_seconds_count' | awk '{s+=$2}END{print s+0}')
WRK_BYTES=$(curl -sf "http://127.0.0.1:$M/metrics" | grep 'router_mesh_sync_batch_bytes_sum.*worker' | awk '{s+=$2}END{print s+0}')

AVG_BATCH=$(echo "$POL_BYTES $POL_COUNT" | awk '{if($2>0) printf "%.1f",$1/$2/1024; else print "0"}')
AVG_ROUND=$(echo "$RND_SUM $RND_COUNT" | awk '{if($2>0) printf "%.2f",$1/$2*1000; else print "0"}')
TOT_KB=$(echo "$POL_BYTES" | awk '{printf "%.1f",$1/1024}')
WRK_KB=$(echo "$WRK_BYTES" | awk '{printf "%.1f",$1/1024}')

echo ""
echo "| ${WORKERS}w / ${RPS}rps / ${DURATION}s | ${TOT_KB} KB | $POL_COUNT | ${AVG_BATCH} KB | ${AVG_ROUND} ms | ${WRK_KB} KB | ${REG}/${WORKERS} reg |"
