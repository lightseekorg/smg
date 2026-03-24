#!/bin/bash
# Local mesh profiling test
# Spins up 3 SMG gateway replicas with mock workers for mesh profiling.
#
# Prerequisites:
#   cargo build --release -p smg
#
# Usage:
#   ./scripts/mesh-profile-local.sh [start|stop|status]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="${REPO_ROOT}/target/mesh-profile"
SMG_BIN="${REPO_ROOT}/target/release/smg"
NUM_WORKERS=20
NUM_GATEWAYS=3
WORKER_BASE_PORT=9000
GATEWAY_BASE_PORT=30000
MESH_BASE_PORT=39500
METRICS_BASE_PORT=29000

mkdir -p "$LOG_DIR"

start_mock_workers() {
    echo "Starting $NUM_WORKERS mock workers..."
    for i in $(seq 0 $((NUM_WORKERS - 1))); do
        port=$((WORKER_BASE_PORT + i))
        # Use python http.server as a minimal mock
        python3 -c "
import http.server, json, threading

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{\"status\":\"healthy\"}')
        elif self.path == '/v1/models':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            resp = json.dumps({'data': [{'id': 'mock-model', 'object': 'model', 'root': 'mock-model', 'max_model_len': 4096}]})
            self.wfile.write(resp.encode())
        elif self.path == '/get_model_info':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            resp = json.dumps({'model_path': 'mock-model', 'is_generation': True, 'max_total_num_tokens': 4096})
            self.wfile.write(resp.encode())
        else:
            self.send_response(404)
            self.end_headers()
    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(b'{\"choices\":[{\"message\":{\"content\":\"hello\"}}]}')
    def log_message(self, format, *args):
        pass  # Suppress logs

http.server.HTTPServer(('127.0.0.1', $port), Handler).serve_forever()
" &
        echo $! >> "$LOG_DIR/worker_pids.txt"
    done
    echo "Mock workers started on ports $WORKER_BASE_PORT-$((WORKER_BASE_PORT + NUM_WORKERS - 1))"
}

start_gateways() {
    echo "Starting $NUM_GATEWAYS SMG gateway replicas with mesh..."

    # Build worker URLs
    WORKER_URLS=""
    for i in $(seq 0 $((NUM_WORKERS - 1))); do
        WORKER_URLS="$WORKER_URLS http://127.0.0.1:$((WORKER_BASE_PORT + i))"
    done

    # Build peer URLs for mesh (each gateway knows about the others)
    for i in $(seq 0 $((NUM_GATEWAYS - 1))); do
        port=$((GATEWAY_BASE_PORT + i))
        mesh_port=$((MESH_BASE_PORT + i))
        metrics_port=$((METRICS_BASE_PORT + i))

        # Build mesh peer list (all others)
        MESH_PEERS=""
        for j in $(seq 0 $((NUM_GATEWAYS - 1))); do
            if [ "$i" != "$j" ]; then
                MESH_PEERS="$MESH_PEERS http://127.0.0.1:$((MESH_BASE_PORT + j))"
            fi
        done

        echo "Starting gateway $i on port $port (mesh: $mesh_port, metrics: $metrics_port)"

        "$SMG_BIN" \
            --host 127.0.0.1 \
            --port "$port" \
            --worker-urls $WORKER_URLS \
            --policy cache_aware \
            --enable-mesh \
            --mesh-host 127.0.0.1 \
            --mesh-port "$mesh_port" \
            --mesh-server-name "gateway-$i" \
            --mesh-peer-urls $MESH_PEERS \
            --prometheus-port "$metrics_port" \
            --prometheus-host 127.0.0.1 \
            --log-level info \
            --disable-health-check \
            > "$LOG_DIR/gateway-$i.log" 2>&1 &

        echo $! >> "$LOG_DIR/gateway_pids.txt"
    done
    echo "Gateways started"
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
    echo "All processes stopped"
}

show_status() {
    echo "=== Gateway Status ==="
    for i in $(seq 0 $((NUM_GATEWAYS - 1))); do
        port=$((GATEWAY_BASE_PORT + i))
        metrics_port=$((METRICS_BASE_PORT + i))
        echo -n "Gateway $i (port $port): "
        if curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
            echo "UP"
        else
            echo "DOWN"
        fi

        # Show mesh metrics
        echo "  Mesh sync metrics:"
        curl -sf "http://127.0.0.1:$metrics_port/metrics" 2>/dev/null \
            | grep -E 'router_mesh_sync|router_mesh_store|router_mesh_peer' \
            | head -10 || echo "  (no metrics available)"
        echo ""
    done

    echo "=== Worker Status ==="
    up=0
    down=0
    for i in $(seq 0 $((NUM_WORKERS - 1))); do
        port=$((WORKER_BASE_PORT + i))
        if curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
            up=$((up + 1))
        else
            down=$((down + 1))
        fi
    done
    echo "$up workers UP, $down workers DOWN"
}

case "${1:-status}" in
    start)
        start_mock_workers
        sleep 1
        start_gateways
        echo ""
        echo "Waiting 5s for mesh to converge..."
        sleep 5
        show_status
        echo ""
        echo "Logs: $LOG_DIR/gateway-*.log"
        echo "Stop: $0 stop"
        ;;
    stop)
        stop_all
        ;;
    status)
        show_status
        ;;
    *)
        echo "Usage: $0 [start|stop|status]"
        exit 1
        ;;
esac
