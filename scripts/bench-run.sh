#!/bin/bash
# Single scenario runner that prints result immediately
# Usage: bench-run.sh <workers> <rps> <duration> [prompt_size]
set -uo pipefail
W=${1:-20}; R=${2:-100}; D=${3:-30}; PS=${4:-0}
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TD="${CARGO_TARGET_DIR:-$(cargo metadata --no-deps --format-version 1 2>/dev/null | python3 -c 'import sys,json;print(json.load(sys.stdin)["target_directory"])' 2>/dev/null)}"
BIN="$TD/release/smg"

# Start workers
for i in $(seq 0 $((W-1))); do python3 "$DIR/mock_worker.py" $((9000+i)) & done
sleep 1

# Start 3 gateways
for i in 0 1 2; do
  P=""; for j in 0 1 2; do [ "$i" != "$j" ] && P="$P 127.0.0.1:$((39500+j))"; done
  "$BIN" --host 127.0.0.1 --port $((30000+i)) --policy cache_aware \
    --enable-mesh --mesh-host 127.0.0.1 --mesh-port $((39500+i)) \
    --mesh-server-name "gw-$i" --mesh-peer-urls $P \
    --prometheus-port $((29000+i)) --prometheus-host 127.0.0.1 \
    --log-level warn >/dev/null 2>&1 &
done
sleep 5

# Register workers
RG=0
for i in $(seq 0 $((W-1))); do
  curl -sf -X POST "http://127.0.0.1:30000/workers" \
    -H "Content-Type: application/json" \
    -d "{\"url\":\"http://127.0.0.1:$((9000+i))\"}" >/dev/null 2>&1 && RG=$((RG+1))
done
sleep 20

# Run load
python3 "$DIR/mesh_load_gen.py" --rps "$R" --duration "$D" \
  --gateway-ports "30000,30001,30002" --prompt-size "$PS" 2>/dev/null | grep -E 'Done' >&2

# Collect
PB=$(curl -sf http://127.0.0.1:29000/metrics | awk '/sync_batch_bytes_sum.*policy/{s+=$2}END{print s+0}')
PC=$(curl -sf http://127.0.0.1:29000/metrics | awk '/sync_batch_bytes_count.*policy/{s+=$2}END{print s+0}')
RS=$(curl -sf http://127.0.0.1:29000/metrics | awk '/sync_round_duration_seconds_sum/{s+=$2}END{print s+0}')
RC=$(curl -sf http://127.0.0.1:29000/metrics | awk '/sync_round_duration_seconds_count/{s+=$2}END{print s+0}')

AB=$(echo "$PB $PC" | awk '{if($2>0)printf "%.1f",$1/$2/1024;else print "0"}')
AR=$(echo "$RS $RC" | awk '{if($2>0)printf "%.2f",$1/$2*1000;else print "0"}')
TK=$(echo "$PB" | awk '{printf "%.1f",$1/1024}')

PSLABEL=""
[ "$PS" -gt 0 ] && PSLABEL=" / ${PS}ch"
printf "| %dw / %drps / %ds%s | %s KB | %d | %s KB | %s ms | %d/%d |\n" "$W" "$R" "$D" "$PSLABEL" "$TK" "$PC" "$AB" "$AR" "$RG" "$W"

# Cleanup
pkill -f mock_worker.py 2>/dev/null || true
pkill -f 'smg.*mesh' 2>/dev/null || true
