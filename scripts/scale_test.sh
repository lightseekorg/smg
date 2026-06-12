#!/usr/bin/env bash
# Scale-test rig for the SMG gateway using mock workers (crates/mock_worker).
#
# Launches one IGW gateway, starts N mock HTTP and/or gRPC workers in a single
# mock-worker process, REST-registers them, then samples the GATEWAY process's
# CPU and /health latency at idle and (optionally) under load. Per-PID sampling
# isolates the gateway so the mock's own CPU does not confound the measurement.
#
# Usage:
#   scripts/scale_test.sh [--http N] [--grpc N] [--policy P] [--rps R]
#                         [--duration S] [--gen-ms MS] [--no-build]
#
# Examples:
#   scripts/scale_test.sh --http 2000 --policy cache_aware --rps 500 --duration 30
#   scripts/scale_test.sh --grpc 1000 --policy least_load
set -euo pipefail

# ---- defaults ----
HTTP=0
GRPC=0
POLICY="cache_aware"
RPS=0
DURATION=20
GEN_MS=5
GW_PORT=30000
HTTP_BASE=9000
GRPC_BASE=19000
MODEL="mock-model"
BUILD=1
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --http) HTTP="$2"; shift 2 ;;
    --grpc) GRPC="$2"; shift 2 ;;
    --policy) POLICY="$2"; shift 2 ;;
    --rps) RPS="$2"; shift 2 ;;
    --duration) DURATION="$2"; shift 2 ;;
    --gen-ms) GEN_MS="$2"; shift 2 ;;
    --gw-port) GW_PORT="$2"; shift 2 ;;
    --no-build) BUILD=0; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ "$HTTP" -eq 0 && "$GRPC" -eq 0 ]]; then
  echo "pass --http N and/or --grpc N" >&2; exit 2
fi

# Thousands of mock ports + gateway connections need plenty of file descriptors.
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

# Kill leftovers from prior runs so mock ports are free to bind.
pkill -9 -x mock-worker 2>/dev/null || true
pkill -9 -x smg 2>/dev/null || true
sleep 2

GW_PID=""; MOCK_PID=""
cleanup() {
  [[ -n "$MOCK_PID" ]] && kill "$MOCK_PID" 2>/dev/null || true
  [[ -n "$GW_PID" ]] && kill "$GW_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Pin the target dir so `cargo build` and binary resolution always agree. The
# environment may otherwise redirect the target dir to a hash that is not stable
# across invocations (causing cold rebuilds + wrong paths). Override by exporting
# CARGO_TARGET_DIR before running this script (e.g. point it at a warm dir).
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
if [[ "$BUILD" -eq 1 ]]; then
  echo "==> building smg + mock-worker (release, sccache off) into $CARGO_TARGET_DIR"
  RUSTC_WRAPPER="" cargo build --release -p smg -p mock-worker
fi
SMG="$CARGO_TARGET_DIR/release/smg"
MOCK="$CARGO_TARGET_DIR/release/mock-worker"
if [[ ! -x "$SMG" || ! -x "$MOCK" ]]; then
  echo "binaries not found in $CARGO_TARGET_DIR/release (run without --no-build)" >&2
  exit 1
fi
echo "    smg=$SMG"

# Redirect both processes to log files. If they inherited this script's stdout
# (often an unread pipe), their heavy logging would fill the pipe buffer and
# block the process — stalling the gateway before it serves.
LOGDIR="${TMPDIR:-/tmp}/smg-scale"
mkdir -p "$LOGDIR"
echo "==> logs: $LOGDIR/{mock,gateway}.log"

# ---- start mock workers (one process, many ports) ----
echo "==> starting mock workers: $HTTP http, $GRPC grpc"
MOCK_ARGS=(--model "$MODEL" --gen-ms "$GEN_MS")
[[ "$HTTP" -gt 0 ]] && MOCK_ARGS+=(--http-base-port "$HTTP_BASE" --http-count "$HTTP")
[[ "$GRPC" -gt 0 ]] && MOCK_ARGS+=(--grpc-base-port "$GRPC_BASE" --grpc-count "$GRPC")
"$MOCK" "${MOCK_ARGS[@]}" >"$LOGDIR/mock.log" 2>&1 &
MOCK_PID=$!
sleep 2

# ---- start gateway in IGW mode ----
echo "==> starting gateway on :$GW_PORT (policy=$POLICY, IGW)"
GW_ARGS=(--host 127.0.0.1 --port "$GW_PORT" --enable-igw --policy "$POLICY")
# gRPC workers need a tokenizer; skip autoload so registration/routing can be
# measured without a real tokenizer (generation itself is not exercised here).
[[ "$GRPC" -gt 0 ]] && GW_ARGS+=(--disable-tokenizer-autoload)
"$SMG" "${GW_ARGS[@]}" >"$LOGDIR/gateway.log" 2>&1 &
GW_PID=$!

echo "==> waiting for gateway /health"
for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$GW_PORT/health" >/dev/null 2>&1 && break
  sleep 1
done

# ---- register workers via REST, in parallel ----
# Health disabled so workers are instantly Ready/routable — isolates routing
# cost from the per-worker health-probe loop (and avoids the 60s promote wait).
# Direct curl per port ({} = port via xargs); model/mode pinned so routing has a
# concrete model (bare-URL auto-detect leaves model_id "unknown" -> 404 on route).
echo "==> registering workers via POST /workers"
H='"health":{"disable_health_check":true}'
W="http://127.0.0.1:$GW_PORT/workers"
if [[ "$HTTP" -gt 0 ]]; then
  seq "$HTTP_BASE" $((HTTP_BASE + HTTP - 1)) | xargs -P 32 -I{} \
    curl -s -o /dev/null --max-time 15 -X POST "$W" -H 'content-type: application/json' \
    -d "{\"url\":\"http://127.0.0.1:{}\",\"connection_mode\":\"http\",\"runtime\":\"sglang\",\"models\":[{\"id\":\"$MODEL\"}],$H}" || true
fi
if [[ "$GRPC" -gt 0 ]]; then
  seq "$GRPC_BASE" $((GRPC_BASE + GRPC - 1)) | xargs -P 32 -I{} \
    curl -s -o /dev/null --max-time 15 -X POST "$W" -H 'content-type: application/json' \
    -d "{\"url\":\"grpc://127.0.0.1:{}\",\"connection_mode\":\"grpc\",\"runtime\":\"tokenspeed\",\"models\":[{\"id\":\"$MODEL\"}],$H}" || true
fi

echo "==> waiting for workers to become Ready"
EXPECT=$((HTTP + GRPC))
n=0
for _ in $(seq 1 120); do
  body=$(curl -sf --max-time 20 "http://127.0.0.1:$GW_PORT/workers" 2>/dev/null || true)
  n=$(printf '%s' "$body" | grep -o '"url"' | wc -l | tr -d ' ' || true)
  n=${n:-0}
  # Accept ~95%: a few ports may collide and a couple of workflows lag.
  [[ "$n" -ge $((EXPECT * 95 / 100)) ]] && break
  sleep 1
done
echo "    registered ~${n:-0} / $EXPECT workers"

# ---- sample gateway CPU + /health latency ----
sample() {
  local label="$1" secs="$2"
  echo "==> [$label] sampling gateway PID $GW_PID for ${secs}s"
  for _ in $(seq 1 "$secs"); do
    cpu=$(ps -p "$GW_PID" -o pcpu= 2>/dev/null | tr -d ' ' || true)
    rss=$(ps -p "$GW_PID" -o rss= 2>/dev/null | tr -d ' ' || true)
    hl=$(curl -so /dev/null -w '%{time_total}' "http://127.0.0.1:$GW_PORT/health" 2>/dev/null || echo NA)
    echo "    [$label] cpu=${cpu:-NA}% rss_kb=${rss:-NA} health_s=${hl}"
    sleep 1
  done
}

sample "idle" 5

# ---- optional load phase ----
if [[ "$RPS" -gt 0 ]]; then
  echo "==> load: ${RPS} rps for ${DURATION}s against /v1/chat/completions"
  python3 "$ROOT/scripts/scale_load.py" \
    --url "http://127.0.0.1:$GW_PORT/v1/chat/completions" \
    --model "$MODEL" --rps "$RPS" --duration "$DURATION" &
  LOAD_PID=$!
  sample "load" "$DURATION"
  wait "$LOAD_PID" 2>/dev/null || true
fi

echo "==> done (gateway pid $GW_PID, mock pid $MOCK_PID); tearing down"
