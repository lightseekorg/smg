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

GW_PID=""; MOCK_PID=""
cleanup() {
  [[ -n "$MOCK_PID" ]] && kill "$MOCK_PID" 2>/dev/null || true
  [[ -n "$GW_PID" ]] && kill "$GW_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if [[ "$BUILD" -eq 1 ]]; then
  echo "==> building smg + mock-worker (release, sccache off)"
  RUSTC_WRAPPER="" cargo build --release -p smg -p mock-worker
fi
# Resolve the real target dir (it may be redirected via CARGO_TARGET_DIR or a
# global cargo config, so it is not necessarily ./target).
TARGET_DIR=$(cargo metadata --format-version 1 --no-deps \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')
SMG="$TARGET_DIR/release/smg"
MOCK="$TARGET_DIR/release/mock-worker"

# ---- start mock workers (one process, many ports) ----
echo "==> starting mock workers: $HTTP http, $GRPC grpc"
MOCK_ARGS=(--model "$MODEL" --gen-ms "$GEN_MS")
[[ "$HTTP" -gt 0 ]] && MOCK_ARGS+=(--http-base-port "$HTTP_BASE" --http-count "$HTTP")
[[ "$GRPC" -gt 0 ]] && MOCK_ARGS+=(--grpc-base-port "$GRPC_BASE" --grpc-count "$GRPC")
"$MOCK" "${MOCK_ARGS[@]}" &
MOCK_PID=$!
sleep 2

# ---- start gateway in IGW mode ----
echo "==> starting gateway on :$GW_PORT (policy=$POLICY, IGW)"
GW_ARGS=(--host 127.0.0.1 --port "$GW_PORT" --enable-igw --policy "$POLICY")
# gRPC workers need a tokenizer; skip autoload so registration/routing can be
# measured without a real tokenizer (generation itself is not exercised here).
[[ "$GRPC" -gt 0 ]] && GW_ARGS+=(--disable-tokenizer-autoload)
"$SMG" "${GW_ARGS[@]}" &
GW_PID=$!

echo "==> waiting for gateway /health"
for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$GW_PORT/health" >/dev/null 2>&1 && break
  sleep 1
done

# ---- register workers via REST, in parallel ----
echo "==> registering workers via POST /workers"
register() {
  local url="$1" body
  if [[ "$url" == grpc://* ]]; then
    body="{\"url\":\"$url\",\"connection_mode\":\"grpc\",\"runtime\":\"tokenspeed\",\"models\":[{\"id\":\"$MODEL\"}]}"
  else
    body="{\"url\":\"$url\",\"connection_mode\":\"http\",\"runtime\":\"sglang\",\"models\":[{\"id\":\"$MODEL\"}]}"
  fi
  curl -sf -X POST "http://127.0.0.1:$GW_PORT/workers" \
    -H 'content-type: application/json' -d "$body" >/dev/null 2>&1 || true
}
export -f register
export GW_PORT MODEL
{
  for ((p = HTTP_BASE; p < HTTP_BASE + HTTP; p++)); do echo "http://127.0.0.1:$p"; done
  for ((p = GRPC_BASE; p < GRPC_BASE + GRPC; p++)); do echo "grpc://127.0.0.1:$p"; done
} | xargs -P 32 -I {} bash -c 'register "$@"' _ {}

echo "==> waiting for workers to become Ready"
EXPECT=$((HTTP + GRPC))
n=0
for _ in $(seq 1 120); do
  body=$(curl -sf "http://127.0.0.1:$GW_PORT/workers" 2>/dev/null || true)
  n=$(printf '%s' "$body" | grep -o '"url"' | wc -l | tr -d ' ')
  [[ "${n:-0}" -ge "$EXPECT" ]] && break
  sleep 1
done
echo "    registered ~${n:-0} / $EXPECT workers"

# ---- sample gateway CPU + /health latency ----
sample() {
  local label="$1" secs="$2"
  echo "==> [$label] sampling gateway PID $GW_PID for ${secs}s"
  for _ in $(seq 1 "$secs"); do
    cpu=$(ps -o %cpu= -p "$GW_PID" 2>/dev/null | tr -d ' ')
    rss=$(ps -o rss= -p "$GW_PID" 2>/dev/null | tr -d ' ')
    hl=$(curl -so /dev/null -w '%{time_total}' "http://127.0.0.1:$GW_PORT/health" 2>/dev/null || echo NA)
    echo "    [$label] cpu=${cpu}% rss_kb=${rss} health_s=${hl}"
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
