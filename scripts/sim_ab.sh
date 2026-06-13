#!/usr/bin/env bash
# No-GPU routing-policy A/B harness for SMG, using the realistic mock engine.
#
# For each routing policy, this launches an IGW gateway plus a fleet of mock
# workers running the continuous-batching engine simulator (crates/mock_worker
# --engine realistic), drives the same Poisson workload (scripts/sim_load.py),
# and prints a side-by-side table of TTFT / ITL / E2E / throughput. Because the
# mock reports realistic loads and emits KV-cache events, least_load and
# cache_aware engage exactly as they would against a real engine — on CPU.
#
# Modes:
#   grpc (default) — full fidelity. The gateway tokenizes prompts and routes on
#     token ids, so BOTH least_load (queued token-work) and cache_aware
#     (event-driven KV overlap) are exercised. Requires a real tokenizer for the
#     model (default: the public `gpt2`, downloaded by the gateway, or pass a
#     local dir via --tokenizer). Workers register with runtime=tokenspeed.
#   http — offline-friendly. No tokenizer needed; drives least_load + latency and
#     the approximate (string-tree) cache_aware. Workers register runtime=sglang.
#
# Usage:
#   scripts/sim_ab.sh [--mode grpc|http] [--workers N] [--rps R] [--duration S]
#                     [--policies "p1 p2 ..."] [--tokenizer ID|DIR]
#                     [--output-tokens N] [--no-build]
# Examples:
#   scripts/sim_ab.sh --mode grpc --workers 8 --rps 120 --duration 30
#   scripts/sim_ab.sh --mode http --workers 16 --policies "random least_load"
set -euo pipefail

# ---- defaults ----
MODE=grpc
WORKERS=8
POLICIES="random round_robin least_load cache_aware"
RPS=80
DURATION=20
OUTPUT_TOKENS=64
TOKENIZER=gpt2
SHARED_PREFIX_WORDS=200
SHARED_PREFIX_FRAC=0.7
GW_PORT=30100
HTTP_BASE=9200
GRPC_BASE=19200
MODEL=mock-model
BUILD=1
# Realistic engine knobs (forwarded to mock-worker).
PREFILL_TPS=8000
DECODE_BASE_MS=6
DECODE_PER_REQ_MS=0.35
MAX_RUNNING=256
KV_TOKENS=524288
BLOCK_SIZE=16
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --policies) POLICIES="$2"; shift 2 ;;
    --rps) RPS="$2"; shift 2 ;;
    --duration) DURATION="$2"; shift 2 ;;
    --output-tokens) OUTPUT_TOKENS="$2"; shift 2 ;;
    --tokenizer) TOKENIZER="$2"; shift 2 ;;
    --shared-prefix-frac) SHARED_PREFIX_FRAC="$2"; shift 2 ;;
    --kv-tokens) KV_TOKENS="$2"; shift 2 ;;
    --max-running) MAX_RUNNING="$2"; shift 2 ;;
    --decode-per-req-ms) DECODE_PER_REQ_MS="$2"; shift 2 ;;
    --gw-port) GW_PORT="$2"; shift 2 ;;
    --no-build) BUILD=0; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

case "$MODE" in grpc|http) ;; *) echo "--mode must be grpc|http" >&2; exit 2 ;; esac

ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
if [[ "$BUILD" -eq 1 ]]; then
  echo "==> building smg + mock-worker (release)"
  RUSTC_WRAPPER="" cargo build --release -p smg -p mock-worker
fi
SMG="$CARGO_TARGET_DIR/release/smg"
MOCK="$CARGO_TARGET_DIR/release/mock-worker"
[[ -x "$SMG" && -x "$MOCK" ]] || { echo "binaries missing (drop --no-build)" >&2; exit 1; }

LOGDIR="${TMPDIR:-/tmp}/smg-sim"
RESULTS="${TMPDIR:-/tmp}/smg-sim/results"
mkdir -p "$LOGDIR" "$RESULTS"
rm -f "$RESULTS"/*.json
echo "==> logs: $LOGDIR  results: $RESULTS  mode: $MODE  workers: $WORKERS"

GW_PID=""; MOCK_PID=""
cleanup() {
  [[ -n "$MOCK_PID" ]] && kill "$MOCK_PID" 2>/dev/null || true
  [[ -n "$GW_PID" ]] && kill "$GW_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

start_fleet() {
  # Fresh fleet per policy → a clean prefix cache (no cross-policy carryover).
  local args=(--model "$MODEL" --engine realistic
    --prefill-tps "$PREFILL_TPS" --decode-base-ms "$DECODE_BASE_MS"
    --decode-per-req-ms "$DECODE_PER_REQ_MS" --max-running "$MAX_RUNNING"
    --kv-tokens "$KV_TOKENS" --block-size "$BLOCK_SIZE" --output-tokens "$OUTPUT_TOKENS")
  if [[ "$MODE" == grpc ]]; then
    args+=(--grpc-base-port "$GRPC_BASE" --grpc-count "$WORKERS")
  else
    args+=(--http-base-port "$HTTP_BASE" --http-count "$WORKERS")
  fi
  "$MOCK" "${args[@]}" >"$LOGDIR/mock.log" 2>&1 &
  MOCK_PID=$!
  sleep 2
}

register_workers() {
  local W="http://127.0.0.1:$GW_PORT/workers"
  local H='"health":{"disable_health_check":true}'
  if [[ "$MODE" == grpc ]]; then
    # runtime=tokenspeed; per-worker tokenizer label so the gateway tokenizes
    # mock-model; kv_block_size seeds the cache-aware block size.
    seq "$GRPC_BASE" $((GRPC_BASE + WORKERS - 1)) | xargs -P 16 -I{} \
      curl -s -o /dev/null --max-time 20 -X POST "$W" -H 'content-type: application/json' \
      -d "{\"url\":\"grpc://127.0.0.1:{}\",\"connection_mode\":\"grpc\",\"runtime\":\"tokenspeed\",\"models\":[{\"id\":\"$MODEL\"}],\"labels\":{\"tokenizer_path\":\"$TOKENIZER\"},\"kv_block_size\":$BLOCK_SIZE,$H}" || true
  else
    seq "$HTTP_BASE" $((HTTP_BASE + WORKERS - 1)) | xargs -P 16 -I{} \
      curl -s -o /dev/null --max-time 20 -X POST "$W" -H 'content-type: application/json' \
      -d "{\"url\":\"http://127.0.0.1:{}\",\"connection_mode\":\"http\",\"runtime\":\"sglang\",\"models\":[{\"id\":\"$MODEL\"}],$H}" || true
  fi
}

run_policy() {
  local policy="$1"
  echo "==> policy: $policy"
  pkill -9 -x mock-worker 2>/dev/null || true
  pkill -9 -x smg 2>/dev/null || true
  sleep 2

  start_fleet

  local gw_args=(--host 127.0.0.1 --port "$GW_PORT" --enable-igw --policy "$policy")
  # NOTE: tokenizer autoload stays ENABLED in grpc mode — the sim needs real
  # tokenization for token-aware routing. (scale_test.sh disables it because it
  # only measures registration/routing cost, not generation.)
  "$SMG" "${gw_args[@]}" >"$LOGDIR/gateway-$policy.log" 2>&1 &
  GW_PID=$!

  for _ in $(seq 1 60); do
    curl -sf "http://127.0.0.1:$GW_PORT/health" >/dev/null 2>&1 && break
    sleep 1
  done
  register_workers

  # Give workers time to reach Ready (and, in grpc mode, the tokenizer to load).
  sleep 5
  # Use /v1/completions (raw prompt, no chat template) so any tokenizer works —
  # base tokenizers like gpt2 have no chat_template. (For chat, start the gateway
  # with --chat-template and switch sim_load.py to --endpoint chat.)
  local warm
  warm=$(curl -s --max-time 30 "http://127.0.0.1:$GW_PORT/v1/completions" \
    -H 'content-type: application/json' \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"warmup\",\"max_tokens\":4}" || true)
  if ! grep -q '"choices"' <<<"$warm"; then
    echo "    WARN: warmup failed for $policy (tokenizer not ready in grpc mode?)" >&2
    echo "    response: ${warm:0:200}" >&2
  fi

  python3 "$ROOT/scripts/sim_load.py" \
    --url "http://127.0.0.1:$GW_PORT/v1/completions" --endpoint completions --model "$MODEL" \
    --rps "$RPS" --duration "$DURATION" --output-tokens "$OUTPUT_TOKENS" \
    --shared-prefix-words "$SHARED_PREFIX_WORDS" --shared-prefix-frac "$SHARED_PREFIX_FRAC" \
    --label "$policy" --json "$RESULTS/$policy.json" || true

  kill "$GW_PID" 2>/dev/null || true; GW_PID=""
  kill "$MOCK_PID" 2>/dev/null || true; MOCK_PID=""
  sleep 2
}

for p in $POLICIES; do
  run_policy "$p"
done

echo
echo "================ A/B comparison (mode=$MODE, workers=$WORKERS, rps=$RPS, ${DURATION}s) ================"
python3 - "$RESULTS" <<'PY'
import glob, json, os, sys
results_dir = sys.argv[1]
rows = []
for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
    with open(path) as f:
        rows.append(json.load(f))
if not rows:
    print("no results"); sys.exit(0)
hdr = ["policy", "ok", "err", "rps", "out_tps",
       "TTFT_p50", "TTFT_p90", "TTFT_p99", "ITL_p50", "E2E_p50", "E2E_p99"]
print("{:<14}{:>7}{:>6}{:>7}{:>9}{:>10}{:>10}{:>10}{:>9}{:>10}{:>10}".format(*hdr))
for r in rows:
    print("{:<14}{:>7}{:>6}{:>7}{:>9}{:>10}{:>10}{:>10}{:>9}{:>10}{:>10}".format(
        r.get("label", "?"), r["ok"], r["err"], r["achieved_rps"], r["output_tps"],
        r["ttft_ms_p50"], r["ttft_ms_p90"], r["ttft_ms_p99"],
        r["itl_ms_p50"], r["e2e_ms_p50"], r["e2e_ms_p99"]))
print("\n(ms unless noted; lower TTFT/ITL/E2E is better; cache_aware should win")
print(" TTFT on shared-prefix workloads, least_load should tighten the tail.)")
PY
echo "==> done"
