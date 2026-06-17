#!/usr/bin/env bash
# Launch one "arm" of the BFCL A/B comparison.
#
#   arm A = pure vLLM OpenAI server  (vLLM owns chat template + tokenization +
#           tool/reasoning parsing)
#   arm B = SMG in front of a vLLM gRPC worker (SMG owns chat template +
#           tokenization + tool/reasoning parsing; vLLM runs raw-token)
#
# Both expose an identical OpenAI /v1 endpoint, so the official BFCL harness can
# point at either and the ONLY thing that differs is the frontend — which is
# exactly the variable the A/B isolates.
#
# Everything is parameterised via env vars so the same script works on the H100
# box and in CI. Writes a pidfile + log per process so run_ab.py / the nightly
# can manage lifecycle and tear down cleanly.
#
# Usage:
#   launch_arm.sh a            # start pure-vLLM arm, print its base_url
#   launch_arm.sh b            # start vLLM-gRPC + SMG arm, print its base_url
#   launch_arm.sh stop         # kill anything this script started (via pidfiles)
set -euo pipefail

ARM="${1:?usage: launch_arm.sh <a|b|stop>}"

MODEL="${BFCL_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
GPU="${BFCL_GPU:-0}"                              # CUDA_VISIBLE_DEVICES (e.g. "0" or "0,1")
TP="${BFCL_TP:-1}"                               # tensor-parallel size (match GPU count)
MAX_MODEL_LEN="${BFCL_MAX_MODEL_LEN:-16384}"
GPU_MEM_UTIL="${BFCL_GPU_MEM_UTIL:-0.55}"
RUN_DIR="${BFCL_RUN_DIR:-/tmp/bfcl_ab}"

# Pure-vLLM (arm A) tool/reasoning parser flags.
VLLM_TOOL_PARSER="${BFCL_VLLM_TOOL_PARSER:-hermes}"
VLLM_REASONING_PARSER="${BFCL_VLLM_REASONING_PARSER:-}"   # empty = none (non-thinking SKU)
# SMG (arm B) parser flags — SMG registry names, NOT vLLM's.
SMG_TOOL_PARSER="${BFCL_SMG_TOOL_PARSER:-qwen}"
SMG_REASONING_PARSER="${BFCL_SMG_REASONING_PARSER:-}"

# Extra args appended to every vLLM process (both arms). e.g.
# BFCL_VLLM_EXTRA="--enforce-eager" — skips CUDA-graph capture, which has been
# more stable under sustained bfcl load on shared/contended GPUs.
VLLM_EXTRA="${BFCL_VLLM_EXTRA:-}"

# Ports. Default to an OS-assigned free port (resolved per-arm in the case
# below) so concurrent arms/jobs sharing a host — e.g. bin-packed CI runners
# under hostNetwork — don't collide on a fixed port. Pin via env for a stable
# port on a dev box.
ARM_A_PORT="${BFCL_ARM_A_PORT:-}"          # pure-vLLM OpenAI port
ARM_B_GRPC_PORT="${BFCL_ARM_B_GRPC_PORT:-}" # vLLM gRPC worker port
ARM_B_GW_PORT="${BFCL_ARM_B_GW_PORT:-}"     # SMG OpenAI gateway port

# Executables (override for venv / box paths).
VLLM_BIN="${VLLM_BIN:-vllm}"                      # `vllm serve` console script
VLLM_PYTHON="${VLLM_PYTHON:-python}"             # python that can `-m vllm.entrypoints.grpc_server`
SMG_LAUNCH="${SMG_LAUNCH:-smg launch}"           # SMG launcher (binary subcmd or `python -m smg.launch_router`)

mkdir -p "$RUN_DIR"

# start <name> <logfile> <command...> — detached, pidfile-tracked.
start() {
  local name="$1" log="$2"; shift 2
  setsid env "$@" >"$log" 2>&1 </dev/null &
  echo $! >"$RUN_DIR/$name.pid"
  echo "[launch_arm] started $name (pid $(cat "$RUN_DIR/$name.pid")) -> $log" >&2
}

wait_http() {  # wait_http <url> <timeout_s>
  local url="$1" timeout="${2:-300}" waited=0
  until curl -sf -m 3 "$url" >/dev/null 2>&1; do
    sleep 5; waited=$((waited + 5))
    if [ "$waited" -ge "$timeout" ]; then echo "[launch_arm] TIMEOUT waiting for $url" >&2; return 1; fi
  done
}

wait_grpc() {  # crude TCP-listen check for the gRPC port
  local port="$1" timeout="${2:-300}" waited=0
  until (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; do
    sleep 5; waited=$((waited + 5))
    if [ "$waited" -ge "$timeout" ]; then echo "[launch_arm] TIMEOUT waiting for grpc :$port" >&2; return 1; fi
  done
  exec 3>&- 2>/dev/null || true
}

free_port() {  # OS-assigned free TCP port (same idiom as the e2e infra's get_open_port)
  python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'
}

case "$ARM" in
  a)
    ARM_A_PORT="${ARM_A_PORT:-$(free_port)}"
    declare -a cmd=(
      CUDA_VISIBLE_DEVICES="$GPU" "$VLLM_BIN" serve "$MODEL"
      --served-model-name "$MODEL"
      --enable-auto-tool-choice --tool-call-parser "$VLLM_TOOL_PARSER"
      --host 0.0.0.0 --port "$ARM_A_PORT"
      --tensor-parallel-size "$TP" --max-model-len "$MAX_MODEL_LEN"
      --gpu-memory-utilization "$GPU_MEM_UTIL"
    )
    [ -n "$VLLM_REASONING_PARSER" ] && cmd+=(--reasoning-parser "$VLLM_REASONING_PARSER")
    # shellcheck disable=SC2206  # intentional word-split of optional extra flags
    [ -n "$VLLM_EXTRA" ] && cmd+=($VLLM_EXTRA)
    start arm_a "$RUN_DIR/arm_a_vllm.log" "${cmd[@]}"
    wait_http "http://127.0.0.1:$ARM_A_PORT/health" "${BFCL_STARTUP_TIMEOUT:-420}"
    echo "http://127.0.0.1:$ARM_A_PORT"
    ;;

  b)
    ARM_B_GRPC_PORT="${ARM_B_GRPC_PORT:-$(free_port)}"
    ARM_B_GW_PORT="${ARM_B_GW_PORT:-$(free_port)}"
    # 1) vLLM gRPC worker (raw-token; SMG will own template+parsing).
    declare -a wcmd=(
      CUDA_VISIBLE_DEVICES="$GPU" "$VLLM_PYTHON" -m vllm.entrypoints.grpc_server
      --model "$MODEL" --host 0.0.0.0 --port "$ARM_B_GRPC_PORT"
      --tensor-parallel-size "$TP" --max-model-len "$MAX_MODEL_LEN"
      --gpu-memory-utilization "$GPU_MEM_UTIL"
    )
    # shellcheck disable=SC2206  # intentional word-split of optional extra flags
    [ -n "$VLLM_EXTRA" ] && wcmd+=($VLLM_EXTRA)
    start arm_b_worker "$RUN_DIR/arm_b_worker.log" "${wcmd[@]}"
    wait_grpc "$ARM_B_GRPC_PORT" "${BFCL_STARTUP_TIMEOUT:-420}"
    # 2) SMG gateway in front, exposing the OpenAI API.
    declare -a smg_cmd=(
      $SMG_LAUNCH
      --model-path "$MODEL"
      --worker-urls "grpc://127.0.0.1:$ARM_B_GRPC_PORT"
      --tool-call-parser "$SMG_TOOL_PARSER"
      --host 0.0.0.0 --port "$ARM_B_GW_PORT"
    )
    [ -n "$SMG_REASONING_PARSER" ] && smg_cmd+=(--reasoning-parser "$SMG_REASONING_PARSER")
    start arm_b_gateway "$RUN_DIR/arm_b_gateway.log" "${smg_cmd[@]}"
    wait_http "http://127.0.0.1:$ARM_B_GW_PORT/health" "${BFCL_STARTUP_TIMEOUT:-420}"
    echo "http://127.0.0.1:$ARM_B_GW_PORT"
    ;;

  stop)
    for pf in "$RUN_DIR"/*.pid; do
      [ -e "$pf" ] || continue
      pid="$(cat "$pf")"
      kill "$pid" 2>/dev/null && echo "[launch_arm] killed $(basename "$pf" .pid) (pid $pid)" >&2 || true
      rm -f "$pf"
    done
    ;;

  *)
    echo "usage: launch_arm.sh <a|b|stop>" >&2; exit 2;;
esac
