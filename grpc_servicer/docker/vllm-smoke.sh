#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-install}"
WORKSPACE="${WORKSPACE:-/workspace}"
VENV_DIR="${VENV_DIR:-/tmp/vllm-grpc-smoke-venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VLLM_SPEC="${VLLM_SPEC:-vllm}"
SKIP_VLLM_INSTALL="${SKIP_VLLM_INSTALL:-0}"
SMOKE_HOST="${SMOKE_HOST:-127.0.0.1}"
SMOKE_PORT="${SMOKE_PORT:-50051}"
SMOKE_KV_ENDPOINT="${SMOKE_KV_ENDPOINT:-tcp://*:5557}"
SMOKE_KV_REPLAY_ENDPOINT="${SMOKE_KV_REPLAY_ENDPOINT:-tcp://*:5558}"
SMOKE_KV_BUFFER_STEPS="${SMOKE_KV_BUFFER_STEPS:-10000}"
SMOKE_KV_TOPIC="${SMOKE_KV_TOPIC:-}"
SMOKE_SERVER_TIMEOUT_SECS="${SMOKE_SERVER_TIMEOUT_SECS:-180}"
INSTALL_TOOL="${SMOKE_INSTALL_TOOL:-auto}"

cd "${WORKSPACE}"

rm -rf "${VENV_DIR}"

VENV_ARGS=()
if [[ "${SKIP_VLLM_INSTALL}" == "1" ]]; then
    VENV_ARGS+=(--system-site-packages)
fi

if [[ "${INSTALL_TOOL}" == "uv" || "${INSTALL_TOOL}" == "auto" ]] \
    && command -v uv >/dev/null 2>&1 \
    && uv venv --python "${PYTHON_BIN}" --seed "${VENV_DIR}" "${VENV_ARGS[@]}"; then
    INSTALL_TOOL="uv"
else
    INSTALL_TOOL="pip"
    "${PYTHON_BIN}" -m venv "${VENV_ARGS[@]}" "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

if [[ "${INSTALL_TOOL}" == "pip" ]]; then
    python -m pip install --upgrade pip setuptools wheel
fi

install_pkg() {
    if [[ "${INSTALL_TOOL}" == "uv" ]]; then
        uv pip install "$@"
    else
        python -m pip install "$@"
    fi
}

echo "==> Using installer: ${INSTALL_TOOL}"

if [[ "${SKIP_VLLM_INSTALL}" == "1" ]]; then
    echo "==> Reusing vLLM from base image"
else
    echo "==> Installing vLLM from ${VLLM_SPEC}"
    install_pkg "${VLLM_SPEC}"
fi

echo "==> Installing local gRPC packages from source"
install_pkg -e crates/grpc_client/python/
install_pkg -e grpc_servicer/

echo "==> Verifying imports"
python - <<'PY'
import importlib
modules = [
    "vllm",
    "smg_grpc_servicer",
    "smg_grpc_servicer.vllm.servicer",
]
for name in modules:
    importlib.import_module(name)
    print(f"import ok: {name}")
PY

echo "==> Installing test dependencies"
install_pkg pytest

echo "==> Verifying vLLM gRPC entrypoint is present"
python - <<'PY'
import importlib.util
spec = importlib.util.find_spec("vllm.entrypoints.grpc_server")
assert spec is not None, "vllm.entrypoints.grpc_server not found in installed vllm"
print(f"entrypoint ok: vllm.entrypoints.grpc_server ({spec.origin})")
PY

echo "==> Running KV cache gRPC tests"
python -m pytest grpc_servicer/tests/ -v --tb=short -m "not gpu"

if [[ "${MODE}" == "install" ]]; then
    echo "install smoke test completed successfully"
    exit 0
fi

if [[ "${MODE}" != "serve" ]]; then
    echo "unknown smoke mode: ${MODE}" >&2
    exit 2
fi

if [[ -z "${SMOKE_MODEL:-}" ]]; then
    echo "SMOKE_MODEL is required for serve mode" >&2
    exit 2
fi

KV_EVENTS_CONFIG=$(
    python - <<'PY'
import json
import os

print(
    json.dumps(
        {
            "enable_kv_cache_events": True,
            "publisher": "zmq",
            "endpoint": os.environ["SMOKE_KV_ENDPOINT"],
            "replay_endpoint": os.environ["SMOKE_KV_REPLAY_ENDPOINT"],
            "buffer_steps": int(os.environ["SMOKE_KV_BUFFER_STEPS"]),
            "topic": os.environ["SMOKE_KV_TOPIC"],
        }
    )
)
PY
)

SERVER_LOG=/tmp/vllm-grpc-server.log
read -r -a EXTRA_ARGS <<<"${VLLM_EXTRA_ARGS:-}"

SERVER_CMD=(
    python
    -m
    vllm.entrypoints.grpc_server
    --model
    "${SMOKE_MODEL}"
    --host
    "${SMOKE_HOST}"
    --port
    "${SMOKE_PORT}"
    --kv-events-config
    "${KV_EVENTS_CONFIG}"
)
SERVER_CMD+=("${EXTRA_ARGS[@]}")

echo "==> Starting vLLM gRPC server"
printf 'command:'
printf ' %q' "${SERVER_CMD[@]}"
printf '\n'
"${SERVER_CMD[@]}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

cleanup() {
    if kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
        kill "${SERVER_PID}" >/dev/null 2>&1 || true
        wait "${SERVER_PID}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

python - <<'PY'
import os
import socket
import sys
import time

host = os.environ["SMOKE_HOST"]
port = int(os.environ["SMOKE_PORT"])
timeout_secs = int(os.environ["SMOKE_SERVER_TIMEOUT_SECS"])
deadline = time.time() + timeout_secs

while time.time() < deadline:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        if sock.connect_ex((host, port)) == 0:
            print(f"server port opened on {host}:{port}")
            sys.exit(0)
    time.sleep(1)

print(f"timed out waiting for {host}:{port}", file=sys.stderr)
sys.exit(1)
PY

echo "serve smoke test completed successfully"
