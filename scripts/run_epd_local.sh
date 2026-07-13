#!/bin/bash
# Bring up TokenSpeed EPD (encode-prefill-decode) multimodal disaggregation on a
# single local GPU-RDMA node (GB200 / GB300 / H100) for reproduction and dev, and
# drive the EPD e2e against it. This mirrors what the 4-GPU CI lane does, but
# builds everything from local source so you can iterate in minutes instead of
# 30-min CI rounds.
#
# It exists because EPD-over-mooncake needs a few environment tweaks that aren't
# obvious (see the ENV section): the mooncake transfer engine must use the dmabuf
# GPUDirect path on the NVIDIA open kernel driver, and its runtime deps
# (libnuma) may not be on the loader path in a torch-only env.
#
# Usage:
#   scripts/run_epd_local.sh install      # build + install the tokenspeed stack + gateway
#   scripts/run_epd_local.sh model        # download the EPD model
#   scripts/run_epd_local.sh run          # run the EPD e2e (default topology 1e1p1d)
#   scripts/run_epd_local.sh all          # install + model + run
#
# Override anything via env, e.g. TORCH_PY=/path/to/python EPD_MODEL=... TS_SRC=...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMG_SRC="${SMG_SRC:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
# TokenSpeed source checkout (engine + kernel + scheduler). Defaults next to smg.
TS_SRC="${TS_SRC:-$(cd "${SMG_SRC}/.." && pwd)/tokenspeed}"
# A python that already provides a CUDA-13 torch for this box's arch. On a GB300
# dev box that's typically a conda env; point TORCH_PY at its bin/python.
TORCH_PY="${TORCH_PY:?set TORCH_PY to a python that has a matching CUDA-13 torch}"
VENV="${VENV:-${SMG_SRC}/.epd-local-venv}"
EPD_MODEL="${EPD_MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"
MODEL_ROOT="${MODEL_ROOT:-/models}"
EPD_TOPOLOGY="${EPD_TOPOLOGY:-1e1p1d}"   # 1e1p1d | 1e2p1d | 2e1p1d | 1e1p2d

# ── CUDA toolkit: pick a complete toolkit that matches torch's CUDA ───────────
if [ -z "${CUDA_HOME:-}" ]; then
    for c in /usr/local/cuda-13.1 /usr/local/cuda-13.0 /usr/local/cuda; do
        if [ -x "$c/bin/nvcc" ] && [ -f "$c/include/crt/host_runtime.h" ]; then CUDA_HOME="$c"; break; fi
    done
fi
CUDA_HOME="${CUDA_HOME:?no complete CUDA toolkit found; set CUDA_HOME}"

py() { "${VENV}/bin/python" "$@"; }

detect_arch() {
    # e.g. GB300 -> "10.3a", H100 -> "9.0a". FlashInfer wants <major>.<minor>a.
    "$TORCH_PY" -c 'import torch; a,b=torch.cuda.get_device_capability(); print(f"{a}.{b}a")'
}

# ── Env every EPD process needs (the non-obvious bits) ───────────────────────
epd_env() {
    # dmabuf GPUDirect: mooncake defaults to legacy nvidia_peermem, which fails to
    # register GPU memory on the NVIDIA open kernel driver. Force dmabuf.
    echo "WITH_NVIDIA_PEERMEM=false"
    # mooncake links libnuma at runtime; a torch-only env often lacks it on the
    # loader path. Preload the system copy if present.
    [ -e /usr/lib64/libnuma.so.1 ] && echo "LD_PRELOAD=/usr/lib64/libnuma.so.1:${LD_PRELOAD:-}"
    # Same-node encode<->prefill lives on one host; use the NVLink IPC intranode
    # transport rather than RoCE loopback (which the fabric may not shortcut).
    echo "MC_INTRANODE_NVLINK=1"
    echo "CUDA_HOME=${CUDA_HOME}"
}

cmd_install() {
    local arch; arch="$(detect_arch)"
    echo ">>> building EPD stack: venv=${VENV} CUDA_HOME=${CUDA_HOME} arch=${arch}"
    # Clean venv, NOT --system-site-packages: inheriting a torch-2.12 conda env
    # breaks its prebuilt C-extensions (torchcomms, ...) as soon as the kernel
    # pins torch 2.11. A fresh env with torch 2.11+cu130 avoids the clash and
    # still runs on sm_103 (verified on GB300).
    # --seed installs pip/setuptools/wheel: tokenspeed-kernel's setup.py shells
    # out to `python -m pip install -r requirements/cuda.txt`, so the venv needs pip.
    uv venv --python "${VENV_PYTHON:-3.12}" --seed "$VENV"

    export CUDA_HOME PATH="${CUDA_HOME}/bin:${PATH}"
    export MAX_JOBS="${MAX_JOBS:-32}" FLASHINFER_CUDA_ARCH_LIST="$arch" TOKENSPEED_KERNEL_BACKEND=cuda
    # Resolve torch + nvidia cu13 wheels from the pytorch cu130 index.
    export UV_EXTRA_INDEX_URL="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
    export PIP_EXTRA_INDEX_URL="$UV_EXTRA_INDEX_URL"
    # The kernel's setup.py shells out to plain pip; without a socket timeout a
    # stalled download hangs forever. Fail fast (30s no-data) and retry instead.
    export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-30}" PIP_RETRIES="${PIP_RETRIES:-10}"
    # cutlass pin: 4.6.0 dropped cute.core.ThrMma that quack needs (see CI script).
    local con; con="$(mktemp)"; echo "nvidia-cutlass-dsl==4.5.2" > "$con"
    export UV_CONSTRAINT="$con" PIP_CONSTRAINT="$con"

    uv pip install --python "${VENV}/bin/python" setuptools wheel pybind11
    # The CUDA-13 torch the kernel expects, up front in the clean env.
    uv pip install --python "${VENV}/bin/python" "torch==${TORCH_VERSION:-2.11.0}+cu130"
    # TokenSpeed: kernel (from source) -> scheduler -> engine. Same order as CI.
    uv pip install --python "${VENV}/bin/python" -e "${TS_SRC}/tokenspeed-kernel/python/" --no-build-isolation
    uv pip install --python "${VENV}/bin/python" -e "${TS_SRC}/tokenspeed-scheduler/"
    uv pip install --python "${VENV}/bin/python" -e "${TS_SRC}/python" --no-build-isolation

    # smg gRPC proto + servicer from source (the EPD encode servicer lives here).
    uv pip uninstall --python "${VENV}/bin/python" tokenspeed-smg-grpc-proto tokenspeed-smg-grpc-servicer 2>/dev/null || true
    uv pip install --python "${VENV}/bin/python" -e "${SMG_SRC}/crates/grpc_client/python/"
    uv pip install --python "${VENV}/bin/python" -e "${SMG_SRC}/grpc_servicer/"
    # smg gateway (the model_gateway python wheel) via maturin.
    uv pip install --python "${VENV}/bin/python" maturin
    (cd "$SMG_SRC" && "${VENV}/bin/maturin" develop --release -m bindings/python/Cargo.toml)

    echo ">>> install complete. verify:"
    env $(epd_env) py -c "import tokenspeed, tokenspeed_kernel, mooncake.engine, smg_grpc_servicer, smg; print('EPD stack import OK')"
}

cmd_model() {
    echo ">>> downloading ${EPD_MODEL} -> ${MODEL_ROOT}"
    ROUTER_LOCAL_MODEL_PATH="$MODEL_ROOT" bash "${SMG_SRC}/scripts/ci_download_model.sh" "$EPD_MODEL"
}

cmd_run() {
    echo ">>> running EPD e2e (topology ${EPD_TOPOLOGY}) with dmabuf + NVLink IPC"
    cd "$SMG_SRC"
    env $(epd_env) \
        E2E_ENGINE=tokenspeed E2E_RUNTIME=tokenspeed E2E_GPU_TIER=4 \
        ROUTER_LOCAL_MODEL_PATH="$MODEL_ROOT" SHOW_WORKER_LOGS=1 \
        "${VENV}/bin/python" -m pytest e2e_test/chat_completions/test_epd_multimodal.py \
        -k "$EPD_TOPOLOGY" -s -vv
}

case "${1:-all}" in
    install) cmd_install ;;
    model)   cmd_model ;;
    run)     cmd_run ;;
    all)     cmd_install; cmd_model; cmd_run ;;
    env)     epd_env ;;
    *) echo "usage: $0 {install|model|run|all|env}" >&2; exit 2 ;;
esac
