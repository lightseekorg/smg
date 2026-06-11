#!/bin/bash
# Install vLLM with flash-attn for CI
# Handles CUDA toolkit setup and flash-attn compilation
# Uses uv for faster package installation

set -euo pipefail

# Activate venv if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Install uv for faster package management (10-100x faster than pip)
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Using uv version: $(uv --version)"

# Floor 0.22.1: older vllm resolved an early transformers v5 that broke
# e5-mistral last-token pooling (the old <0.19.1 pin); 0.22.1+ only admits
# transformers >= 5.5.1. e2e-1gpu-embeddings is the quality gate.
# --torch-backend=auto matches the torch CUDA variant to the pod's driver.
echo "Installing vLLM..."
uv pip install "vllm>=0.22.1" --torch-backend=auto

# NIXL for vLLM PD disaggregation. The bare metapackage pulls both cu12 and
# cu13 backends, so install the top-level shim alone, then the backend
# matching torch's CUDA (same normalization as vLLM's own CI).
echo "Installing nixl..."
CUDA_MAJOR=$(python3 -c "import torch; print(torch.version.cuda.split('.')[0])")
uv pip install --no-deps "nixl>=1.2.0"
uv pip install "nixl-cu${CUDA_MAJOR}>=1.2.0"

# Remove nixl_ep (MoE all-to-all, unused in CI): vLLM imports it eagerly when
# present, tying every worker startup to its extra native deps
SITE_PACKAGES=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['platlib'])")
rm -rf "${SITE_PACKAGES}/nixl_ep"

# Import canary: fail here (not mid-e2e) if the nixl install is broken
# (torch first so its bundled CUDA libraries are loaded)
python3 -c "import torch, nixl"
echo "nixl import canary OK"

# Mooncake transfer engine, only on the MooncakeConnector PD leg so a broken
# wheel cannot fail the unrelated vLLM jobs
if [ "${E2E_VLLM_KV_BACKEND:-nixl}" = "mooncake" ]; then
    echo "Installing mooncake-transfer-engine..."
    uv pip install "mooncake-transfer-engine>=0.3.8"

    # The mooncake wheel links libcudart.so.12 regardless of the torch CUDA
    # flavor; provide it from the cu12 runtime wheel (venv + job env only —
    # the wheel ships no unversioned libcudart.so, so cu13 lookups are unaffected)
    uv pip install nvidia-cuda-runtime-cu12
    CUDART12_DIR=$(python3 -c "import glob, sysconfig; print(glob.glob(sysconfig.get_paths()['platlib'] + '/nvidia/cuda_runtime/lib/libcudart.so.12')[0].rsplit('/', 1)[0])")
    export LD_LIBRARY_PATH="${CUDART12_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >> "${GITHUB_ENV:-/dev/null}"

    # Import canary: fail here (not mid-e2e) if the mooncake install is broken —
    # vLLM swallows this ImportError at module load (torch first for CUDA libs)
    python3 -c "import torch; from mooncake.engine import TransferEngine"
    echo "mooncake import canary OK"
fi

# FlashInfer JIT cache: vLLM JIT-compiles flashinfer kernels at engine startup
# and the pods have no CUDA toolchain — install the precompiled cache instead,
# same recipe as vLLM's own Dockerfile.
echo "Installing flashinfer-jit-cache..."
CUDA_TAG=$(python3 -c "import torch; print(torch.version.cuda.replace('.', ''))")
FLASHINFER_VERSION=$(python3 -c "import importlib.metadata as m; print(m.version('flashinfer-python'))")
uv pip install "flashinfer-jit-cache==${FLASHINFER_VERSION}" \
    --index-url "https://flashinfer.ai/whl/cu${CUDA_TAG}"

# Install gRPC packages from source (not PyPI) so PR changes are always tested
echo "Installing smg-grpc-proto and smg-grpc-servicer from source..."
uv pip install -e crates/grpc_client/python/
uv pip install -e grpc_servicer/

echo "vLLM installation complete"
