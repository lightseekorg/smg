#!/bin/bash
# Install SGLang for CI
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

# Install CUDA toolkit (nvcc) — required for SGLang JIT kernel compilation.
# SGLang >= 0.5.9 JIT-compiles CUDA kernels (RoPE, etc.) at runtime via tvm_ffi,
# which invokes nvcc. The CI runners have CUDA runtime (driver) but not the compiler.
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
    echo "Installing CUDA toolkit (nvcc not found at ${CUDA_HOME}/bin/nvcc)..."
    curl -fsSL -o /tmp/cuda-keyring.deb \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i /tmp/cuda-keyring.deb
    rm /tmp/cuda-keyring.deb
    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends cuda-nvcc-12-9 cuda-cudart-dev-12-9
    # Ensure CUDA_HOME points to the installed toolkit
    if [ ! -d "${CUDA_HOME}/bin" ] && [ -d "/usr/local/cuda-12.9/bin" ]; then
        sudo ln -sfn /usr/local/cuda-12.9 "${CUDA_HOME}"
    fi
    echo "nvcc installed: $(${CUDA_HOME}/bin/nvcc --version | tail -1)"
else
    echo "nvcc already available: $(${CUDA_HOME}/bin/nvcc --version | tail -1)"
fi

# Install SGLang with all dependencies
# TODO(changsu): Revert to `uv pip install "sglang[all]"` once sglang releases
# a stable version (>= 0.5.10) that includes the smg_grpc_servicer migration
# (sgl-project/sglang#20478). The rc0 is needed because the old built-in
# grpc_server.py references proto types (EmbedError, EmbedComplete) that were
# removed from our proto definitions.
echo "Installing SGLang..."
uv pip install --prerelease=allow "sglang[all]==0.5.10rc0"

# Install flashinfer-jit-cache: sglang bundles flashinfer_python but only for attention ops.
# Multi-GPU models need trtllm_comm kernels (fused allreduce + layernorm) which FlashInfer
# JIT-compiles at runtime requiring nvcc. The jit-cache provides these pre-compiled.
# Version must match flashinfer_python from sglang.
FLASHINFER_VERSION=$(uv pip show flashinfer-python 2>/dev/null | grep "^Version:" | awk '{print $2}')
CU_VERSION=$(python3 -c "import torch; print('cu' + torch.version.cuda.replace('.', ''))" 2>/dev/null || echo "cu129")
if [ -n "$FLASHINFER_VERSION" ]; then
    echo "Installing flashinfer-jit-cache==${FLASHINFER_VERSION} (${CU_VERSION})..."
    uv pip install "flashinfer-jit-cache==${FLASHINFER_VERSION}" \
        --index-url "https://flashinfer.ai/whl/${CU_VERSION}"
else
    echo "WARNING: flashinfer-python not found, skipping flashinfer-jit-cache install"
fi

# Install mooncake for SGLang PD disaggregation (KV transfer)
# Mooncake's native transfer engine requires InfiniBand/RDMA libraries at runtime.
# See: https://github.com/sgl-project/sglang/blob/main/scripts/ci/cuda/ci_install_dependency.sh
echo "Installing mooncake system dependencies..."
sudo apt-get install -y --no-install-recommends libnuma-dev libibverbs-dev libibverbs1 ibverbs-providers ibverbs-utils
echo "Installing mooncake..."
uv pip install mooncake-transfer-engine==0.3.8.post1

# Install gRPC packages from source (not PyPI) so PR changes are always tested
echo "Installing smg-grpc-proto and smg-grpc-servicer from source..."
uv pip install -e crates/grpc_client/python/
uv pip install -e grpc_servicer/

echo "SGLang installation complete"
