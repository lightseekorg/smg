#!/bin/bash
# Install SGLang for CI

set -euo pipefail

# Activate venv if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

pip install --upgrade pip

# Install SGLang with all dependencies
echo "Installing SGLang..."
pip install "sglang[all]"

# Install flashinfer-jit-cache: sglang bundles flashinfer_python but only for attention ops.
# Multi-GPU models need trtllm_comm kernels (fused allreduce + layernorm) which FlashInfer
# JIT-compiles at runtime requiring nvcc. The jit-cache provides these pre-compiled.
# Version must match flashinfer_python from sglang.
FLASHINFER_VERSION=$(pip show flashinfer-python 2>/dev/null | grep "^Version:" | awk '{print $2}')
CU_VERSION=$(python3 -c "import torch; print('cu' + torch.version.cuda.replace('.', ''))" 2>/dev/null || echo "cu129")
if [ -n "$FLASHINFER_VERSION" ]; then
    echo "Installing flashinfer-jit-cache==${FLASHINFER_VERSION} (${CU_VERSION})..."
    pip install "flashinfer-jit-cache==${FLASHINFER_VERSION}" \
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
pip install mooncake-transfer-engine==0.3.8.post1

echo "SGLang installation complete"
