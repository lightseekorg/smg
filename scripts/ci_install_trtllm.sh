#!/bin/bash
# Install TensorRT-LLM with gRPC support for CI
#
# gRPC server support (PR #11037) is not yet in a pip release,
# so we install the latest stable wheel (which has all compiled binaries)
# and then overlay the Python source from main branch on top. This gives
# us the gRPC serve command from main + compiled C++/CUDA libs from stable.
#
# Prerequisites (expected on k8s-runner-gpu nodes):
#   - NVIDIA driver 580+ (CUDA 13)
#   - CUDA toolkit at /usr/local/cuda
#   - libopenmpi-dev
#
# At runtime we use --backend pytorch, which avoids TRT engine compilation.

set -euo pipefail

# Activate venv if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# System dependencies
sudo apt-get update
sudo apt-get install -y libopenmpi-dev

# Set CUDA_HOME and LD_LIBRARY_PATH from system CUDA
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA version: $(cat ${CUDA_HOME}/version.json 2>/dev/null | python3 -c 'import sys,json; print(json.load(sys.stdin)["cuda"]["version"])' 2>/dev/null || nvcc --version 2>/dev/null | tail -1 || echo 'unknown')"

# Install PyTorch with CUDA support
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Step 1: Install the latest stable TRT-LLM wheel from NVIDIA PyPI.
echo "Installing stable TensorRT-LLM wheel..."
pip install --no-cache-dir tensorrt_llm --extra-index-url=https://pypi.nvidia.com

# Step 2: Add pip-installed NVIDIA libraries to LD_LIBRARY_PATH.
# TRT-LLM pulls in nvidia-* pip packages with .so files that aren't
# on the default search path.
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")

NVIDIA_LIB_DIRS=$(find "$SITE_PACKAGES/nvidia" -name "lib" -type d 2>/dev/null | sort -u | paste -sd':')
if [ -n "$NVIDIA_LIB_DIRS" ]; then
    export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}:${LD_LIBRARY_PATH:-}"
fi

TRTLLM_LIB_DIR=$(find "$SITE_PACKAGES" -path "*/tensorrt_llm/libs" -type d 2>/dev/null | head -1)
if [ -n "$TRTLLM_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="${TRTLLM_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# Step 3: Clone main branch for gRPC serve command (not in any release yet).
TRTLLM_DIR="/tmp/tensorrt-llm-src"
if [ ! -d "$TRTLLM_DIR" ]; then
    echo "Cloning TensorRT-LLM main branch for gRPC source..."
    git clone --depth 1 https://github.com/NVIDIA/TensorRT-LLM.git "$TRTLLM_DIR"
fi

# Step 4: Overlay Python source from main branch onto the installed package.
# This replaces the Python files (including adding gRPC serve command) while
# preserving all compiled .so/.pyd binaries from the stable wheel.
INSTALL_DIR=$(python3 -c "import tensorrt_llm, pathlib; print(pathlib.Path(tensorrt_llm.__file__).parent)")
echo "Installed package at: $INSTALL_DIR"
echo "Overlaying main branch Python source..."

rsync -a \
    --include='*.py' \
    --include='*/' \
    --exclude='*.so' \
    --exclude='*.pyd' \
    --exclude='bindings/' \
    --exclude='libs/' \
    "$TRTLLM_DIR/tensorrt_llm/" "$INSTALL_DIR/"

# Persist LD_LIBRARY_PATH for subsequent CI steps
if [ -n "${GITHUB_ENV:-}" ]; then
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> "$GITHUB_ENV"
fi

echo "TensorRT-LLM installation complete"
python3 -c "import tensorrt_llm; print(f'TensorRT-LLM version: {tensorrt_llm.__version__}')"
python3 -c "from tensorrt_llm.commands.serve import main; print('gRPC serve command: available')"
