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
#   - CUDA 13.1 toolkit at /usr/local/cuda
#   - libopenmpi-dev
#
# At runtime we use --backend pytorch, which avoids TRT engine compilation.

set -euo pipefail

# Activate venv if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# System dependencies
export DEBIAN_FRONTEND=noninteractive
sudo dpkg --configure -a --force-confnew 2>/dev/null || true
sudo apt-get update
sudo apt-get install -y libopenmpi-dev

# ── CUDA setup ───────────────────────────────────────────────────────────────
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"

# Re-activate venv to keep venv Python first in PATH
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Debug: print what CUDA we actually have
echo "=== CUDA diagnostics ==="
echo "CUDA_HOME=$CUDA_HOME"
ls -la "$CUDA_HOME" 2>/dev/null || echo "WARNING: $CUDA_HOME does not exist!"
nvidia-smi 2>/dev/null | head -4 || echo "WARNING: nvidia-smi not found"
nvcc --version 2>/dev/null || echo "WARNING: nvcc not found"
echo "Looking for libcublasLt.so*:"
find /usr/local -name "libcublasLt.so*" 2>/dev/null || echo "  not found in /usr/local"
find /usr -name "libcublasLt.so*" 2>/dev/null | head -5 || echo "  not found in /usr"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"
echo "=== end CUDA diagnostics ==="

# Add CUDA libs to LD_LIBRARY_PATH
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}"

# Install PyTorch with CUDA 13 support (TRT-LLM wheels are built against CUDA 13)
pip install --no-cache-dir torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu130

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

echo "Final LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# Debug: verify libcublasLt.so.13 is findable
echo "Checking libcublasLt.so.13 resolution:"
ldconfig -p 2>/dev/null | grep libcublasLt || echo "  not in ldconfig cache"
find "$SITE_PACKAGES/nvidia" -name "libcublasLt.so*" 2>/dev/null || echo "  not in pip nvidia packages"
find "$CUDA_HOME" -name "libcublasLt.so*" 2>/dev/null || echo "  not in CUDA_HOME"

# Step 3: Clone main branch for gRPC serve command (not in any release yet).
TRTLLM_DIR="/tmp/tensorrt-llm-src"
if [ ! -d "$TRTLLM_DIR" ]; then
    echo "Cloning TensorRT-LLM main branch for gRPC source..."
    git clone --depth 1 https://github.com/NVIDIA/TensorRT-LLM.git "$TRTLLM_DIR"
fi

# Step 4: Overlay Python source from main branch onto the installed package.
# This replaces the Python files (including adding gRPC serve command) while
# preserving all compiled .so/.pyd binaries from the stable wheel.
# TRT-LLM prints a version banner to stdout on import; use tail -1 to get only the path
INSTALL_DIR=$(python3 -c "import tensorrt_llm, pathlib; print(pathlib.Path(tensorrt_llm.__file__).parent)" 2>/dev/null | tail -1)
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

# Step 5: Copy vendored triton_kernels module (required by main branch __init__.py).
# The main branch references triton_kernels at the site-packages level.
if [ -d "$TRTLLM_DIR/triton_kernels" ]; then
    echo "Installing vendored triton_kernels module..."
    cp -r "$TRTLLM_DIR/triton_kernels" "$SITE_PACKAGES/triton_kernels"
fi

# Persist LD_LIBRARY_PATH for subsequent CI steps
if [ -n "${GITHUB_ENV:-}" ]; then
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> "$GITHUB_ENV"
fi

echo "TensorRT-LLM installation complete"
python3 -c "import tensorrt_llm; print(f'TensorRT-LLM version: {tensorrt_llm.__version__}')"
python3 -c "from tensorrt_llm.commands.serve import main; print('gRPC serve command: available')"
