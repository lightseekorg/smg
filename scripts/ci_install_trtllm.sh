#!/bin/bash
# Install TensorRT-LLM with gRPC support for CI
#
# gRPC server support (PR #11037) is not yet in a pip release,
# so we install the latest stable wheel (which has all compiled binaries)
# and then overlay the Python source from main branch on top. This gives
# us the gRPC serve command from main + compiled C++/CUDA libs from stable.
# No Docker or C++ build toolchain required.
#
# Prerequisites (expected on k8s-runner-gpu H100 nodes):
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

# Install PyTorch with CUDA support (TRT-LLM requires torch with CUDA)
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Step 1: Install the latest stable TRT-LLM wheel from NVIDIA PyPI.
# This gives us all compiled binaries (bindings, .so files, CUDA kernels).
echo "Installing stable TensorRT-LLM wheel..."
pip install --no-cache-dir tensorrt_llm --extra-index-url=https://pypi.nvidia.com

# Step 2: Set LD_LIBRARY_PATH for ALL pip-installed NVIDIA/CUDA libraries.
# TRT-LLM pulls in multiple CUDA packages (cuda-runtime 12.x AND 13.x, TensorRT,
# nccl, etc.) whose shared libraries aren't on the default search path.
# We add ALL nvidia `lib` directories to cover every .so dependency.
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")

NVIDIA_LIB_DIRS=$(find "$SITE_PACKAGES/nvidia" -name "lib" -type d 2>/dev/null | sort -u | paste -sd':')
if [ -n "$NVIDIA_LIB_DIRS" ]; then
    export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}:${LD_LIBRARY_PATH:-}"
    echo "Added NVIDIA libs to LD_LIBRARY_PATH:"
    echo "$NVIDIA_LIB_DIRS" | tr ':' '\n' | while read -r d; do echo "  $d"; done
fi

# Also add TRT-LLM's own libs directory (libtensorrt_llm.so etc.)
TRTLLM_LIB_DIR=$(find "$SITE_PACKAGES" -path "*/tensorrt_llm/libs" -type d 2>/dev/null | head -1)
if [ -n "$TRTLLM_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="${TRTLLM_LIB_DIR}:${LD_LIBRARY_PATH:-}"
    echo "Added TRT-LLM libs: $TRTLLM_LIB_DIR"
fi

# Add system CUDA as fallback
if [ -d "/usr/local/cuda/lib64" ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
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

# Copy Python files from git source, preserving compiled binaries
rsync -a \
    --include='*.py' \
    --include='*/' \
    --exclude='*.so' \
    --exclude='*.pyd' \
    --exclude='bindings/' \
    --exclude='libs/' \
    "$TRTLLM_DIR/tensorrt_llm/" "$INSTALL_DIR/"

# Persist LD_LIBRARY_PATH for subsequent CI steps via GITHUB_ENV
if [ -n "${GITHUB_ENV:-}" ]; then
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> "$GITHUB_ENV"
fi

echo "TensorRT-LLM installation complete"
python3 -c "import tensorrt_llm; print(f'TensorRT-LLM version: {tensorrt_llm.__version__}')"
python3 -c "from tensorrt_llm.commands.serve import main; print('gRPC serve command: available')"
