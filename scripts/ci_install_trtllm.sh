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
#   - CUDA toolkit (CUDA_HOME=/usr/local/cuda)
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

# Step 2: Clone main branch for gRPC serve command (not in any release yet).
TRTLLM_DIR="/tmp/tensorrt-llm-src"
if [ ! -d "$TRTLLM_DIR" ]; then
    echo "Cloning TensorRT-LLM main branch for gRPC source..."
    git clone --depth 1 https://github.com/NVIDIA/TensorRT-LLM.git "$TRTLLM_DIR"
fi

# Step 3: Overlay Python source from main branch onto the installed package.
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

echo "TensorRT-LLM installation complete"
python3 -c "import tensorrt_llm; print(f'TensorRT-LLM version: {tensorrt_llm.__version__}')"
python3 -c "from tensorrt_llm.commands.serve import main; print('gRPC serve command: available')"
