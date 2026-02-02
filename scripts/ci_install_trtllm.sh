#!/bin/bash
# Install TensorRT-LLM with gRPC support for CI
#
# gRPC server support (PR #11037) is not yet in a pip release,
# so we clone main and install with TRTLLM_PRECOMPILED_LOCATION
# pointing at the latest stable wheel. This extracts pre-compiled C++/CUDA
# binaries from the stable wheel and layers the Python source (including gRPC
# server code) from the git checkout on top.
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

# Install PyTorch with CUDA support (TRT-LLM requires torch>=2.9.1 with cu130)
pip install --no-cache-dir torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu130

# Clone TensorRT-LLM (shallow clone to save time)
TRTLLM_DIR="tensorrt-llm"
if [ ! -d "$TRTLLM_DIR" ]; then
    echo "Cloning TensorRT-LLM repository..."
    git clone --depth 1 https://github.com/NVIDIA/TensorRT-LLM.git "$TRTLLM_DIR"
fi

# Download the latest stable TRT-LLM wheel from PyPI/NVIDIA for its compiled binaries.
# Main branch (1.3.0rc2) isn't on PyPI yet, so we grab the latest available release
# and use TRTLLM_PRECOMPILED_LOCATION to extract .so files from it.
WHEEL_DIR="/tmp/trtllm-wheel"
mkdir -p "$WHEEL_DIR"
echo "Downloading latest stable TensorRT-LLM wheel for precompiled binaries..."
pip download tensorrt_llm --dest "$WHEEL_DIR" --no-deps --extra-index-url=https://pypi.nvidia.com

WHEEL_PATH=$(ls "$WHEEL_DIR"/tensorrt_llm-*.whl | head -1)
echo "Using precompiled wheel: $WHEEL_PATH"

# Install using pre-compiled binaries from the downloaded wheel + Python source from git.
# TRTLLM_PRECOMPILED_LOCATION tells setup.py to extract compiled .so files from this
# specific wheel instead of trying to download a version-matched one from PyPI.
echo "Installing TensorRT-LLM (precompiled binaries + main branch Python source)..."
cd "$TRTLLM_DIR"
# NOTE: Do NOT use -e (editable) mode — it requires the C++ `bindings` module
# which needs a full build. Regular install works with precompiled .so files.
TRTLLM_PRECOMPILED_LOCATION="$WHEEL_PATH" pip install --no-cache-dir .

echo "TensorRT-LLM installation complete"
python3 -c "import tensorrt_llm; print(f'TensorRT-LLM version: {tensorrt_llm.__version__}')"
python3 -c "from tensorrt_llm.commands.serve import main; print('gRPC serve command: available')"
