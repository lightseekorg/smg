#!/bin/bash
# Install TensorRT-LLM with gRPC support for CI
#
# gRPC server support (PR #11037) is not yet in a pip release,
# so we clone main and do an editable install with TRTLLM_USE_PRECOMPILED=1.
# This pulls pre-compiled C++/CUDA binaries from the PyPI wheel and layers
# the Python source (including gRPC server) from the git checkout on top.
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

# Install using pre-compiled binaries from PyPI + Python source from git
# This avoids the full C++ build (63GB Docker image) while getting gRPC support
# from the main branch that isn't in the PyPI release yet.
echo "Installing TensorRT-LLM (precompiled binaries + main branch Python source)..."
cd "$TRTLLM_DIR"
TRTLLM_USE_PRECOMPILED=1 pip install --no-cache-dir -e .

echo "TensorRT-LLM installation complete"
python3 -c "import tensorrt_llm; print(f'TensorRT-LLM version: {tensorrt_llm.__version__}')"
python3 -c "from tensorrt_llm.commands.serve import main; print('gRPC serve command: available')"
