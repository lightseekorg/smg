#!/bin/bash
# Install vLLM with flash-attn for CI
# Handles CUDA toolkit setup and flash-attn compilation

set -euo pipefail

# Activate venv if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "Installing vLLM..."
python3 -m pip install vllm

# Ensure CUDA toolkit (nvcc) is available for building flash-attn
if [ ! -f /usr/local/cuda/bin/nvcc ]; then
    echo "nvcc not found at /usr/local/cuda/bin/nvcc, setting up CUDA toolkit..."
    sudo apt-get update
    sudo apt-get install -y cuda-nvcc-12-8 cuda-cudart-dev-12-8 || \
        sudo apt-get install -y nvidia-cuda-toolkit

    # Find where nvcc was installed and create symlinks
    NVCC_PATH=$(which nvcc 2>/dev/null || find /usr -name nvcc -type f 2>/dev/null | head -1)
    if [ -n "$NVCC_PATH" ]; then
        echo "Found nvcc at: $NVCC_PATH"
        CUDA_BIN_DIR=$(dirname "$NVCC_PATH")
        CUDA_DIR=$(dirname "$CUDA_BIN_DIR")

        # Create /usr/local/cuda symlink if it doesn't exist
        if [ ! -e /usr/local/cuda ]; then
            sudo ln -sf "$CUDA_DIR" /usr/local/cuda
            echo "Created symlink: /usr/local/cuda -> $CUDA_DIR"
        elif [ ! -f /usr/local/cuda/bin/nvcc ]; then
            # /usr/local/cuda exists but nvcc is missing, create bin symlink
            sudo mkdir -p /usr/local/cuda/bin
            sudo ln -sf "$NVCC_PATH" /usr/local/cuda/bin/nvcc
            echo "Created symlink: /usr/local/cuda/bin/nvcc -> $NVCC_PATH"
        fi
    else
        echo "ERROR: nvcc not found after installation"
        exit 1
    fi
fi

# Verify nvcc is accessible
echo "CUDA setup:"
ls -la /usr/local/cuda/bin/nvcc || true
/usr/local/cuda/bin/nvcc --version || true

export CUDA_HOME=/usr/local/cuda
# Add CUDA to PATH but keep venv Python first
export PATH="$CUDA_HOME/bin:$PATH"

# Re-activate venv to ensure venv Python is first in PATH
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Install flash-attn
echo "Installing flash-attn..."
python3 -m pip install flash-attn --no-build-isolation

# Install flashinfer (remove conflicting versions first)
echo "Installing flashinfer..."
python3 -m pip uninstall -y flashinfer flashinfer-python flashinfer-cubin flashinfer-jit-cache 2>/dev/null || true
python3 -m pip install flashinfer-python==0.5.3 flashinfer-cubin==0.5.3

# Install nixl for vLLM PD disaggregation (NIXL KV transfer)
echo "Installing nixl..."
python3 -m pip install nixl

echo "vLLM installation complete"
