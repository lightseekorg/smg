#!/bin/bash
# Install TensorRT-LLM with gRPC support for CI
#
# gRPC server support (PR #11037) is not yet in a pip release,
# so we install the latest stable wheel (which has all compiled binaries)
# and then overlay the Python source from main branch on top. This gives
# us the gRPC serve command from main + compiled C++/CUDA libs from stable.
# No Docker or C++ build toolchain required.
#
# TRT-LLM 1.1.0 requires CUDA 13 libraries (libcublasLt.so.13 etc.).
# We install these via NVIDIA's apt repository.
#
# At runtime we use --backend pytorch, which avoids TRT engine compilation.

set -euo pipefail

# Activate venv if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Step 1: Install CUDA 13 libraries via NVIDIA apt repo.
# TRT-LLM 1.1.0 is built against CUDA 13. The pip packages only provide
# a subset of CUDA 13 libs (cuda-runtime, nvrtc) but not cublas, cusolver, etc.
# Install the full CUDA 13 library set via apt.
echo "Installing CUDA 13 libraries..."
export DEBIAN_FRONTEND=noninteractive
if ! dpkg -l cuda-keyring 2>/dev/null | grep -q '^ii'; then
    curl -fsSL -o cuda-keyring_1.1-1_all.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    sudo DEBIAN_FRONTEND=noninteractive dpkg -i --force-confnew cuda-keyring_1.1-1_all.deb
fi
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y libopenmpi-dev cuda-libraries-13-1

# Add CUDA 13 libs to LD_LIBRARY_PATH
export LD_LIBRARY_PATH="/usr/local/cuda-13.1/lib64:${LD_LIBRARY_PATH:-}"

# Install PyTorch with CUDA support
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Step 2: Install the latest stable TRT-LLM wheel from NVIDIA PyPI.
echo "Installing stable TensorRT-LLM wheel..."
pip install --no-cache-dir tensorrt_llm --extra-index-url=https://pypi.nvidia.com

# Step 3: Set LD_LIBRARY_PATH for pip-installed NVIDIA libraries.
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")

NVIDIA_LIB_DIRS=$(find "$SITE_PACKAGES/nvidia" -name "lib" -type d 2>/dev/null | sort -u | paste -sd':')
if [ -n "$NVIDIA_LIB_DIRS" ]; then
    export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}:${LD_LIBRARY_PATH:-}"
fi

TRTLLM_LIB_DIR=$(find "$SITE_PACKAGES" -path "*/tensorrt_llm/libs" -type d 2>/dev/null | head -1)
if [ -n "$TRTLLM_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="${TRTLLM_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

# Add system CUDA as fallback
if [ -d "/usr/local/cuda/lib64" ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi

echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# Step 4: Clone main branch for gRPC serve command (not in any release yet).
TRTLLM_DIR="/tmp/tensorrt-llm-src"
if [ ! -d "$TRTLLM_DIR" ]; then
    echo "Cloning TensorRT-LLM main branch for gRPC source..."
    git clone --depth 1 https://github.com/NVIDIA/TensorRT-LLM.git "$TRTLLM_DIR"
fi

# Step 5: Overlay Python source from main branch onto the installed package.
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
