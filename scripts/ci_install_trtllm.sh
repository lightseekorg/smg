#!/bin/bash
# Install TensorRT-LLM from source with gRPC support for CI
#
# gRPC server support (PR #11037) is not yet in a pip release,
# so we build from source (main branch) which compiles the C++
# extensions properly and includes the gRPC serve command.
#
# Prerequisites (expected on k8s-runner-gpu nodes):
#   - NVIDIA driver 580+ (CUDA 13)
#   - CUDA 13.0 toolkit at /usr/local/cuda-13.0
#   - H100 GPUs (sm90)
#
# At runtime we use --backend pytorch, which avoids TRT engine compilation.

set -euo pipefail

# Activate venv if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# ── System dependencies ──────────────────────────────────────────────────────
export DEBIAN_FRONTEND=noninteractive
sudo dpkg --configure -a --force-confnew 2>/dev/null || true

# Add NVIDIA CUDA/TensorRT apt repository (needed for libnvinfer-dev, tensorrt-dev)
if ! dpkg -l cuda-keyring 2>/dev/null | grep -q '^ii'; then
    echo "Setting up NVIDIA apt repository..."
    curl -fsSL -o /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i /tmp/cuda-keyring.deb
    rm -f /tmp/cuda-keyring.deb
fi

sudo apt-get update
sudo apt-get install -y libopenmpi-dev git-lfs libnvinfer10 libnvinfer-dev tensorrt-dev cuda-toolkit-13-0 cmake

# ── CUDA setup ───────────────────────────────────────────────────────────────
# Prefer /usr/local/cuda-13.0 if it exists, otherwise fall back to /usr/local/cuda
if [ -d "/usr/local/cuda-13.0" ]; then
    export CUDA_HOME="/usr/local/cuda-13.0"
else
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
fi
# Re-activate venv first, then add CUDA to PATH so it takes precedence
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}"

# Debug: print what CUDA we actually have
echo "=== CUDA diagnostics ==="
echo "CUDA_HOME=$CUDA_HOME"
echo "PATH=$PATH"
ls -la "$CUDA_HOME/bin/nvcc" 2>/dev/null || echo "WARNING: nvcc not at $CUDA_HOME/bin/nvcc"
find /usr/local -name "nvcc" -type f 2>/dev/null || echo "WARNING: nvcc not found anywhere in /usr/local"
which nvcc 2>/dev/null || echo "WARNING: nvcc not on PATH"
nvcc --version 2>/dev/null || echo "WARNING: nvcc --version failed"
nvidia-smi 2>/dev/null | head -4 || echo "WARNING: nvidia-smi not found"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"
python3 --version
echo "=== end CUDA diagnostics ==="

# ── TensorRT symlinks (for CMake to find TensorRT) ──────────────────────────
sudo mkdir -p /usr/local/tensorrt
sudo ln -sf /usr/include/x86_64-linux-gnu /usr/local/tensorrt/include
sudo ln -sf /usr/lib/x86_64-linux-gnu /usr/local/tensorrt/lib

# ── NCCL 2.27 setup ──────────────────────────────────────────────────────────
# Use pip-installed NCCL 2.27+ which has both headers and libraries.
# This matches the working installation guide approach.
pip install --upgrade pip
pip install --no-cache-dir "nvidia-nccl-cu13>=2.27.7"

SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")

# Use pip NCCL package as NCCL_ROOT (has both include/ and lib/ directories)
NCCL_ROOT="$SITE_PACKAGES/nvidia/nccl"

echo "=== NCCL diagnostics ==="
echo "NCCL_ROOT=$NCCL_ROOT"
ls -la "$NCCL_ROOT/" 2>/dev/null || echo "WARNING: NCCL_ROOT not found"
ls -la "$NCCL_ROOT/include/" 2>/dev/null || echo "WARNING: NCCL include not found"
ls -la "$NCCL_ROOT/lib/" 2>/dev/null || echo "WARNING: NCCL lib not found"
echo "=== end NCCL diagnostics ==="

# Symlink pip NCCL header to system path for other tools that look there
NCCL_INCLUDE=$(find "$NCCL_ROOT" -name "nccl.h" 2>/dev/null | head -1)
if [ -n "$NCCL_INCLUDE" ]; then
    echo "Found pip NCCL header at: $NCCL_INCLUDE"
    sudo mv /usr/include/nccl.h /usr/include/nccl.h.bak 2>/dev/null || true
    sudo ln -sf "$NCCL_INCLUDE" /usr/include/nccl.h
    echo "Symlinked pip NCCL header to /usr/include/nccl.h"
else
    echo "WARNING: Could not find pip-installed NCCL header"
fi

# Create libnccl.so symlink - pip package only has libnccl.so.2, but CMake looks for libnccl.so
NCCL_LIB=$(find "$NCCL_ROOT" -name "libnccl.so.2" 2>/dev/null | head -1)
if [ -n "$NCCL_LIB" ]; then
    NCCL_LIB_DIR=$(dirname "$NCCL_LIB")
    echo "Found NCCL library at: $NCCL_LIB"
    ln -sf "$NCCL_LIB" "$NCCL_LIB_DIR/libnccl.so"
    echo "Created symlink: $NCCL_LIB_DIR/libnccl.so -> libnccl.so.2"
    ls -la "$NCCL_LIB_DIR"/libnccl*
else
    echo "WARNING: Could not find pip-installed NCCL library"
fi

# ── Clone TensorRT-LLM ──────────────────────────────────────────────────────
TRTLLM_DIR="/tmp/tensorrt-llm-src"
if [ ! -d "$TRTLLM_DIR" ]; then
    echo "Cloning TensorRT-LLM main branch..."
    git clone --depth 1 https://github.com/NVIDIA/TensorRT-LLM.git "$TRTLLM_DIR"
fi

cd "$TRTLLM_DIR"
git lfs install --force
git lfs pull

# ── Install TensorRT-LLM Python requirements ─────────────────────────────────
# cutlass_library and other build dependencies
if [ -f "requirements-dev.txt" ]; then
    echo "Installing TensorRT-LLM build requirements..."
    pip install --no-cache-dir -r requirements-dev.txt
fi

# ── Patch FindTensorRT.cmake ─────────────────────────────────────────────────
# CMake needs to find TensorRT in system paths
CMAKE_FILE="cpp/cmake/modules/FindTensorRT.cmake"
if [ -f "$CMAKE_FILE" ]; then
    echo "Patching FindTensorRT.cmake for system paths..."
    python3 <<'PYTHON_EOF'
import pathlib
import re
import sys

cmake_file = sys.argv[1] if len(sys.argv) > 1 else "cpp/cmake/modules/FindTensorRT.cmake"
p = pathlib.Path(cmake_file)
text = p.read_text()

# Add system paths to CMAKE_FIND_ROOT_PATH
if '/usr/local/tensorrt' not in text or 'list(APPEND CMAKE_FIND_ROOT_PATH' not in text:
    text = text.replace(
        'set(TensorRT_WELL_KNOWN_ROOT /usr/local/tensorrt)',
        'set(TensorRT_WELL_KNOWN_ROOT /usr/local/tensorrt)\nlist(APPEND CMAKE_FIND_ROOT_PATH /usr/local/tensorrt /usr)',
    )

# Patch find_path for NvInfer.h to include system paths
text = re.sub(
    r'(find_path\(\s*TensorRT_INCLUDE_DIR\s+NAMES\s+NvInfer\.h\s+PATHS\s+\$\{TensorRT_WELL_KNOWN_ROOT\}/include)',
    r'\1 /usr/include/x86_64-linux-gnu',
    text,
)

# Add system library paths to find_library calls (matches installation guide)
text = re.sub(
    r'(find_library\([^)]*PATHS\s+\$\{TensorRT_WELL_KNOWN_ROOT\}/lib)(\s*\))',
    r'\1 /usr/lib/x86_64-linux-gnu\2',
    text,
    flags=re.DOTALL,
)

# Add NO_CMAKE_FIND_ROOT_PATH to find_path and find_library calls
for pattern in [r'(find_path\([^)]*)\)', r'(find_library\([^)]*)\)']:
    for match in re.finditer(pattern, text, re.DOTALL):
        block = match.group(0)
        if 'TensorRT' in block and 'NO_CMAKE_FIND_ROOT_PATH' not in block:
            patched = block[:-1] + '\n  NO_CMAKE_FIND_ROOT_PATH)'
            text = text.replace(block, patched)

p.write_text(text)
print('FindTensorRT.cmake patched')
PYTHON_EOF
fi

# ── Patch FindNCCL.cmake ─────────────────────────────────────────────────────
# The upstream FindNCCL.cmake doesn't use NCCL_ROOT hint at all!
# We need to add PATHS ${NCCL_ROOT}/lib and NO_CMAKE_FIND_ROOT_PATH
NCCL_CMAKE_FILE="cpp/cmake/modules/FindNCCL.cmake"
if [ -f "$NCCL_CMAKE_FILE" ]; then
    echo "Patching FindNCCL.cmake to use NCCL_ROOT hint..."
    python3 <<'PYTHON_EOF'
import pathlib

p = pathlib.Path("cpp/cmake/modules/FindNCCL.cmake")
text = p.read_text()

# Replace simple find_library/find_path calls with ones that use NCCL_ROOT hint
# Original: find_library(NCCL_LIBRARY NAMES nccl)
# Patched:  find_library(NCCL_LIBRARY NAMES nccl PATHS ${NCCL_ROOT}/lib NO_CMAKE_FIND_ROOT_PATH)

# The pip nvidia-nccl-cu13 package has libnccl.so.2 directly in NCCL_ROOT, not in lib/
text = text.replace(
    'find_library(NCCL_LIBRARY NAMES nccl)',
    'find_library(NCCL_LIBRARY NAMES nccl PATHS ${NCCL_ROOT} ${NCCL_ROOT}/lib NO_CMAKE_FIND_ROOT_PATH)'
)

text = text.replace(
    'find_library(NCCL_STATIC_LIBRARY NAMES nccl_static)',
    'find_library(NCCL_STATIC_LIBRARY NAMES nccl_static PATHS ${NCCL_ROOT} ${NCCL_ROOT}/lib NO_CMAKE_FIND_ROOT_PATH)'
)

text = text.replace(
    'find_path(NCCL_INCLUDE_DIR NAMES nccl.h)',
    'find_path(NCCL_INCLUDE_DIR NAMES nccl.h PATHS ${NCCL_ROOT}/include NO_CMAKE_FIND_ROOT_PATH)'
)

p.write_text(text)
print('FindNCCL.cmake patched to use NCCL_ROOT hint')
PYTHON_EOF
fi

# ── Build TensorRT-LLM from source ──────────────────────────────────────────
echo "Building TensorRT-LLM from source (this may take a while)..."

python3 scripts/build_wheel.py \
    --cuda_architectures "90-real" \
    --trt_root /usr/local/tensorrt \
    --nccl_root "$NCCL_ROOT" \
    --install \
    --no-venv \
    -j "$(nproc)" \
    -D "ENABLE_UCX=OFF" \
    --clean

# Return to repo dir
cd -

# ── Add pip-installed NVIDIA libraries to LD_LIBRARY_PATH ────────────────────
NVIDIA_LIB_DIRS=$(find "$SITE_PACKAGES/nvidia" -name "lib" -type d 2>/dev/null | sort -u | paste -sd':')
if [ -n "$NVIDIA_LIB_DIRS" ]; then
    export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}:${LD_LIBRARY_PATH:-}"
fi

TRTLLM_LIB_DIR=$(find "$SITE_PACKAGES" -path "*/tensorrt_llm/libs" -type d 2>/dev/null | head -1)
if [ -n "$TRTLLM_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="${TRTLLM_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

# Persist LD_LIBRARY_PATH for subsequent CI steps
if [ -n "${GITHUB_ENV:-}" ]; then
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> "$GITHUB_ENV"
fi

# ── Verification ─────────────────────────────────────────────────────────────
echo "=== TensorRT-LLM verification ==="
python3 -c "import tensorrt_llm; print(f'TensorRT-LLM version: {tensorrt_llm.__version__}')"
python3 -c "from tensorrt_llm.commands.serve import main; print('gRPC serve command: available')"

# Smoke-test: verify the serve command can parse --help without crashing
echo "Verifying gRPC serve command..."
python3 -m tensorrt_llm.commands.serve serve --help 2>&1 | head -20 || echo "WARNING: serve --help failed"

echo "TensorRT-LLM installation complete"
