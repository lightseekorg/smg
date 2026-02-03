#!/bin/bash
# Install TensorRT-LLM with gRPC support for CI
#
# gRPC server support (PR #11037) is not yet in a pip release,
# so we install the latest stable wheel (compiled binaries) and then
# overlay ALL Python source from main branch on top. We patch the few
# imports that reference compiled C++ extensions missing from the wheel.
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
nvidia-smi 2>/dev/null | head -4 || echo "WARNING: nvidia-smi not found"
nvcc --version 2>/dev/null || echo "WARNING: nvcc not found"
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

# Step 3: Clone main branch for gRPC serve command (not in any release yet).
TRTLLM_DIR="/tmp/tensorrt-llm-src"
if [ ! -d "$TRTLLM_DIR" ]; then
    echo "Cloning TensorRT-LLM main branch for gRPC source..."
    git clone --depth 1 https://github.com/NVIDIA/TensorRT-LLM.git "$TRTLLM_DIR"
fi

# Step 4: Full Python overlay from main branch, skipping compiled extensions.
# TRT-LLM prints a version banner to stdout on import; use tail -1 to get only the path.
INSTALL_DIR=$(python3 -c "import tensorrt_llm, pathlib; print(pathlib.Path(tensorrt_llm.__file__).parent)" 2>/dev/null | tail -1)
echo "Installed package at: $INSTALL_DIR"
echo "Overlaying main branch Python source..."

python3 -c "
import shutil, pathlib
src = pathlib.Path('$TRTLLM_DIR/tensorrt_llm')
dst = pathlib.Path('$INSTALL_DIR')
skip = {'bindings', 'libs'}
count = 0
for f in src.rglob('*.py'):
    rel = f.relative_to(src)
    if rel.parts[0] in skip:
        continue
    out = dst / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(f, out)
    count += 1
print(f'Copied {count} Python files')
"

# Step 5: Copy vendored triton_kernels module (required by main branch __init__.py).
if [ -d "$TRTLLM_DIR/triton_kernels" ]; then
    echo "Installing vendored triton_kernels module..."
    cp -r "$TRTLLM_DIR/triton_kernels" "$SITE_PACKAGES/triton_kernels"
fi

# Step 6: Patch imports of compiled C++ extensions that don't exist in the
# stable wheel. The gRPC serve path (--backend pytorch) doesn't need these.
echo "Patching incompatible compiled extension imports..."
python3 -c "
import pathlib

install_dir = pathlib.Path('$INSTALL_DIR')

patches = [
    (
        'runtime/kv_cache_manager_v2/rawref/__init__.py',
        'from ._rawref import NULL, ReferenceType, ref',
        'try:\\n    from ._rawref import NULL, ReferenceType, ref\\nexcept (ImportError, ModuleNotFoundError):\\n    NULL = None; ReferenceType = None\\n    class _RefStub:\\n        def __class_getitem__(cls, item): return cls\\n    ref = _RefStub',
    ),
]

# Also add deferred annotations to files that use rawref type hints at runtime
deferred = [
    'runtime/kv_cache_manager_v2/_utils.py',
    'runtime/kv_cache_manager_v2/_life_cycle_registry.py',
    'runtime/kv_cache_manager_v2/_block_radix_tree.py',
]
for rel_path in deferred:
    fpath = install_dir / rel_path
    if not fpath.exists():
        print(f'  SKIP deferred (not found): {rel_path}')
        continue
    text = fpath.read_text()
    if 'from __future__ import annotations' not in text:
        text = 'from __future__ import annotations\\n' + text
        fpath.write_text(text)
        print(f'  DEFERRED ANNOTATIONS: {rel_path}')
    else:
        print(f'  SKIP deferred (already present): {rel_path}')

for rel_path, old, new in patches:
    fpath = install_dir / rel_path
    if not fpath.exists():
        print(f'  SKIP (not found): {rel_path}')
        continue
    text = fpath.read_text()
    if old in text:
        text = text.replace(old, new)
        fpath.write_text(text)
        print(f'  PATCHED: {rel_path}')
    else:
        print(f'  SKIP (pattern not found): {rel_path}')
"

# Persist LD_LIBRARY_PATH for subsequent CI steps
if [ -n "${GITHUB_ENV:-}" ]; then
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> "$GITHUB_ENV"
fi

echo "TensorRT-LLM installation complete"
python3 -c "import tensorrt_llm; print(f'TensorRT-LLM version: {tensorrt_llm.__version__}')"
python3 -c "from tensorrt_llm.commands.serve import main; print('gRPC serve command: available')"

# Smoke-test: verify the serve command can parse --help without crashing
echo "Verifying gRPC serve command..."
python3 -m tensorrt_llm.commands.serve --help 2>&1 | head -20 || echo "WARNING: serve --help failed"
