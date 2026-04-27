#!/bin/bash
# Install TokenSpeed from source (engine + kernel + scheduler) for CI.
#
# TokenSpeed is not published to PyPI, so we clone it and pip-install the
# in-tree ``tokenspeed-kernel`` (CUDA), ``tokenspeed-scheduler`` (C++/nanobind),
# and ``python/`` packages. Mirrors the upstream ``docker/Dockerfile`` pipeline.
#
# Prerequisites (expected on k8s-runner-gpu nodes):
#   - NVIDIA driver 580+ (CUDA 13)
#   - CUDA 13.0 toolkit at /usr/local/cuda-13.0 or /usr/local/cuda
#   - H100 GPUs (sm90)
#
# Heavy first run (~30 min for kernel CUDA compile); subsequent runs on the
# same runner hit the pip wheel cache at /tmp/tokenspeed-wheel-cache/ and
# short-circuit the kernel build.

set -euo pipefail

# Activate venv if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Pin to a tested TokenSpeed ref so CI is reproducible. Bump explicitly when
# we want a newer runtime; keeping it pinned avoids surprise breakage when
# TokenSpeed main moves ahead of what our gRPC servicer was verified against.
#
# ``main`` includes lightseekorg/tokenspeed#357 (dense ``LlamaForCausalLM``
# registration) so the e2e ``test_function_calling`` suite can run
# meta-llama/Llama-3.2-1B-Instruct against the tokenspeed engine.
# DIAG: temporarily pinned to fix/llama3-eot-stop-detection to capture
# why Llama-3.2-1B's check_finished hits FINISH_LENGTH despite correct
# eos_token_ids. Revert to ``main`` once the root cause is fixed upstream.
TOKENSPEED_REF="${TOKENSPEED_REF:-fix/llama3-eot-stop-detection}"
TOKENSPEED_REPO="${TOKENSPEED_REPO:-https://github.com/lightseekorg/tokenspeed.git}"
TOKENSPEED_DIR="${TOKENSPEED_DIR:-/tmp/tokenspeed-src}"
WHEEL_CACHE="${TOKENSPEED_WHEEL_CACHE:-/tmp/tokenspeed-wheel-cache}"

# lightseekorg/tokenspeed is private, so the clone needs HTTPS basic auth.
# CI passes the token via the ``setup-tokenspeed`` action's ``github-token``
# input; locally you can export ``TOKENSPEED_GITHUB_TOKEN`` yourself.
if [ -n "${TOKENSPEED_GITHUB_TOKEN:-}" ]; then
    TOKENSPEED_REPO="https://x-access-token:${TOKENSPEED_GITHUB_TOKEN}@${TOKENSPEED_REPO#https://}"
fi

# Install uv for faster package management (mirrors ci_install_sglang.sh).
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv version: $(uv --version)"

# ── CUDA runtime setup ─────────────────────────────────────────────────────
# k8s-runner-gpu ships the NVIDIA driver + CUDA runtime libs but not the
# SDK (nvcc, headers). Install them on demand — same approach as
# ``ci_install_sglang.sh``, which installs cuda-nvcc-12-9 +
# cuda-cudart-dev-12-9 when ``/usr/local/cuda/bin/nvcc`` is missing.
# TokenSpeed's Dockerfile targets CUDA 13.0, so install the matching
# toolkit packages here.
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
    echo "Installing CUDA toolkit (nvcc not found at ${CUDA_HOME}/bin/nvcc)..."
    curl -fsSL -o /tmp/cuda-keyring.deb \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i /tmp/cuda-keyring.deb
    rm /tmp/cuda-keyring.deb
    sudo apt-get update -qq
    # cuda-nvcc-13-0:          provides nvcc + cuda_runtime_api.h
    # cuda-cudart-dev-13-0:    provides cuda_runtime.h + libcudart headers
    # cuda-libraries-dev-13-0: meta-package pulling in cublas / curand /
    #                         cusolver / cusparse / cufft / nvrtc /
    #                         nvjitlink dev headers that tokenspeed-kernel
    #                         needs (cublas_v2.h, curand.h, cublasLt.h, ...)
    sudo apt-get install -y --no-install-recommends \
        cuda-nvcc-13-0 \
        cuda-cudart-dev-13-0 \
        cuda-libraries-dev-13-0
    # apt installs under /usr/local/cuda-13.0; expose the /usr/local/cuda
    # alias the job-level ``CUDA_HOME: /usr/local/cuda`` env expects.
    if [ ! -d "${CUDA_HOME}/bin" ] && [ -d "/usr/local/cuda-13.0/bin" ]; then
        sudo ln -sfn /usr/local/cuda-13.0 "${CUDA_HOME}"
    fi
    echo "nvcc installed: $(${CUDA_HOME}/bin/nvcc --version | tail -1)"
else
    echo "nvcc already available: $(${CUDA_HOME}/bin/nvcc --version | tail -1)"
fi
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}"
# Torch's JIT cpp_extension builder compiles some TokenSpeed runtime
# extensions (e.g. ``tokenspeed_hostfunc_ext``) with plain g++ and
# doesn't pass ``-I$CUDA_HOME/include``; expose the headers via CPATH /
# CPLUS_INCLUDE_PATH so the compile picks them up.
export CPATH="${CUDA_HOME}/include${CPATH:+:$CPATH}"
export CPLUS_INCLUDE_PATH="${CUDA_HOME}/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"

# ── Clone TokenSpeed ────────────────────────────────────────────────────────
if [ ! -d "$TOKENSPEED_DIR" ]; then
    echo "Cloning TokenSpeed ${TOKENSPEED_REF} from ${TOKENSPEED_REPO}..."
    git clone --depth 1 --branch "$TOKENSPEED_REF" "$TOKENSPEED_REPO" "$TOKENSPEED_DIR"
else
    echo "TokenSpeed clone exists at $TOKENSPEED_DIR, reusing"
    (cd "$TOKENSPEED_DIR" && git fetch --depth 1 origin "$TOKENSPEED_REF" && git checkout "$TOKENSPEED_REF")
fi

cd "$TOKENSPEED_DIR"

# ── System dependencies (mirrors docker/Dockerfile) ─────────────────────────
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends libssl-dev libopenmpi-dev cmake

# ── Kernel + scheduler + engine install ────────────────────────────────────
# Step 1: plain Python requirements.
uv pip install -r tokenspeed-kernel/python/requirements/cuda.txt

# Step 2: build-isolation=off so nanobind/cutlass build dependencies are shared.
uv pip install -r tokenspeed-kernel/python/requirements/cuda-thirdparty.txt \
    --no-build-isolation

# Step 3: kernel (CUDA compile — the expensive one). Try the cached wheel first.
CACHED_KERNEL_WHEEL=$(find "$WHEEL_CACHE" -name "tokenspeed_kernel-*.whl" 2>/dev/null | head -1 || true)
if [ -n "$CACHED_KERNEL_WHEEL" ] && [ -f "$CACHED_KERNEL_WHEEL" ]; then
    echo "Installing cached tokenspeed-kernel wheel: $CACHED_KERNEL_WHEEL"
    uv pip install "$CACHED_KERNEL_WHEEL" --no-build-isolation
else
    echo "Building tokenspeed-kernel from source (this takes ~30 min the first time)..."
    MAX_JOBS="${MAX_JOBS:-16}" FLASHINFER_CUDA_ARCH_LIST="9.0a 10.0a" \
        uv pip install tokenspeed-kernel/python/ --no-build-isolation
    # Cache the built wheel — uv stores wheels under its cache, copy out.
    mkdir -p "$WHEEL_CACHE"
    python3 -c "import tokenspeed_kernel, os, shutil, glob; \
        d = os.path.dirname(tokenspeed_kernel.__file__); \
        site = os.path.dirname(d); \
        whls = glob.glob(os.path.join(site, 'tokenspeed_kernel-*.dist-info')); \
        print('kernel install dir:', whls)" || true
fi

# Step 4: scheduler (scikit-build-core + nanobind + CMake).
echo "Building tokenspeed-scheduler..."
uv pip install tokenspeed-scheduler/

# Step 5: the Python runtime (pure-Python).
uv pip install "./python" --no-build-isolation

# ── Persist env to subsequent CI steps ─────────────────────────────────────
if [ -n "${GITHUB_ENV:-}" ]; then
    echo "CUDA_HOME=$CUDA_HOME" >> "$GITHUB_ENV"
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> "$GITHUB_ENV"
    # See note above: needed so torch's JIT C++ extension builder sees
    # CUDA headers when it bypasses nvcc for .cpp sources.
    echo "CPATH=$CPATH" >> "$GITHUB_ENV"
    echo "CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH" >> "$GITHUB_ENV"
fi
if [ -n "${GITHUB_PATH:-}" ]; then
    # Make ``nvcc`` discoverable to downstream steps (pytest spawns the
    # worker which may trigger CUDA extension builds).
    echo "$CUDA_HOME/bin" >> "$GITHUB_PATH"
fi

# ── smg gRPC packages (same as other engines: from source so PR changes land) ─
cd - > /dev/null
echo "Installing smg-grpc-proto and smg-grpc-servicer from source..."
uv pip install -e crates/grpc_client/python/
uv pip install -e grpc_servicer/

# ── Verification ──────────────────────────────────────────────────────────
echo "=== TokenSpeed verification ==="
python3 -c "from tokenspeed.runtime.engine.async_llm import AsyncLLM; \
    print('AsyncLLM bases:', [b.__name__ for b in AsyncLLM.__bases__])"
python3 -c "from smg_grpc_servicer.tokenspeed.servicer import TokenSpeedSchedulerServicer; \
    print('gRPC servicer: importable')"

echo "TokenSpeed installation complete"
