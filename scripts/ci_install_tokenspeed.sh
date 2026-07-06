#!/bin/bash
# Install TokenSpeed from source (engine + kernel + scheduler) for CI.
#
# Mirrors the upstream install pattern (see tokenspeed's docs / test/ci_system/
# install_deps.sh): one editable pip install per package, in engine →
# kernel → scheduler order. The kernel package's metadata pulls in its
# own CUDA dependencies, so we don't pre-install requirements files.
#
# Prerequisites (expected on k8s-runner-gpu nodes):
#   - NVIDIA driver 580+ (CUDA 13)
#   - CUDA 13.0 toolkit at /usr/local/cuda-13.0 or /usr/local/cuda
#   - H100 GPUs (sm90)

set -euo pipefail

# Activate venv if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Pinned SHA from lightseekorg/tokenspeed main. Bump explicitly (ideally via
# a scheduled bump-and-CI routine) rather than floating against ``main`` —
# upstream has renamed APIs before and the gRPC servicer broke until we
# caught up.
TOKENSPEED_REF="${TOKENSPEED_REF:-5e145afae8e5651cd66234e68c988c31aac6639f}"
TOKENSPEED_REPO="${TOKENSPEED_REPO:-https://github.com/lightseekorg/tokenspeed.git}"
TOKENSPEED_DIR="${TOKENSPEED_DIR:-/tmp/tokenspeed-src}"

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
# ``ci_install_sglang.sh``.
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
    echo "Installing CUDA toolkit (nvcc not found at ${CUDA_HOME}/bin/nvcc)..."
    curl -fsSL -o /tmp/cuda-keyring.deb \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i /tmp/cuda-keyring.deb
    rm /tmp/cuda-keyring.deb
    sudo apt-get update -qq
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
# ``git clone --branch`` only accepts branch/tag names, not SHAs, so we
# init+fetch+checkout instead. Works for both SHAs and refs.
if [ ! -d "$TOKENSPEED_DIR" ]; then
    echo "Cloning TokenSpeed ${TOKENSPEED_REF} from ${TOKENSPEED_REPO}..."
    git init -q "$TOKENSPEED_DIR"
    (cd "$TOKENSPEED_DIR" \
        && git remote add origin "$TOKENSPEED_REPO" \
        && git fetch --depth 1 origin "$TOKENSPEED_REF" \
        && git checkout FETCH_HEAD)
else
    echo "TokenSpeed clone exists at $TOKENSPEED_DIR, reusing"
    (cd "$TOKENSPEED_DIR" && git fetch --depth 1 origin "$TOKENSPEED_REF" && git checkout "$TOKENSPEED_REF")
fi

cd "$TOKENSPEED_DIR"

# ── System dependencies (mirrors docker/Dockerfile) ─────────────────────────
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends libssl-dev libopenmpi-dev cmake

# ── TokenSpeed packages ────────────────────────────────────────────────────
export MAX_JOBS="${MAX_JOBS:-16}"
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-9.0a 10.0a}"

# The kernel requirements leave ``nvidia-cutlass-dsl`` unpinned, and 4.6.0
# dropped ``cute.core.ThrMma`` — which quack (pulled via flash-attn's cute
# backend) uses, breaking ``import tokenspeed``. Pin to the last compatible
# release. Both UV_CONSTRAINT and PIP_CONSTRAINT are needed: the kernel's
# setup.py shells out to ``pip install -r requirements/cuda.txt`` during the
# native build (see _install_backend_build_requirements), and that subprocess
# pip does not see UV_CONSTRAINT — without PIP_CONSTRAINT it pulls 4.6.0
# alongside the pinned 4.5.2 and wins on sys.path.
TOKENSPEED_CONSTRAINTS="$(mktemp)"
echo "nvidia-cutlass-dsl==4.5.2" > "$TOKENSPEED_CONSTRAINTS"
export UV_CONSTRAINT="$TOKENSPEED_CONSTRAINTS"
export PIP_CONSTRAINT="$TOKENSPEED_CONSTRAINTS"

# Preseed build-time tooling: ``./python`` and ``tokenspeed-kernel`` use
# ``setuptools.build_meta`` without declaring ``setuptools`` in
# ``build-system.requires``, and we install with ``--no-build-isolation``.
uv pip install setuptools wheel pybind11

uv pip install -e tokenspeed-kernel/python/ --no-build-isolation
uv pip install -e tokenspeed-scheduler/
uv pip install -e "./python" --no-build-isolation

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
# TokenSpeed's engine package pins its own builds of these modules
# (tokenspeed-smg-grpc-proto / tokenspeed-smg-grpc-servicer). Those dists
# install the same smg_grpc_proto / smg_grpc_servicer import paths into
# site-packages, which shadow the editable installs below — the worker
# would then serve stale proto descriptors ("Method not found!" for any
# RPC added in the PR). Drop them first; the source installs replace them.
uv pip uninstall tokenspeed-smg-grpc-proto tokenspeed-smg-grpc-servicer
uv pip install -e crates/grpc_client/python/
uv pip install -e grpc_servicer/

# ── cutlass provenance (diagnostic) ─────────────────────────────────────────
# quack 0.5.0 uses the deprecated ``cute.core.ThrMma`` shim (present in 4.5.2,
# removed in 4.6.0). Identical pip installs have imported different cutlass
# builds across runners, so surface exactly what loads and from where.
echo "=== cutlass provenance ==="
uv pip show nvidia-cutlass-dsl 2>/dev/null | grep -iE "^(Name|Version|Location):" || true
python3 -c "
import sys, cutlass, cutlass.cute.core as core
print('import cutlass  ->', cutlass.__file__)
print('cutlass version ->', getattr(cutlass, '__version__', '?'))
print('cute.core file  ->', core.__file__)
print('cute.core ThrMma->', hasattr(core, 'ThrMma'))
print('sys.path:')
[print('   ', p) for p in sys.path]
" || true

# ── Verification ──────────────────────────────────────────────────────────
echo "=== TokenSpeed verification ==="
python3 -c "from tokenspeed.runtime.engine.async_llm import AsyncLLM; \
    print('AsyncLLM bases:', [b.__name__ for b in AsyncLLM.__bases__])"
python3 -c "from smg_grpc_servicer.tokenspeed.servicer import TokenSpeedSchedulerServicer; \
    print('gRPC servicer: importable')"
python3 -c "
import pathlib
import smg_grpc_proto
import smg_grpc_servicer

repo = pathlib.Path.cwd().resolve()
paths = [pathlib.Path(m.__file__).resolve() for m in (smg_grpc_proto, smg_grpc_servicer)]
shadowed = [str(p) for p in paths if repo not in p.parents]
assert not shadowed, f'smg gRPC modules shadowed by site-packages copies: {shadowed}'
print('smg gRPC modules resolve to repo source: OK')
"

echo "TokenSpeed installation complete"
