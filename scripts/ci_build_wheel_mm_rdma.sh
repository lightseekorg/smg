#!/bin/bash
# Build the mm-rdma smg wheel for the RDMA e2e, natively on the bare Ubuntu
# runner (GNU libstdc++).
#
# Why a separate build (not the shared ci_build_wheel.sh): the shared wheel is
# cross-compiled with `maturin --zig`, which ships LLVM libc++. nixl-sys (pulled
# by the mm-rdma feature) compiles a C++ stub that references GNU libstdc++
# (std::cerr / _ZSt4cerr), so it must be linked against GNU libstdc++ or the
# wheel fails to import ("undefined symbol: _ZSt4cerr"). Building natively with
# the runner's GNU toolchain links the C++ stub cleanly, and the wheel runs on
# the GPU e2e runner unchanged (same runner image, same libstdc++.so.6).
#
# Assumes ./.github/actions/setup-rust already ran (Rust + build-essential/g++ +
# protoc). This wheel is CI-only (installed by the RDMA e2e job), so a plain
# linux tag is fine.
set -euxo pipefail

# libclang for nixl-sys bindgen; setup-rust installs build-essential (g++ ->
# libstdc++) but not clang.
export DEBIAN_FRONTEND=noninteractive
if command -v sudo >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends libclang-dev
else
    apt-get update
    apt-get install -y --no-install-recommends libclang-dev
fi

# setup-rust adds cargo to GITHUB_PATH; source for good measure (and local runs).
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
fi
export RUSTC_WRAPPER="${RUSTC_WRAPPER:-sccache}"

python3 -m pip install --upgrade pip maturin

echo "Building mm-rdma wheel (native, GNU libstdc++)..."
cd bindings/python
# abi3 wheel (pyo3 abi3-py38) -> one build works for all 3.8+.
# --manylinux off: CI-only wheel; skip auditwheel's manylinux policy and
# dynamically link libstdc++.so.6 (present on the GPU e2e runner).
maturin build \
    --profile ci \
    --features vendored-openssl,mm-rdma \
    --manylinux off \
    --out dist
echo "mm-rdma wheel: OK"
ls -lh dist/
