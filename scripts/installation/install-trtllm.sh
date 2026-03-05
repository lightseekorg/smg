#!/bin/sh
set -e

# Install trtllm from source.
# Usage: install-trtllm.sh [path-to-trtllm-src]
# Default path: /tmp/trtllm-src

if ! command -v git-lfs >/dev/null 2>&1; then
  echo "git-lfs is not installed. Installing..." >&2
  apt-get update && apt-get install -y --no-install-recommends git-lfs && rm -rf /var/lib/apt/lists/*
fi

TRTLLM_SRC="${1:-/tmp/trtllm-src}"
cd "${TRTLLM_SRC}"
git submodule update --init --recursive
git lfs pull
python3 ./scripts/build_wheel.py --clean
pip install --force-reinstall ./build/tensorrt_llm*.whl
