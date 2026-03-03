#!/bin/sh
set -e

# Install trtllm from source.
# Usage: install-trtllm.sh [path-to-trtllm-src]
# Default path: /tmp/trtllm-src

TRTLLM_SRC="${1:-/tmp/trtllm-src}"
cd "${TRTLLM_SRC}"
git submodule update --init --recursive
git lfs pull
python3 ./scripts/build_wheel.py --clean
pip install --force-reinstall ./build/tensorrt_llm*.whl
