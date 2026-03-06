#!/bin/sh
set -e

# Install vllm from source.
# Usage: install-vllm.sh [path-to-vllm-src]
# Default path: /tmp/vllm-src

VLLM_SRC="${1:-/tmp/vllm-src}"
cd "${VLLM_SRC}"
VLLM_USE_PRECOMPILED=1 pip install --force-reinstall --editable .
