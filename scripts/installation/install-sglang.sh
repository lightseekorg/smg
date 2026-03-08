#!/bin/sh
set -e

# Install sglang from source.
# Usage: install-sglang.sh [path-to-sglang-src]
# Default path: /tmp/sglang-src

SGL_SRC="${1:-/tmp/sglang-src}"
cd "${SGL_SRC}/python"
pip install --no-deps --force-reinstall --editable .
