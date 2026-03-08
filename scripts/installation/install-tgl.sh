#!/bin/sh
set -e

# Install tgl from source.
# Usage: install-tgl.sh [path-to-tgl-src]
# Default path: /tmp/tgl-src

TGL_SRC="${1:-/tmp/tgl-src}"
cd "${TGL_SRC}/python"
pip install --no-deps --force-reinstall --editable .
