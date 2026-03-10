#!/bin/sh
set -e

# Install sglang from source.
# Usage: install-sglang.sh [path-to-sglang-src]
# Default path: /tmp/sglang-src

SGL_SRC="${1:-/tmp/sglang-src}"
cd "${SGL_SRC}/python"
pip install --no-deps --force-reinstall --editable .

# Pin gRPC packages to 1.78.0 — grpcio 1.78.1 was yanked from PyPI 
# TODO: remove when new SGLang 0.5.10 is released
pip install --force-reinstall grpcio==1.78.0 grpcio-reflection==1.78.0 grpcio-tools==1.78.0 grpcio-health-checking==1.78.0
