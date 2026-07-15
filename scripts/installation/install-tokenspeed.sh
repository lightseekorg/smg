#!/bin/sh
set -e

# Install tokenspeed from source.
# Usage: install-tokenspeed.sh [path-to-tokenspeed-src]
# Default path: /tmp/tokenspeed-src
#
# Only used when the engine image is built with ENGINE_REPO set (build the
# engine from source into a runner base). The published lightseekorg/tokenspeed
# images already bake the engine in, so the default release path does not run
# this script.

TS_SRC="${1:-/tmp/tokenspeed-src}"
cd "${TS_SRC}/python"
pip install --no-deps --force-reinstall --editable .
