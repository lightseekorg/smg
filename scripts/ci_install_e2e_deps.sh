#!/bin/bash
# Install e2e test dependencies
# Usage: ci_install_e2e_deps.sh [extra_deps...]

set -euo pipefail

# Activate venv if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "Installing e2e test dependencies..."
python3 -m pip install e2e_test/

# Install SmgClient (pure Python client for cross-SDK parity testing)
echo "Installing smg-client..."
python3 -m pip install clients/python/

# Install any extra dependencies passed as arguments
if [ $# -gt 0 ]; then
    echo "Installing extra dependencies: $@"
    python3 -m pip --no-cache-dir install --upgrade "$@"
fi

# Pin grpcio-health-checking to the protobuf-6 stable line.
#
# Engine setup (sglang's `uv pip install --prerelease=allow` in
# ci_install_sglang.sh) can pull a prerelease wheel such as
# grpcio-health-checking==1.82.0rc1, whose bundled protobuf gencode (7.x) is
# newer than the protobuf 6.x runtime the engine stack pins. e2e gRPC workers
# then die at `import grpc_health.v1.health_pb2` with "Detected incompatible
# Protobuf Gencode/Runtime versions ... gencode 7.x runtime 6.x". The 1.81.x
# line caps protobuf<7 and is built against the 6.x gencode, so it stays
# compatible. This runs last to override whatever engine/extra-dep installs
# selected. Drop the cap once the stack moves to a protobuf 7 runtime.
python3 -m pip install "grpcio-health-checking==1.81.*"

echo "E2E test dependencies installed"
