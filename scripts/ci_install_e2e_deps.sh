#!/bin/bash
# Install e2e test dependencies
# Usage: ci_install_e2e_deps.sh [extra_deps...]

set -euo pipefail

# Activate venv if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "Installing e2e test dependencies..."
python3 -m pip install pytest pytest-rerunfailures httpx openai anthropic grpcio grpcio-health-checking numpy pandas

# Install any extra dependencies passed as arguments
if [ $# -gt 0 ]; then
    echo "Installing extra dependencies: $@"
    python3 -m pip --no-cache-dir install --upgrade "$@"
fi

echo "E2E test dependencies installed"
