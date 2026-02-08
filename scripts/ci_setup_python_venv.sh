#!/bin/bash
# Setup Python venv for CI jobs on k8s runners
# Creates a virtual environment with Python 3.12 and adds it to GITHUB_PATH
# Uses uv to manage Python independently of system packages

set -euo pipefail

# Install uv for Python management (10-100x faster than pip)
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create venv with Python 3.12 (uv manages its own Python builds)
uv venv --python 3.12 .venv

# Add to GitHub Actions PATH if running in CI
if [ -n "${GITHUB_PATH:-}" ]; then
    echo "$PWD/.venv/bin" >> "$GITHUB_PATH"
    echo "CUDA_HOME=/usr/local/cuda" >> "$GITHUB_ENV"
else
    echo "Activate venv with: source .venv/bin/activate"
fi

echo "Python venv setup complete"
