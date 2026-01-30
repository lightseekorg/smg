#!/bin/bash
# Setup Python venv for CI jobs on k8s runners
# Creates a virtual environment and adds it to GITHUB_PATH

set -euo pipefail

# Install pip & venv if needed
if ! command -v pip3 &> /dev/null || ! python3 -m venv --help &> /dev/null; then
    echo "Installing pip and venv..."
    sudo apt update
    sudo apt install -y python3-pip python3-venv
fi

# Create venv
python3 -m venv .venv

# Add to GitHub Actions PATH if running in CI
if [ -n "${GITHUB_PATH:-}" ]; then
    echo "$PWD/.venv/bin" >> "$GITHUB_PATH"
    echo "CUDA_HOME=/usr/local/cuda" >> "$GITHUB_ENV"
else
    echo "Activate venv with: source .venv/bin/activate"
fi

echo "Python venv setup complete"
