#!/bin/bash
# Install SGLang and its dependencies for CI
# Clones SGLang repo and runs the CUDA dependency installation script

set -euo pipefail

# Activate venv if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

SGLANG_DIR="${1:-sglang}"

# Clone SGLang if not already present
if [ ! -d "$SGLANG_DIR" ]; then
    echo "Cloning SGLang repository..."
    git clone https://github.com/sgl-project/sglang.git "$SGLANG_DIR"
fi

# Install SGLang dependencies
echo "Installing SGLang dependencies..."
sudo apt update
cd "$SGLANG_DIR"
sudo CUDA_HOME=/usr/local/cuda IS_BLACKWELL=1 --preserve-env=PATH bash scripts/ci/cuda/ci_install_dependency.sh

echo "SGLang installation complete"
