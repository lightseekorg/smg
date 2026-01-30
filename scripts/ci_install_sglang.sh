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

# Optionally check out the latest release tag
SGLANG_USE_LATEST_TAG="${SGLANG_USE_LATEST_TAG:-0}"
if [ "$SGLANG_USE_LATEST_TAG" = "1" ]; then
    cd "$SGLANG_DIR"
    LATEST_TAG=$(git describe --tags $(git rev-list --tags --max-count=1))
    echo "Checking out latest SGLang tag: $LATEST_TAG"
    git checkout "$LATEST_TAG"
    cd -
fi

# Install SGLang dependencies
echo "Installing SGLang dependencies..."
sudo apt update
cd "$SGLANG_DIR"

# Handle script path differences between sglang versions
if [ -f "scripts/ci/cuda/ci_install_dependency.sh" ]; then
    INSTALL_SCRIPT="scripts/ci/cuda/ci_install_dependency.sh"
elif [ -f "scripts/ci_install_dependency.sh" ]; then
    INSTALL_SCRIPT="scripts/ci_install_dependency.sh"
else
    echo "ERROR: Could not find ci_install_dependency.sh in sglang repo"
    find scripts -name "ci_install_dependency*" 2>/dev/null || true
    exit 1
fi

echo "Using install script: $INSTALL_SCRIPT"
sudo CUDA_HOME=/usr/local/cuda IS_BLACKWELL=1 --preserve-env=PATH bash "$INSTALL_SCRIPT"

echo "SGLang installation complete"
