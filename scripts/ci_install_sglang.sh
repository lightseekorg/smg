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

# By default, check out the latest stable release tag.
# Set SGLANG_USE_MAIN=1 to stay on the main branch instead.
SGLANG_USE_MAIN="${SGLANG_USE_MAIN:-0}"
if [ "$SGLANG_USE_MAIN" != "1" ]; then
    cd "$SGLANG_DIR"
    git fetch --tags
    LATEST_TAG=$(git tag -l 'v*' --sort=-v:refname | head -1)
    if [ -n "$LATEST_TAG" ]; then
        echo "Checking out latest SGLang stable tag: $LATEST_TAG"
        git checkout "$LATEST_TAG"
    else
        echo "WARNING: No stable tags found, staying on default branch"
    fi
    cd -
fi

# Install SGLang dependencies
echo "Installing SGLang dependencies..."
sudo apt update
cd "$SGLANG_DIR"

# Handle script path differences between sglang versions
if [ -f "scripts/ci/cuda/ci_install_dependency.sh" ]; then
    INSTALL_SCRIPT="scripts/ci/cuda/ci_install_dependency.sh"
elif [ -f "scripts/ci/ci_install_dependency.sh" ]; then
    INSTALL_SCRIPT="scripts/ci/ci_install_dependency.sh"
elif [ -f "scripts/ci_install_dependency.sh" ]; then
    INSTALL_SCRIPT="scripts/ci_install_dependency.sh"
else
    echo "ERROR: Could not find ci_install_dependency.sh in sglang repo"
    echo "Current directory: $(pwd)"
    echo "Available scripts:"
    find scripts -name "ci_install_dependency*" -o -name "*install*" | head -20 || true
    exit 1
fi

echo "Using install script: $INSTALL_SCRIPT"
if [ ! -f "$INSTALL_SCRIPT" ]; then
    echo "ERROR: Install script not found at $INSTALL_SCRIPT"
    exit 1
fi

sudo CUDA_HOME=/usr/local/cuda IS_BLACKWELL=1 --preserve-env=PATH bash "$INSTALL_SCRIPT"

echo "SGLang installation complete"
