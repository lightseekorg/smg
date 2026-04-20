#!/bin/bash
# Install vLLM with flash-attn for CI
# Handles CUDA toolkit setup and flash-attn compilation
# Uses uv for faster package installation

set -euo pipefail

# Activate venv if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Install uv for faster package management (10-100x faster than pip)
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Using uv version: $(uv --version)"

# Pin vLLM below 0.19.1. vLLM 0.19.1 bundled a transformers v5 upgrade
# (transformers 4.57 → 5.5) which broke e5-mistral-7b-instruct embedding
# quality (self-similarity ~0.33 instead of ~1.0). Last-known-good combo
# is vllm==0.19.0 / transformers==4.57.6 per run 24591985132 (commit
# 82a3fb1a); regression first seen in run 24608587304 (commit dcede344)
# once vllm 0.19.1 started resolving. See run 24644816475 / job
# 72068881582 for the failure signature.
echo "Installing vLLM..."
uv pip install "vllm<0.19.1"

# Install nixl for vLLM PD disaggregation (NIXL KV transfer)
echo "Installing nixl..."
uv pip install nixl

# Install gRPC packages from source (not PyPI) so PR changes are always tested
echo "Installing smg-grpc-proto and smg-grpc-servicer from source..."
uv pip install -e crates/grpc_client/python/
uv pip install -e grpc_servicer/

echo "vLLM installation complete"
