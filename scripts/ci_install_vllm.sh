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

# Pin vLLM below 0.19.1. vLLM 0.19.1 bundled the transformers v5 upgrade,
# which breaks intfloat/e5-mistral-7b-instruct embedding quality
# (self-similarity ~0.33 instead of ~1.0) via an EOS / last-token pooling
# regression. Last-known-good combo is:
#   vllm==0.19.0 + transformers==4.57.6   (run 24591985132, 82a3fb1a)
# Regression first appeared in run 24608587304 (dcede344). Failure signature:
# https://github.com/lightseekorg/smg/actions/runs/24644816475/job/72068881582
#
# We rely on vllm 0.19.0's own wheel metadata (transformers<5,>=4.56.0) to
# force transformers back to 4.x. Not pinning transformers separately — if
# we ship vllm we don't want to second-guess the library's own dep range;
# drop this pin whenever vllm publishes a release that re-verifies embedding
# correctness post-transformers-v5.
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
