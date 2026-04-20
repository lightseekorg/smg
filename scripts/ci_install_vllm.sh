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

# Pin vLLM below 0.19.1 AND transformers below 5.0 (belt and suspenders).
#
# What broke: e2e-1gpu-embeddings (vllm) started failing on main after
# vllm 0.19.1 was published. The failing combo was
#   vllm==0.19.1 + transformers==5.5.4
# which reports self-similarity ~0.33 (expected ~1.0) on
# intfloat/e5-mistral-7b-instruct. Last-known-good combo was
#   vllm==0.19.0 + transformers==4.57.6   (run 24591985132, 82a3fb1a)
# regression first appeared in run 24608587304 (dcede344). Job showing
# the failure: https://github.com/lightseekorg/smg/actions/runs/24644816475/job/72068881582
#
# Why both pins: we have not isolated whether vllm 0.19.1 regressed on
# its own or transformers 5.x is the culprit — vllm 0.19.1's wheel
# permits transformers 5.5.1+ (metadata: transformers!=5.0.*,...,!=5.5.0,>=4.56.0),
# while vllm 0.19.0's wheel hard-caps transformers<5 (metadata: transformers<5,>=4.56.0).
# Pinning both guarantees we restore the last-known-good regardless of
# which side actually broke it. See PR description for follow-up work.
echo "Installing vLLM..."
uv pip install "vllm<0.19.1" "transformers<5"

# Install nixl for vLLM PD disaggregation (NIXL KV transfer)
echo "Installing nixl..."
uv pip install nixl

# Install gRPC packages from source (not PyPI) so PR changes are always tested
echo "Installing smg-grpc-proto and smg-grpc-servicer from source..."
uv pip install -e crates/grpc_client/python/
uv pip install -e grpc_servicer/

echo "vLLM installation complete"
