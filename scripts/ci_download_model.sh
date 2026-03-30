#!/usr/bin/env bash
# Download HuggingFace models with file-locking to prevent concurrent download
# conflicts when multiple CI pods share the same NVMe cache on a node.
#
# Usage:
#   ci_download_model.sh --gpu-tier <tier>                   # Download all models for a GPU tier
#   ci_download_model.sh <model_id> [<model_id> ...]         # Download specific models
#
# Example:
#   ci_download_model.sh --gpu-tier 1
#   ci_download_model.sh Qwen/Qwen2.5-7B-Instruct meta-llama/Llama-3.1-8B-Instruct

set -euo pipefail

export HF_HOME="${HF_HOME:-/models}"
LOCK_DIR="${HF_HOME}/.locks"
MAX_RETRIES=3
RETRY_DELAY=30

resolve_models_for_tier() {
    local tier="$1"
    # ROUTER_LOCAL_MODEL_PATH must be unset so model_specs doesn't resolve to local paths
    ROUTER_LOCAL_MODEL_PATH="" python3 -c "
import sys
from e2e_test.infra.model_specs import MODEL_SPECS
for model_id, spec in MODEL_SPECS.items():
    if spec['tp'] == int(sys.argv[1]):
        print(model_id)
" "$tier"
}

download_model() {
    local model_id="$1"
    local model_dir="${HF_HOME}/hub/models--${model_id//\//--}"
    local lock_file="${LOCK_DIR}/${model_id//\//_}.lock"

    echo "Acquiring lock for ${model_id}..."
    (
        # Serialize downloads across pods sharing the same volume
        flock -w 1800 200 || { echo "ERROR: Timed out waiting for lock on ${model_id}"; exit 1; }

        # Let huggingface-cli handle its own cache detection — it verifies file
        # integrity and skips already-complete downloads. We don't check snapshots/
        # ourselves because a killed download can leave it partially populated.
        echo "Downloading ${model_id} to ${HF_HOME}..."
        local attempt=0
        while [ $attempt -lt $MAX_RETRIES ]; do
            attempt=$((attempt + 1))
            if huggingface-cli download "$model_id" --quiet 2>&1; then
                echo "Successfully downloaded ${model_id}."
                exit 0
            fi
            echo "Download attempt ${attempt}/${MAX_RETRIES} failed for ${model_id}."
            if [ $attempt -lt $MAX_RETRIES ]; then
                # Clean up stale HF hub lock files (both legacy and new locations)
                find "${model_dir}" -name "*.lock" -type f -delete 2>/dev/null || true
                local hf_lock_dir="${HF_HOME}/hub/.locks/models--${model_id//\//--}"
                find "${hf_lock_dir}" -name "*.lock" -type f -delete 2>/dev/null || true
                echo "Retrying in ${RETRY_DELAY}s..."
                sleep "$RETRY_DELAY"
            fi
        done
        echo "ERROR: Failed to download ${model_id} after ${MAX_RETRIES} attempts."
        exit 1
    ) 200>"$lock_file"
}

# Parse arguments
models=()
if [ "${1:-}" = "--gpu-tier" ]; then
    if [ -z "${2:-}" ]; then
        echo "Usage: $0 --gpu-tier <tier>"
        exit 1
    fi
    while IFS= read -r model; do
        models+=("$model")
    done < <(resolve_models_for_tier "$2")
    if [ ${#models[@]} -eq 0 ]; then
        echo "ERROR: No models resolved for GPU tier $2"
        exit 1
    fi
    echo "Resolved ${#models[@]} model(s) for GPU tier $2: ${models[*]}"
elif [ $# -gt 0 ]; then
    models=("$@")
else
    echo "Usage: $0 --gpu-tier <tier> | <model_id> [<model_id> ...]"
    exit 1
fi

mkdir -p "$LOCK_DIR"

for model_id in "${models[@]}"; do
    download_model "$model_id"
done
