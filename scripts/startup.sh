#!/bin/bash
# Initialize environment variables from mounted secrets

set -e

# Function to read secret file and set env var
set_secret_env() {
    local secret_file="$1"
    local env_var="$2"

    if [[ -f "$secret_file" ]]; then
        local secret_value
        secret_value=$(cat "$secret_file" | tr -d '\n\r')
        export "$env_var=$secret_value"
        echo "Set $env_var from $secret_file"
    else
        echo "Warning: Secret file $secret_file not found"
    fi
}

# Read secrets from mounted volume
set_secret_env "/mnt/openai-secrets/openai-passthrough-api-key" "OPENAI_API_KEY"

echo "Secret initialization complete"

# Execute the main application (python3 -m smg.launch_router) with all arguments
exec python3 -m smg.launch_router "$@"
