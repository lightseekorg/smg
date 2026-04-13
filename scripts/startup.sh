#!/bin/bash
set -e

persist_api_key() {
    local var_name="$1"
    local value="$2"

    echo "export ${var_name}=\"${value}\"" >> /etc/environment
    echo "export ${var_name}=\"${value}\"" >> /etc/bash.bashrc
    echo "export ${var_name}=\"${value}\"" >> /root/.bashrc
}

load_api_key() {
    local var_name="$1"
    local secret_path="$2"
    local required="${3:-false}"
    local current_value="${!var_name:-}"

    echo "Loading ${var_name}..."
    if [[ -n "${current_value}" ]]; then
        current_value=$(printf '%s' "${current_value}" | tr -d '\n\r')
        echo "${var_name} provided via environment"
    elif [[ -f "${secret_path}" ]]; then
        current_value=$(tr -d '\n\r' < "${secret_path}")
        echo "${var_name} loaded from ${secret_path}"
    else
        if [[ "${required}" == "true" ]]; then
            echo "ERROR: ${var_name} not found in environment or ${secret_path}"
            exit 1
        fi
        echo "${var_name} not provided, skipping"
        return 1
    fi

    if [[ ${#current_value} -eq 0 ]]; then
        if [[ "${required}" == "true" ]]; then
            echo "ERROR: ${var_name} is empty!"
            exit 1
        fi
        echo "${var_name} is empty, skipping"
        return 1
    fi

    printf -v "${var_name}" '%s' "${current_value}"
    export "${var_name}"
    echo "${var_name} exported successfully (length: ${#current_value})"
    persist_api_key "${var_name}" "${current_value}"
    echo "✅ ${var_name} made available to all container shells"
}

# ── 1. Read secrets FIRST ────────────────────────────────────────────────
load_api_key "OPENAI_API_KEY" "/mnt/openai-secrets/openai-passthrough-api-key" "true"
load_api_key "XAI_API_KEY" "/mnt/xai-secrets/xai-api-key" "true"

# ── 2. Oracle configuration ──────────────────────────────────────────────
cat > /opt/oracle/instantclient/network/admin/sqlnet.ora <<'EOF'
TOKEN_AUTH=OCI_TOKEN
SSL_SERVER_DN_MATCH=YES
EOF

# ── 3. OCI token - get FIRST token synchronously before server starts ────
echo "Getting initial OCI IAM token..."
oci iam db-token get --auth instance_principal
echo "✅ Initial token acquired"

# Start background refresh loop
( while true; do sleep 3000; oci iam db-token get --auth instance_principal; done ) &

# ── 4. Start server ─────────────────────────────────────────────────────
echo "Starting SMG application..."
python3 -m smg.launch_router "$@" &
APP_PID=$!

# ── 5. Setup worker in background ───────────────────────────────────────
bash /app/setup-worker.sh &
SETUP_PID=$!

wait "${SETUP_PID}"

# ── 6. Keep container alive ─────────────────────────────────────────────
wait "${APP_PID}"
