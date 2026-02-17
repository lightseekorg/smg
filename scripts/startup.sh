#!/bin/bash
set -e

# ── 1. Read secrets FIRST ────────────────────────────────────────────────
echo "Reading OpenAI API key from secret file..."
if [[ -f "/mnt/openai-secrets/openai-passthrough-api-key" ]]; then
    OPENAI_API_KEY=$(tr -d '\n\r' < /mnt/openai-secrets/openai-passthrough-api-key)
    export OPENAI_API_KEY

    if [[ ${#OPENAI_API_KEY} -eq 0 ]]; then
        echo "ERROR: OPENAI_API_KEY is empty!"
        exit 1
    fi

    echo "OPENAI_API_KEY exported successfully (length: ${#OPENAI_API_KEY})"
    echo "export OPENAI_API_KEY=\"$OPENAI_API_KEY\"" >> /etc/environment
    echo "export OPENAI_API_KEY=\"$OPENAI_API_KEY\"" >> /etc/bash.bashrc
    echo "export OPENAI_API_KEY=\"$OPENAI_API_KEY\"" >> /root/.bashrc
    echo "✅ API key made available to all container shells"
else
    echo "ERROR: OpenAI API key file not found"
    exit 1
fi

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

# ── 6. Keep container alive ─────────────────────────────────────────────
wait $APP_PID