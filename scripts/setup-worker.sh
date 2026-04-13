#!/bin/bash
set -e

# Wait for server readiness and create OpenAI worker

echo "Waiting for server to be ready..."
TIMEOUT=60
COUNTER=0
while ! curl -s --max-time 1 http://localhost:9999/health >/dev/null 2>&1; do
    if [ $COUNTER -ge $TIMEOUT ]; then
        echo "❌ Timeout waiting for server to be ready"
        exit 1
    fi
    echo "Waiting for server... ($COUNTER/$TIMEOUT)"
    sleep 2
    COUNTER=$((COUNTER + 2))
done
echo "✅ Server is ready"

register_external_worker() {
    local provider_name="$1"
    local provider_url="$2"
    local api_key="$3"
    local response_body
    local http_code

    if [[ -z "${api_key}" ]]; then
        echo "❌ Missing API key for ${provider_name} worker"
        return 1
    fi

    echo "Creating ${provider_name} worker..."
    response_body=$(mktemp)
    http_code=$(curl -s -o "${response_body}" -w "%{http_code}" -X POST http://localhost:9999/workers \
        -H "Content-Type: application/json" \
        -d "{
            \"url\": \"${provider_url}\",
            \"api_key\": \"${api_key}\",
            \"runtime\": \"external\",
            \"disable_health_check\": true
        }")

    if [[ "${http_code}" == "200" || "${http_code}" == "202" ]]; then
        echo "✅ ${provider_name} worker creation request sent"
        echo "Response: $(cat "${response_body}")"
        rm -f "${response_body}"
        return 0
    fi

    echo "❌ Failed to create ${provider_name} worker (HTTP ${http_code})"
    echo "Response: $(cat "${response_body}")"
    rm -f "${response_body}"
    return 1
}

register_external_worker "OpenAI" "https://api.openai.com" "${OPENAI_API_KEY:-}"
register_external_worker "xAI" "https://api.x.ai" "${XAI_API_KEY:-}"
