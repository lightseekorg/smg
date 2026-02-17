#!/bin/bash
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

echo "Creating OpenAI worker..."
WORKER_RESPONSE=$(curl -s -X POST http://localhost:9999/workers \
    -H "Content-Type: application/json" \
    -d "{
        \"url\": \"https://api.openai.com\",
        \"api_key\": \"$OPENAI_API_KEY\",
        \"runtime\": \"external\",
        \"disable_health_check\": true
    }")

if [[ $? -eq 0 ]]; then
    echo "✅ Worker creation request sent"
    echo "Response: $WORKER_RESPONSE"
else
    echo "❌ Failed to create worker"
fi