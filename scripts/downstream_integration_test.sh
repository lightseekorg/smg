#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

print_usage() {
    echo "Downstream Integration Test Runner"
    echo ""
    echo "Usage:"
    echo "  $0 --oracle-dsn <dsn> --oracle-user <user> --oracle-password <password> --api-key <key> [--oracle-client-lib <dir>]"
    echo ""
    echo "Options:"
    echo "  --oracle-dsn              Oracle DSN"
    echo "  --oracle-external-auth    Enable Oracle external auth"
    echo "  --oracle-client-lib       Oracle client library directory"
    echo "  --oracle-user             Oracle username"
    echo "  --oracle-password         Oracle password"
    echo "  --api-key                 External worker API key"
    echo "  --port                    SMG port (default: 9999)"
    echo "  --log-level               Log level (default: debug)"
    echo "  -h, --help                Show help"
    echo ""
    echo "Examples:"
    echo "  $0 --oracle-dsn \"tcps://...\" --oracle-user ADMIN --oracle-password 'secret'"
    echo "  $0 --oracle-dsn \"tcps://...\" --oracle-external-auth"
}

ORACLE_DSN=""
ORACLE_USER=""
ORACLE_PASSWORD=""
ORACLE_EXTERNAL_AUTH=false
ORACLE_CLIENT_LIB=""
API_KEY=""
SMG_PORT="9999"
SMG_LOG_LEVEL="debug"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --oracle-dsn)
            ORACLE_DSN="$2"
            shift 2
            ;;
        --oracle-user)
            ORACLE_USER="$2"
            shift 2
            ;;
        --oracle-password)
            ORACLE_PASSWORD="$2"
            shift 2
            ;;
        --oracle-external-auth)
            ORACLE_EXTERNAL_AUTH=true
            shift 1
            ;;
        --oracle-client-lib)
            ORACLE_CLIENT_LIB="$2"
            shift 2
            ;;
        --api-key)
            API_KEY="$2"
            shift 2
            ;;
        --port)
            SMG_PORT="$2"
            shift 2
            ;;
        --log-level)
            SMG_LOG_LEVEL="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Use --help for usage."
            exit 1
            ;;
    esac
done

if [[ -z "$ORACLE_DSN" ]]; then
    echo "Missing required --oracle-dsn."
    echo "Use --help for usage."
    exit 1
fi

if [[ "$ORACLE_EXTERNAL_AUTH" == "true" ]]; then
    if [[ -n "$ORACLE_USER" || -n "$ORACLE_PASSWORD" ]]; then
        echo "Warning: ignoring --oracle-user/--oracle-password for external auth."
        ORACLE_USER=""
        ORACLE_PASSWORD=""
    fi
else
    if [[ -z "$ORACLE_USER" || -z "$ORACLE_PASSWORD" ]]; then
        echo "Missing required --oracle-user/--oracle-password for password auth."
        echo "Use --help for usage."
        exit 1
    fi
fi

if [[ -z "$API_KEY" ]]; then
    echo "Missing required --api-key."
    echo "Use --help for usage."
    exit 1
fi

cd "$ROOT_DIR"

if [[ -n "$ORACLE_CLIENT_LIB" ]]; then
    export DYLD_LIBRARY_PATH="$ORACLE_CLIENT_LIB${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
    export DYLD_FALLBACK_LIBRARY_PATH="$ORACLE_CLIENT_LIB${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
fi

LOG_FILE="$ROOT_DIR/.smg_downstream.log"

SMG_ARGS=(
    --enable-igw
    --port "$SMG_PORT"
    --history-backend oracle
    --oracle-dsn "$ORACLE_DSN"
    --log-level "$SMG_LOG_LEVEL"
)

if [[ "$ORACLE_EXTERNAL_AUTH" == "true" ]]; then
    SMG_ARGS+=(--oracle-external-auth)
else
    SMG_ARGS+=(--oracle-user "$ORACLE_USER" --oracle-password "$ORACLE_PASSWORD")
fi

cargo run --bin smg -- "${SMG_ARGS[@]}" >"$LOG_FILE" 2>&1 &
SMG_PID=$!

cleanup() {
    if kill -0 "$SMG_PID" 2>/dev/null; then
        kill "$SMG_PID" 2>/dev/null || true
        wait "$SMG_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

wait_for_ready() {
    local timeout_secs=1200
    local url="$1"
    local start
    start=$(date +%s)

    while true; do
        if curl -s -f "$url" >/dev/null; then
            return 0
        fi

        if ! kill -0 "$SMG_PID" 2>/dev/null; then
            echo "SMG process exited before becoming ready."
            return 1
        fi

        local now
        now=$(date +%s)
        if (( now - start >= timeout_secs )); then
            echo "Timed out waiting for SMG to become ready."
            return 1
        fi

        sleep 1
    done
}

echo "Waiting for SMG to become ready (timeout: 1200s)..."
wait_for_ready "http://localhost:${SMG_PORT}/health"
echo "SMG is ready."

echo "Registering external worker..."
if ! curl -s -f -X POST "http://localhost:${SMG_PORT}/workers" \
    -H "Content-Type: application/json" \
    -d "{\"url\":\"https://api.openai.com\",\"api_key\":\"${API_KEY}\",\"runtime\":\"external\",\"disable_health_check\":true}" \
    >/dev/null; then
    echo "Failed to register external worker. See $LOG_FILE for SMG logs."
    exit 1
fi
echo "External worker registered."

STORE_ID="test-store-$(date +%Y%m%d)-$RANDOM"
META_TITLE="auto-title-$RANDOM"
META_MODEL="gpt-5.1"
META_USER="user-$RANDOM"

echo "Creating conversation..."
CREATE_RESP=$(curl -s -f -X POST "http://localhost:${SMG_PORT}/v1/conversations" \
    -H "Content-Type: application/json" \
    -H "opc-conversation-store-id: ${STORE_ID}" \
    -d "{\"metadata\":{\"title\":\"${META_TITLE}\",\"model\":\"${META_MODEL}\",\"user_id\":\"${META_USER}\"}}")

if ! command -v jq >/dev/null 2>&1; then
    echo "jq is required for JSON checks. Install jq and re-run."
    exit 1
fi

CONVERSATION_ID=$(echo "$CREATE_RESP" | jq -r '.id // empty')
if [[ -z "$CONVERSATION_ID" ]]; then
    echo "Failed to parse conversation id. Response: $CREATE_RESP"
    exit 1
fi

echo "Adding conversation item..."
ITEM_CONTENT="Tell me about San Andreas GTA"
ADD_RESP=$(curl -s -f -X POST "http://localhost:${SMG_PORT}/v1/conversations/${CONVERSATION_ID}/items" \
    -H "Content-Type: application/json" \
    -H "opc-conversation-store-id: ${STORE_ID}" \
    -d "{\"items\":[{\"item_type\":\"message\",\"role\":\"user\",\"content\":\"${ITEM_CONTENT}\"}]}" \
)

MESSAGE_ID=$(echo "$ADD_RESP" | jq -r '.data[0].id // empty')
if [[ -z "$MESSAGE_ID" ]]; then
    echo "Failed to parse message id. Response: $ADD_RESP"
    exit 1
fi
echo "Conversation item added."

echo "Getting conversation items..."
ITEMS_RESP=$(curl -s -f "http://localhost:${SMG_PORT}/v1/conversations/${CONVERSATION_ID}/items" \
    -H "opc-conversation-store-id: ${STORE_ID}")
if ! echo "$ITEMS_RESP" | jq -e --arg mid "$MESSAGE_ID" --arg content "$ITEM_CONTENT" '.data | any(.id == $mid and .type == "message" and .role == "user" and .content == $content)' >/dev/null; then
    echo "Conversation items check failed."
    exit 1
fi
echo "Conversation items check passed."

echo "Getting specific item..."
ITEM_RESP=$(curl -s -f "http://localhost:${SMG_PORT}/v1/conversations/${CONVERSATION_ID}/items/${MESSAGE_ID}" \
    -H "opc-conversation-store-id: ${STORE_ID}")
if ! echo "$ITEM_RESP" | jq -e --arg mid "$MESSAGE_ID" --arg content "$ITEM_CONTENT" '.id == $mid and .type == "message" and .role == "user" and .content == $content' >/dev/null; then
    echo "Specific item check failed."
    exit 1
fi
echo "Specific item check passed."

echo "Deleting item..."
curl -s -f -X DELETE "http://localhost:${SMG_PORT}/v1/conversations/${CONVERSATION_ID}/items/${MESSAGE_ID}" \
    -H "opc-conversation-store-id: ${STORE_ID}" >/dev/null

DELETE_CHECK_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${SMG_PORT}/v1/conversations/${CONVERSATION_ID}/items/${MESSAGE_ID}" \
    -H "opc-conversation-store-id: ${STORE_ID}")
if [[ "$DELETE_CHECK_CODE" != "404" && "$DELETE_CHECK_CODE" != "410" ]]; then
    echo "Delete check failed (expected 404/410, got $DELETE_CHECK_CODE)."
    exit 1
fi
echo "Delete check passed."

echo "Generating response..."
RESP_CREATE=$(curl -s -f "http://localhost:${SMG_PORT}/v1/responses" \
    -H "Content-Type: application/json" \
    -H "opc-conversation-store-id: ${STORE_ID}" \
    -H "Authorization: Bearer ${API_KEY}" \
    -d "{\"model\":\"${META_MODEL}\",\"input\":\"Tell me about San Jose\"}")

RESPONSE_ID=$(echo "$RESP_CREATE" | jq -r '.id // empty')
if [[ -z "$RESPONSE_ID" ]]; then
    echo "Failed to parse response id. Response: $RESP_CREATE"
    exit 1
fi
RESPONSE_MSG_ID=$(echo "$RESP_CREATE" | jq -r '.output[0].id // empty')
echo "Response generated."

echo "Getting response..."
RESP_GET=$(curl -s -f "http://localhost:${SMG_PORT}/v1/responses/${RESPONSE_ID}" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${API_KEY}" \
    -H "opc-conversation-store-id: ${STORE_ID}")
if [[ -z "$RESPONSE_MSG_ID" ]]; then
    echo "Failed to parse response message id. Response: $RESP_CREATE"
    exit 1
fi

if ! echo "$RESP_GET" | jq -e --arg mid "$RESPONSE_MSG_ID" 'type=="array" and any(.id == $mid)' >/dev/null; then
    echo "Get response check failed."
    exit 1
fi
echo "Get response check passed."
