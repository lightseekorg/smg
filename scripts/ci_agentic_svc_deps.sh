#!/bin/bash
# CI helper for shared services (Oracle DB, Brave Search MCP).
# Usage:
#   bash ci_agentic_svc_deps.sh check [--oracle <host>] [--brave <host>]
#   bash ci_agentic_svc_deps.sh setup-oracle-client
#   bash ci_agentic_svc_deps.sh create-oracle-user <host>
#   bash ci_agentic_svc_deps.sh cleanup-oracle-user <host>

set -uo pipefail

COMMAND="${1:?Usage: ci_agentic_svc_deps.sh <command> [args...]}"
shift

check_port() {
    local name="$1" host="$2" port="$3"
    echo -n "Checking $name on $host:$port... "
    if python3 -c "import socket; s=socket.create_connection(('$host', $port), 5); s.close()" 2>/dev/null; then
        echo "ok"
    else
        echo "FAILED"
        return 1
    fi
}

cmd_check() {
    local failed=0
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --oracle)
                if [[ -n "${2:-}" && "$2" != --* ]]; then
                    check_port "Oracle DB" "$2" 1521 || failed=1; shift 2
                else
                    check_port "Oracle DB" "oracle-db" 1521 || failed=1; shift
                fi ;;
            --brave)
                if [[ -n "${2:-}" && "$2" != --* ]]; then
                    check_port "Brave Search MCP" "$2" 8080 || failed=1; shift 2
                else
                    check_port "Brave Search MCP" "brave-search" 8080 || failed=1; shift
                fi ;;
            *)  echo "Unknown service: $1"; exit 1 ;;
        esac
    done
    return $failed
}

cmd_setup_oracle_client() {
    set -e
    sudo apt-get update
    sudo apt-get install -y unzip wget
    sudo apt-get install -y libaio1t64 || sudo apt-get install -y libaio1

    LIBAIO_PATH=$(find /usr/lib -name "libaio.so*" -type f 2>/dev/null | head -1)
    INSTANT_CLIENT_DIR="$HOME/instant-client"
    INSTANT_CLIENT_ZIP="instantclient-basic-linux.x64-23.26.1.0.0.zip"

    if [ ! -d "$INSTANT_CLIENT_DIR/instantclient_23_26" ]; then
        echo "Downloading Oracle Instant Client..."
        mkdir -p "$INSTANT_CLIENT_DIR"
        (cd "$INSTANT_CLIENT_DIR" &&
            wget "https://download.oracle.com/otn_software/linux/instantclient/2326100/$INSTANT_CLIENT_ZIP" &&
            unzip "$INSTANT_CLIENT_ZIP" &&
            rm "$INSTANT_CLIENT_ZIP")
    else
        echo "Oracle Instant Client already exists, skipping download"
    fi

    if [ -n "$LIBAIO_PATH" ]; then
        cp "$LIBAIO_PATH" "$INSTANT_CLIENT_DIR/instantclient_23_26/"
        ln -sf "$INSTANT_CLIENT_DIR/instantclient_23_26/$(basename "$LIBAIO_PATH")" \
            "$INSTANT_CLIENT_DIR/instantclient_23_26/libaio.so.1"
    fi

    echo "LD_LIBRARY_PATH=$INSTANT_CLIENT_DIR/instantclient_23_26:${LD_LIBRARY_PATH:-}" >> "$GITHUB_ENV"
}

cmd_create_oracle_user() {
    set -e
    local oracle_host="${1:-oracle-db}"
    local oracle_dsn="${oracle_host}:1521/FREEPDB1"

    pip install oracledb

    # Use last two segments of pod name (e.g. arc-runner-gpu-h100-jfvzm-m2lm8 -> JFVZM_M2LM8)
    RAW_NAME=$(echo "$HOSTNAME" | rev | cut -d'-' -f1,2 | rev | tr '[:lower:]-' '[:upper:]_')
    TEST_USER="TEST_${RAW_NAME}"
    # Prefix with 'P' so the password always starts with a letter (Oracle requirement)
    TEST_PASS="P$(openssl rand -hex 8)"
    echo "Creating Oracle test user: $TEST_USER"

    export ORA_TEST_USER="$TEST_USER"
    export ORA_TEST_PASS="$TEST_PASS"
    export ORA_DSN="$oracle_dsn"

    python3 << 'PYEOF'
import os, oracledb
user = os.environ["ORA_TEST_USER"]
pwd = os.environ["ORA_TEST_PASS"]
dsn = os.environ["ORA_DSN"]
conn = oracledb.connect(user="system", password="oracle", dsn=dsn)
cur = conn.cursor()
cur.execute(f'CREATE USER {user} IDENTIFIED BY "{pwd}" QUOTA UNLIMITED ON USERS')
cur.execute(f"GRANT CONNECT, RESOURCE TO {user}")
conn.commit()
conn.close()
print("Oracle test user created successfully")
PYEOF

    echo "ATP_USER=$TEST_USER" >> "$GITHUB_ENV"
    echo "ATP_PASSWORD=$TEST_PASS" >> "$GITHUB_ENV"
    echo "ATP_DSN=$oracle_dsn" >> "$GITHUB_ENV"
}

cmd_cleanup_oracle_user() {
    local oracle_host="${1:-oracle-db}"
    local oracle_dsn="${oracle_host}:1521/FREEPDB1"

    if [ -z "${ATP_USER:-}" ] || [ "$ATP_USER" = "system" ]; then
        echo "No test user to clean up"
        return 0
    fi

    echo "Dropping Oracle test user: $ATP_USER"
    pip install oracledb 2>/dev/null || true

    export ORA_DROP_USER="$ATP_USER"
    export ORA_DSN="$oracle_dsn"

    python3 << 'PYEOF' || echo "Warning: cleanup script failed"
import os, oracledb
try:
    user = os.environ["ORA_DROP_USER"]
    dsn = os.environ["ORA_DSN"]
    conn = oracledb.connect(user="system", password="oracle", dsn=dsn)
    cur = conn.cursor()
    cur.execute(f"DROP USER {user} CASCADE")
    conn.commit()
    conn.close()
    print("Oracle test user dropped successfully")
except Exception as e:
    print(f"Warning: failed to drop test user: {e}")
PYEOF
}

case "$COMMAND" in
    check)                cmd_check "$@" ;;
    setup-oracle-client)  cmd_setup_oracle_client ;;
    create-oracle-user)   cmd_create_oracle_user "$@" ;;
    cleanup-oracle-user)  cmd_cleanup_oracle_user "$@" ;;
    *)                    echo "Unknown command: $COMMAND"; exit 1 ;;
esac
