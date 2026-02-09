#!/bin/bash
# Regenerate Go gRPC stubs from proto files.
# Run this after modifying any .proto files in grpc_client/proto/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PROTO_DIR="$REPO_ROOT/grpc_client/proto"
GO_OUT_DIR="$REPO_ROOT/bindings/golang/internal/proto"

# Only generate Go stubs for sglang_scheduler (the only proto used by Go binding)
PROTO_FILE="$PROTO_DIR/sglang_scheduler.proto"

if [[ ! -f "$PROTO_FILE" ]]; then
    echo "Error: Proto file not found: $PROTO_FILE"
    exit 1
fi

echo "Generating Go stubs from: $PROTO_FILE"
echo "Output directory: $GO_OUT_DIR"

mkdir -p "$GO_OUT_DIR"

protoc \
    --proto_path="$PROTO_DIR" \
    --go_out="$GO_OUT_DIR" \
    --go_opt=paths=source_relative \
    --go-grpc_out="$GO_OUT_DIR" \
    --go-grpc_opt=paths=source_relative \
    "$PROTO_FILE"

echo "Generated:"
ls -la "$GO_OUT_DIR"/*.go
