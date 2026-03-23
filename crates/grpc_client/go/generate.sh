#!/bin/bash
set -e

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTO_SOURCE_DIR="$SCRIPT_DIR/../proto"
GO_PROTO_DIR="$SCRIPT_DIR/proto"
OUTPUT_DIR="$SCRIPT_DIR/generated"

# Ensure directories exist
mkdir -p "$GO_PROTO_DIR"
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/common"
mkdir -p "$OUTPUT_DIR/sglang_encoder"
mkdir -p "$OUTPUT_DIR/sglang_scheduler"
mkdir -p "$OUTPUT_DIR/trtllm"
mkdir -p "$OUTPUT_DIR/vllm"

# Copy proto files
echo "Copying proto files..."
cp "$PROTO_SOURCE_DIR"/*.proto "$GO_PROTO_DIR"/

# Find tools in PATH or GOPATH
PROTOC_GEN_GO=$(which protoc-gen-go || echo "$(go env GOPATH)/bin/protoc-gen-go")
PROTOC_GEN_GO_GRPC=$(which protoc-gen-go-grpc || echo "$(go env GOPATH)/bin/protoc-gen-go-grpc")

if [ ! -f "$PROTOC_GEN_GO" ]; then
    echo "Error: protoc-gen-go not found. Please install with: go install google.golang.org/protobuf/cmd/protoc-gen-go@latest"
    exit 1
fi

if [ ! -f "$PROTOC_GEN_GO_GRPC" ]; then
    echo "Error: protoc-gen-go-grpc not found. Please install with: go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest"
    exit 1
fi

# Common mappings for imports
MAPPINGS="--go_opt=Mcommon.proto=github.com/lightseek/smg/crates/grpc_client/go/generated/common"
MAPPINGS_GRPC="--go-grpc_opt=Mcommon.proto=github.com/lightseek/smg/crates/grpc_client/go/generated/common"

echo "Generating Go code..."

cd "$GO_PROTO_DIR"

# Common
protoc \
  --plugin=protoc-gen-go="$PROTOC_GEN_GO" \
  --proto_path=. \
  --go_out="$OUTPUT_DIR/common" \
  --go_opt=paths=source_relative \
  --go_opt=Mcommon.proto=github.com/lightseek/smg/crates/grpc_client/go/generated/common \
  common.proto

# SGLang Encoder
protoc \
  --plugin=protoc-gen-go="$PROTOC_GEN_GO" \
  --plugin=protoc-gen-go-grpc="$PROTOC_GEN_GO_GRPC" \
  --proto_path=. \
  --go_out="$OUTPUT_DIR/sglang_encoder" \
  --go_opt=paths=source_relative \
  $MAPPINGS \
  --go_opt=Msglang_encoder.proto=github.com/lightseek/smg/crates/grpc_client/go/generated/sglang_encoder \
  --go-grpc_out="$OUTPUT_DIR/sglang_encoder" \
  --go-grpc_opt=paths=source_relative \
  $MAPPINGS_GRPC \
  --go-grpc_opt=Msglang_encoder.proto=github.com/lightseek/smg/crates/grpc_client/go/generated/sglang_encoder \
  sglang_encoder.proto

# SGLang Scheduler
protoc \
  --plugin=protoc-gen-go="$PROTOC_GEN_GO" \
  --plugin=protoc-gen-go-grpc="$PROTOC_GEN_GO_GRPC" \
  --proto_path=. \
  --go_out="$OUTPUT_DIR/sglang_scheduler" \
  --go_opt=paths=source_relative \
  $MAPPINGS \
  --go_opt=Msglang_scheduler.proto=github.com/lightseek/smg/crates/grpc_client/go/generated/sglang_scheduler \
  --go-grpc_out="$OUTPUT_DIR/sglang_scheduler" \
  --go-grpc_opt=paths=source_relative \
  $MAPPINGS_GRPC \
  --go-grpc_opt=Msglang_scheduler.proto=github.com/lightseek/smg/crates/grpc_client/go/generated/sglang_scheduler \
  sglang_scheduler.proto

# TRT-LLM Service
protoc \
  --plugin=protoc-gen-go="$PROTOC_GEN_GO" \
  --plugin=protoc-gen-go-grpc="$PROTOC_GEN_GO_GRPC" \
  --proto_path=. \
  --go_out="$OUTPUT_DIR/trtllm" \
  --go_opt=paths=source_relative \
  $MAPPINGS \
  --go_opt=Mtrtllm_service.proto=github.com/lightseek/smg/crates/grpc_client/go/generated/trtllm \
  --go-grpc_out="$OUTPUT_DIR/trtllm" \
  --go-grpc_opt=paths=source_relative \
  $MAPPINGS_GRPC \
  --go-grpc_opt=Mtrtllm_service.proto=github.com/lightseek/smg/crates/grpc_client/go/generated/trtllm \
  trtllm_service.proto

# vLLM Engine
protoc \
  --plugin=protoc-gen-go="$PROTOC_GEN_GO" \
  --plugin=protoc-gen-go-grpc="$PROTOC_GEN_GO_GRPC" \
  --proto_path=. \
  --go_out="$OUTPUT_DIR/vllm" \
  --go_opt=paths=source_relative \
  $MAPPINGS \
  --go_opt=Mvllm_engine.proto=github.com/lightseek/smg/crates/grpc_client/go/generated/vllm \
  --go-grpc_out="$OUTPUT_DIR/vllm" \
  --go-grpc_opt=paths=source_relative \
  $MAPPINGS_GRPC \
  --go-grpc_opt=Mvllm_engine.proto=github.com/lightseek/smg/crates/grpc_client/go/generated/vllm \
  vllm_engine.proto

echo "Generation complete!"
