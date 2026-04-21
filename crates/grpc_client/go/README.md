# SMG gRPC Client (Go)

This module contains auto-generated Go gRPC code for Shepherd Model Gateway services (SGLang, vLLM, TRT-LLM).

## Installation

To use this module in your Go project:

```bash
go get github.com/lightseekorg/smg/crates/grpc_client/go
```

## Package Structure

The generated code is divided into packages under the `generated` directory:

-   `generated/common`: Common types (e.g., KV Events)
-   `generated/sglang_encoder`: SGLang Encoder service
-   `generated/sglang_scheduler`: SGLang Scheduler service
-   `generated/trtllm`: TensorRT-LLM Service
-   `generated/vllm`: vLLM Engine service

## Generation

To regenerate the code from `.proto` files:

1.  Install prerequisites:
    ```bash
    go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
    go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
    ```
2.  Run the generation script:
    ```bash
    bash generate.sh
    ```

The generated code is committed to **Git release tags** automatically by the CI pipeline for convenience.

> [!IMPORTANT]
> When consuming this module in external projects, you **must** specify a version tag (e.g., `@v1.0.0`). Consuming the `main` branch directly will result in missing generated files due to full-tree `.gitignore`.


## Release

Releases are created automatically when the version in the `VERSION` file is updated and merged into `main`. It will create a Git tag: `crates/grpc_client/go/v<VERSION>`.
