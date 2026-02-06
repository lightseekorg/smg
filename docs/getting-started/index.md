---
title: Getting Started
---

# Getting Started

Shepherd Model Gateway (SMG) routes requests across LLM inference workers with load balancing, failover, and observability. This page gets you from zero to a working gateway.

## Install

=== "pip (recommended)"

    Pre-built wheels are available for Linux (x86_64, aarch64, musllinux), macOS (Apple Silicon), and Windows (x86_64), with Python 3.9–3.14.

    ```bash
    pip install smg
    ```

    This also installs the `smg serve` command for launching workers and the gateway together.

=== "Cargo (crates.io)"

    ```bash
    cargo install smg
    ```

=== "Docker"

    Multi-architecture images are available for x86_64 and ARM64.

    ```bash
    docker pull lightseekorg/smg:latest
    ```

    Verify:

    ```bash
    docker run --rm lightseekorg/smg:latest --version
    ```

    Available tags: `latest` (stable), `v0.3.x` (specific version), `main` (development).

=== "From Source"

    ```bash
    # Install Rust
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    source "$HOME/.cargo/env"

    # Clone and build
    git clone https://github.com/lightseekorg/smg.git
    cd smg
    cargo build --release
    ```

    The binary is available at `./target/release/smg`.

## Option 1: All-in-One with `smg serve`

The `smg serve` command launches inference workers and the gateway router together. It supports data parallelism for running multiple worker replicas.

=== "SGLang"

    ```bash
    smg serve \
      --backend sglang \
      --model-path meta-llama/Llama-3.1-8B-Instruct \
      --data-parallel-size 2 \
      --connection-mode grpc \
      --host 0.0.0.0 \
      --port 30000
    ```

=== "vLLM"

    ```bash
    smg serve \
      --backend vllm \
      --model meta-llama/Llama-3.1-8B-Instruct \
      --data-parallel-size 2 \
      --host 0.0.0.0 \
      --port 30000
    ```

=== "TensorRT-LLM"

    ```bash
    smg serve \
      --backend trtllm \
      --model meta-llama/Llama-3.1-8B-Instruct \
      --data-parallel-size 2 \
      --host 0.0.0.0 \
      --port 30000
    ```

This starts `--data-parallel-size` worker replicas on separate GPUs, waits for them to become healthy, and then starts the gateway router.

| Option | Default | Description |
|--------|---------|-------------|
| `--backend` | `sglang` | Inference backend: `sglang`, `vllm`, or `trtllm` |
| `--connection-mode` | `grpc` | Worker connection mode: `grpc` or `http` (vLLM and TensorRT-LLM only support gRPC) |
| `--data-parallel-size` | `1` | Number of worker replicas (one per GPU) |
| `--worker-base-port` | `31000` | Base port for worker processes |
| `--host` | `127.0.0.1` | Router host |
| `--port` | `8080` | Router port |

## Option 2: Start SMG and Workers Separately

For more control, start the gateway and workers independently.

### Start Workers

=== "SGLang (gRPC)"

    ```bash
    python -m sglang.launch_server \
      --model-path meta-llama/Llama-3.1-8B-Instruct \
      --host 0.0.0.0 \
      --port 50051 \
      --grpc-mode
    ```

=== "SGLang (HTTP)"

    ```bash
    python -m sglang.launch_server \
      --model-path meta-llama/Llama-3.1-8B-Instruct \
      --host 0.0.0.0 \
      --port 8000
    ```

=== "vLLM (gRPC)"

    ```bash
    python -m vllm.entrypoints.grpc_server \
      --model meta-llama/Llama-3.1-8B-Instruct \
      --host 0.0.0.0 \
      --port 50051 \
      --tensor-parallel-size 1
    ```

=== "TensorRT-LLM (gRPC)"

    ```bash
    python -m tensorrt_llm.commands.serve serve \
      meta-llama/Llama-3.1-8B-Instruct \
      --grpc \
      --host 0.0.0.0 \
      --port 50051 \
      --backend pytorch \
      --tp_size 1
    ```

Wait until the worker is ready before starting SMG.

### Start SMG

For gRPC workers, use the `grpc://` URL scheme and provide `--model-path` so the gateway can load the tokenizer:

=== "gRPC Workers"

    ```bash
    smg \
      --worker-urls grpc://localhost:50051 \
      --model-path meta-llama/Llama-3.1-8B-Instruct \
      --policy round_robin \
      --host 0.0.0.0 \
      --port 30000
    ```

=== "HTTP Workers"

    ```bash
    smg \
      --worker-urls http://localhost:8000 \
      --policy round_robin \
      --host 0.0.0.0 \
      --port 30000
    ```

### PD Disaggregation Workers

For prefill-decode disaggregation, start separate prefill and decode workers:

=== "SGLang PD (gRPC)"

    ```bash
    # Prefill worker
    python -m sglang.launch_server \
      --model-path meta-llama/Llama-3.1-8B-Instruct \
      --host 0.0.0.0 \
      --port 50051 \
      --grpc-mode \
      --disaggregation-mode prefill \
      --disaggregation-bootstrap-port 8998

    # Decode worker
    python -m sglang.launch_server \
      --model-path meta-llama/Llama-3.1-8B-Instruct \
      --host 0.0.0.0 \
      --port 50052 \
      --grpc-mode \
      --disaggregation-mode decode \
      --disaggregation-bootstrap-port 8999
    ```

    Start SMG with bootstrap ports for SGLang coordination:

    ```bash
    smg \
      --pd-disaggregation \
      --prefill grpc://localhost:50051 8998 \
      --decode grpc://localhost:50052 \
      --host 0.0.0.0 \
      --port 30000
    ```

=== "SGLang PD (HTTP)"

    ```bash
    # Prefill worker
    python -m sglang.launch_server \
      --model-path meta-llama/Llama-3.1-8B-Instruct \
      --host 0.0.0.0 \
      --port 8000 \
      --disaggregation-mode prefill \
      --disaggregation-bootstrap-port 8998

    # Decode worker
    python -m sglang.launch_server \
      --model-path meta-llama/Llama-3.1-8B-Instruct \
      --host 0.0.0.0 \
      --port 8001 \
      --disaggregation-mode decode \
      --disaggregation-bootstrap-port 8999
    ```

    Start SMG with bootstrap ports for SGLang coordination:

    ```bash
    smg \
      --pd-disaggregation \
      --prefill http://localhost:8000 8998 \
      --decode http://localhost:8001 \
      --host 0.0.0.0 \
      --port 30000
    ```

=== "vLLM PD (gRPC + NIXL)"

    vLLM uses NIXL for KV cache transfer between prefill and decode workers:

    ```bash
    # Prefill worker
    VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
    python -m vllm.entrypoints.grpc_server \
      --model meta-llama/Llama-3.1-8B-Instruct \
      --host 0.0.0.0 \
      --port 50051 \
      --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'

    # Decode worker
    VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
    python -m vllm.entrypoints.grpc_server \
      --model meta-llama/Llama-3.1-8B-Instruct \
      --host 0.0.0.0 \
      --port 50052 \
      --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'
    ```

    Start SMG (no bootstrap ports needed — NIXL handles KV transfer):

    ```bash
    smg \
      --pd-disaggregation \
      --prefill grpc://localhost:50051 \
      --decode grpc://localhost:50052 \
      --model-path meta-llama/Llama-3.1-8B-Instruct \
      --host 0.0.0.0 \
      --port 30000
    ```

See [PD Disaggregation](pd-disaggregation.md) for full details including Mooncake backend and scaling.

## Send a Request

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 50
  }'
```

Expected response:

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 14,
    "completion_tokens": 8,
    "total_tokens": 22
  }
}
```

## Verify Health

```bash
# Gateway health
curl http://localhost:30000/health

# Worker status
curl http://localhost:30000/workers
```

## Next Steps

<div class="grid" markdown>

<div class="card" markdown>

### Multiple Workers

Route across many workers with different backends and cloud APIs.

[Add more workers →](multiple-workers.md)

</div>

<div class="card" markdown>

### Monitoring

Enable Prometheus metrics and track request rates, latency, and worker health.

[Set up monitoring →](monitoring.md)

</div>

<div class="card" markdown>

### gRPC Workers

Run SMG as a full OpenAI server with tokenization, chat templates, and tool calling at the gateway.

[Enable gRPC mode →](grpc-workers.md)

</div>

<div class="card" markdown>

### PD Disaggregation

Separate prefill and decode phases onto specialized workers for optimal latency.

[Set up PD →](pd-disaggregation.md)

</div>

</div>

## Troubleshooting

??? question "Gateway starts but can't connect to worker"

    **Symptoms:** Gateway logs show connection errors.

    **Solutions:**

    1. Verify the worker is running: `curl http://localhost:8000/health`
    2. Check network connectivity between gateway and worker
    3. If using Docker, ensure proper network configuration (`--network host` or Docker network)

??? question "Request times out"

    **Symptoms:** Requests hang or return 504 errors.

    **Solutions:**

    1. Check worker health: `curl http://localhost:30000/workers`
    2. Increase timeout: `--request-timeout-secs 120`
    3. Check worker logs for errors

??? question "Model not found error"

    **Symptoms:** `model not found` in response.

    **Solutions:**

    1. The `model` field in requests should match the model loaded on the worker
    2. Check available models: `curl http://localhost:30000/v1/models`
