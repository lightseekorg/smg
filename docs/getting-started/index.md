---
title: Getting Started
---

# Getting Started

Shepherd Model Gateway (SMG) routes and manages LLM traffic across workers. This page gives you a fast path to a working gateway, then points you to feature-specific setup guides.

## Install

=== "pip (recommended)"

    Pre-built wheels are available for Linux (x86_64, aarch64, musllinux), macOS (Apple Silicon), and Windows (x86_64), with Python 3.9–3.14.

    ```bash
    pip install smg
    ```

    This installs both:

    - `smg serve` (Python orchestration command for workers + gateway)
    - `smg launch` (router launch path in Rust CLI)

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

## Step 1: Start SMG

Choose one of these startup paths.

### Option A: All-in-one with `smg serve`

`smg serve` launches backend worker process(es) and then starts SMG with generated worker URLs.

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

=== "TensorRT-LLM (gRPC)"

    ```bash
    smg serve \
      --backend trtllm \
      --model meta-llama/Llama-3.1-8B-Instruct \
      --data-parallel-size 2 \
      --host 0.0.0.0 \
      --port 30000
    ```

This starts `--data-parallel-size` worker replicas, waits for readiness, then starts the gateway.

| Option | Default | Description |
|--------|---------|-------------|
| `--backend` | `sglang` | Inference backend: `sglang`, `vllm`, or `trtllm` |
| `--connection-mode` | `grpc` | Worker connection mode: `grpc` or `http` (vLLM and TensorRT-LLM only support gRPC) |
| `--data-parallel-size` | `1` | Number of worker replicas (one per GPU) |
| `--worker-base-port` | `31000` | Base port for worker processes |
| `--host` | `127.0.0.1` | Router host |
| `--port` | `8080` | Router port |

### Option B: Launch gateway only with `smg launch`

Use this when workers are already running or managed by another platform.

For gRPC workers:

```bash
smg launch \
  --worker-urls grpc://localhost:50051 \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --policy round_robin \
  --host 0.0.0.0 \
  --port 30000
```

For HTTP workers:

```bash
smg launch \
  --worker-urls http://localhost:8000 \
  --policy round_robin \
  --host 0.0.0.0 \
  --port 30000
```

## Step 2: Verify Core Endpoints

Health:

```bash
curl http://localhost:30000/health
curl http://localhost:30000/readiness
```

OpenAI-compatible chat completions:

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}]
  }'
```

Responses API:

```bash
curl http://localhost:30000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "input": "Say hello in one sentence."
  }'
```

## Step 3: Choose Your Setup Track

### Core Deployment

- [Multiple Workers](multiple-workers.md)
- [gRPC Workers](grpc-workers.md)
- [PD Disaggregation](pd-disaggregation.md)
- [Service Discovery](service-discovery.md)

### Operations and Security

- [Monitoring](monitoring.md)
- [Logging](logging.md)
- [TLS](tls.md)
- [Control Plane Auth](control-plane-auth.md)
- [Control Plane Operations](control-plane-operations.md)

### Reliability and Data

- [Reliability Controls](reliability-controls.md)
- [Data Connections](data-connections.md)
- [Tokenization and Parsing APIs](tokenization-and-parsing.md)

### Advanced Features

- [Load Balancing](load-balancing.md)
- [Tokenizer Caching](tokenizer-caching.md)
- [MCP in Responses API](mcp.md)

---

## Worker Startup Recipes (Standalone)

Use these when workers are not started via `smg serve`.

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
    smg launch \
      --pd-disaggregation \
      --prefill grpc://localhost:50051 8998 \
      --decode grpc://localhost:50052 \
      --model-path meta-llama/Llama-3.1-8B-Instruct \
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
    smg launch \
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

## Deploy with Docker

For local deployment, run SMG in a container and point it at your worker:

```bash
docker pull lightseekorg/smg:latest

docker run -d \
  --name smg \
  -p 30000:30000 \
  -p 29000:29000 \
  lightseekorg/smg:latest \
  --worker-urls http://host.docker.internal:8000 \
  --policy cache_aware \
  --prometheus-port 29000
```

Verify:

```bash
docker ps | grep smg
curl http://localhost:30000/health
```

## Deploy to Kubernetes (Quick Start)

Run SMG in-cluster and use service discovery to pick up worker pods automatically.

Start SMG with service discovery:

```bash
smg \
  --service-discovery \
  --selector app=sglang-worker \
  --service-discovery-namespace inference \
  --service-discovery-port 8000 \
  --policy cache_aware
```

Required RBAC permissions:

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: smg-discovery
  namespace: inference
rules:
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["get", "list", "watch"]
```

Verify:

```bash
kubectl get pods -n inference -l app=sglang-worker
curl http://localhost:30000/workers
```

## Navigate by Category

### Core Setup

- [Multiple Workers](multiple-workers.md) — connect local or external worker endpoints
- [gRPC Workers](grpc-workers.md) — gateway-side tokenization, parsing, and tool handling
- [PD Disaggregation](pd-disaggregation.md) — split prefill and decode paths
- [Service Discovery](service-discovery.md) — Kubernetes pod-based worker registration

### Operations

- [Monitoring](monitoring.md) — Prometheus metrics, tracing, and alerts
- [Logging](logging.md) — structured logs and aggregation patterns
- [TLS](tls.md) — HTTPS gateway configuration
- [Control Plane Auth](control-plane-auth.md) — secure worker/tokenizer/WASM management endpoints

### Reliability and Data

- [Reliability Controls](reliability-controls.md) — concurrency limits, retries, and circuit breakers
- [Data Connections](data-connections.md) — history backend setup for Postgres, Redis, and Oracle
- [Tokenization and Parsing APIs](tokenization-and-parsing.md) — tokenize, detokenize, and parser endpoints

### Advanced Features

- [Load Balancing](load-balancing.md) — policy selection and tuning
- [Tokenizer Caching](tokenizer-caching.md) — L0/L1 cache setup for gRPC mode
- [MCP in Responses API](mcp.md) — configure and execute MCP tools through `/v1/responses`

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
