---
title: Multiple Workers
---

# Multiple Workers

SMG can route across many workers simultaneously — local inference servers, remote cloud APIs, or a mix of both. This guide covers how to add workers and balance traffic across them.

<div class="prerequisites" markdown>

#### Before you begin

- Completed the [Getting Started](index.md) guide

</div>

## Supported Worker Types

SMG connects to workers over HTTP or gRPC, and supports both local inference servers and remote API providers:

| Worker Type | Protocol | Example URL |
|-------------|----------|-------------|
| SGLang | HTTP / gRPC | `http://worker:8000` or `grpc://worker:50051` |
| vLLM | gRPC | `grpc://worker:50051` |
| TensorRT-LLM | gRPC | `grpc://worker:50051` |
| OpenAI (GPT) | HTTP | `https://api.openai.com` |
| Anthropic (Claude) | HTTP | `https://api.anthropic.com` |
| xAI (Grok) | HTTP | `https://api.x.ai` |
| Google (Gemini) | HTTP | `https://generativelanguage.googleapis.com` |
| Any OpenAI-compatible API | HTTP | `https://your-provider.com` |

## Static Workers via CLI

Pass multiple URLs to `--worker-urls`:

```bash
smg \
  --worker-urls http://worker1:8000 http://worker2:8000 http://worker3:8000 \
  --policy round_robin \
  --host 0.0.0.0 \
  --port 30000
```

For gRPC workers, use the `grpc://` scheme and provide `--model-path` so the gateway can load the tokenizer:

```bash
smg \
  --worker-urls grpc://worker1:50051 grpc://worker2:50052 \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --policy round_robin
```

See [gRPC Workers](grpc-workers.md) for details on what gRPC mode enables.

## Cloud API Workers

Route to cloud providers by setting `--backend openai` and passing the provider URL. API keys are read from environment variables. SMG auto-detects the provider (OpenAI, Anthropic, xAI, Gemini) from the model name and applies the correct API transformations.

=== "OpenAI"

    ```bash
    export OPENAI_API_KEY=sk-...

    smg \
      --backend openai \
      --worker-urls https://api.openai.com \
      --host 0.0.0.0 \
      --port 30000
    ```

=== "Anthropic"

    ```bash
    export ANTHROPIC_API_KEY=sk-ant-...

    smg \
      --backend openai \
      --worker-urls https://api.anthropic.com \
      --host 0.0.0.0 \
      --port 30000
    ```

=== "xAI (Grok)"

    ```bash
    export XAI_API_KEY=xai-...

    smg \
      --backend openai \
      --worker-urls https://api.x.ai \
      --host 0.0.0.0 \
      --port 30000
    ```

=== "Gemini"

    ```bash
    export GEMINI_API_KEY=...

    smg \
      --backend openai \
      --worker-urls https://generativelanguage.googleapis.com \
      --host 0.0.0.0 \
      --port 30000
    ```

## Dynamic Workers with IGW Mode

In Inference Gateway (IGW) mode, SMG starts with no workers and you add or remove them at runtime via the REST API:

```bash
smg --enable-igw --host 0.0.0.0 --port 30000
```

### Add a worker

```bash
curl -X POST http://localhost:30000/workers \
  -H "Content-Type: application/json" \
  -d '{"url": "http://worker1:8000"}'
```

Response:

```json
{
  "status": "accepted",
  "worker_id": "a1b2c3d4",
  "url": "http://worker1:8000",
  "location": "/workers/a1b2c3d4",
  "message": "Worker addition queued for background processing"
}
```

### List workers

```bash
curl http://localhost:30000/workers
```

### Remove a worker

```bash
curl -X DELETE http://localhost:30000/workers/{worker_id}
```

### Worker configuration options

The `POST /workers` endpoint accepts additional fields:

```json
{
  "url": "http://worker:8000",
  "api_key": "optional-key",
  "model_id": "llama-70b",
  "runtime": "sglang",
  "worker_type": "regular",
  "priority": 50,
  "cost": 1.0,
  "disable_health_check": false,
  "labels": {"region": "us-east"}
}
```

| Field | Default | Description |
|-------|---------|-------------|
| `url` | (required) | Worker URL (`http://`, `grpc://`, or `https://` for cloud) |
| `api_key` | — | API key for authenticated workers |
| `model_id` | auto-discovered | Model identifier |
| `runtime` | `sglang` | Runtime: `sglang`, `vllm`, `trtllm`, or `external` |
| `worker_type` | `regular` | Type: `regular`, `prefill`, or `decode` |
| `priority` | `50` | Routing priority (0–100, higher = preferred) |
| `cost` | `1.0` | Cost multiplier for cost-aware routing |
| `disable_health_check` | `false` | Skip health checking for this worker |
| `labels` | `{}` | Arbitrary metadata |

## Verify

```bash
# List connected workers
curl http://localhost:30000/workers

# Check health
curl http://localhost:30000/health

# Send a request
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Next Steps

- [Monitoring](monitoring.md) — Track request rates, latency, and worker health
- [gRPC Workers](grpc-workers.md) — Enable tokenization, chat templates, and tool parsing at the gateway
- [PD Disaggregation](pd-disaggregation.md) — Separate prefill and decode onto specialized workers
