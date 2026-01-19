# Shepherd Model Gateway (SMG)

High-performance model-routing gateway for large-scale LLM deployments. Centralizes worker lifecycle management, balances traffic across HTTP/gRPC/OpenAI-compatible backends, and provides enterprise-ready control over history storage, MCP tooling, and privacy-sensitive workflows.

<p align="center">
  <img src="docs/assets/images/architecture-animated.svg" alt="SMG Architecture" width="100%">
</p>

## Key Features

- **Unified Control Plane** - Register, monitor, and orchestrate workers across heterogeneous model fleets
- **Multi-Protocol Data Plane** - Route traffic across HTTP, gRPC, and OpenAI-compatible backends
- **High-Performance gRPC Pipeline** - Native Rust tokenization, reasoning parsers, and tool-call execution
- **Enterprise Privacy** - Conversation history and MCP sessions operate within the router boundary
- **Comprehensive Observability** - 40+ Prometheus metrics, OpenTelemetry tracing, structured logging

## Quick Start

### Docker

```bash
docker pull lightseekorg/smg:latest
```

### Build from Source

```bash
cargo build --release
```

### Run

```bash
# HTTP workers
./target/release/smg \
  --worker-urls http://worker1:8000 http://worker2:8000 \
  --policy cache_aware

# gRPC workers (highest performance)
./target/release/smg \
  --worker-urls grpc://127.0.0.1:20000 \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --reasoning-parser deepseek-r1
```

### Test

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Documentation

Full documentation available at [lightseekorg.github.io/smg](https://lightseekorg.github.io/smg):

- [Installation](docs/getting-started/installation.md)
- [Quick Start](docs/getting-started/quickstart.md)
- [Architecture](docs/concepts/architecture/overview.md)
- [API Reference](docs/reference/api/openai.md)
- [Deployment Guide](docs/tasks/deployment/kubernetes.md)

## Load Balancing Policies

| Policy | Description |
|--------|-------------|
| `random` | Uniform random selection |
| `round_robin` | Cycles through workers |
| `power_of_two` | Samples two, picks lighter |
| `cache_aware` | Cache locality + load balancing (default) |

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | OpenAI-compatible chat |
| `POST /v1/completions` | Text completions |
| `POST /v1/embeddings` | Embedding generation |
| `POST /v1/responses` | Agentic response flows |
| `GET /v1/models` | List available models |

## Python Bindings

```bash
pip install maturin
cd bindings/python
maturin develop
```

```bash
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 \
  --policy cache_aware
```

## License

Apache 2.0
