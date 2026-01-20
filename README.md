# Shepherd Model Gateway (SMG)

High-performance model-routing gateway for large-scale LLM deployments. Centralizes worker lifecycle management, balances traffic across HTTP/gRPC/OpenAI-compatible backends, and provides enterprise-ready control over history storage, MCP tooling, and privacy-sensitive workflows.

<p align="center">
  <img src="docs/assets/images/architecture-animated.svg" alt="SMG Architecture" width="100%">
</p>

## Key Features

### Performance & Routing
- **Cache-Aware Load Balancing** - Native integration with vLLM, SGLang, and TensorRT-LLM KV cache schedulers for optimal prefix reuse
- **High-Performance gRPC Pipeline** - Native Rust tokenization, chat templates, reasoning parsers, and tool-call execution
- **Sub-millisecond Routing** - Intelligent request distribution with circuit breakers and automatic failover

### Universal Backend Support
- **Self-Hosted Inference** - vLLM, SGLang, TensorRT-LLM via HTTP or gRPC
- **Third-Party Providers** - OpenAI, Anthropic, Google Gemini, xAI Grok, Together AI, OpenRouter, AWS Bedrock, OCI Generative AI, and more
- **Unified API** - One endpoint for all backends with automatic protocol translation

### Complete API Coverage
- **Full OpenAI Compatibility** - Chat, Completions, Embeddings, and Responses API for agentic workflows
- **Anthropic Messages API** - Native support for Claude models
- **MCP Tool Execution** - Model Context Protocol for function calling and tool use

### Enterprise Ready
- **WebAssembly Plugins** - Write custom request/response transformations in any language that compiles to WASM
- **Multi-Tenant Rate Limiting** - Per-tenant quotas with OIDC authentication
- **Privacy Boundary** - Conversation history and MCP sessions stay within the gateway

### Observability
- **40+ Prometheus Metrics** - Request latency, token throughput, cache hit rates, circuit breaker states
- **OpenTelemetry Tracing** - Distributed tracing across the entire request lifecycle
- **Structured Logging** - JSON logs with request correlation IDs

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
