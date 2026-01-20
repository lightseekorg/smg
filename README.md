# Shepherd Model Gateway (SMG)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://lightseekorg.github.io/smg)

High-performance model-routing gateway for large-scale LLM deployments. Centralizes worker lifecycle management, balances traffic across HTTP/gRPC/OpenAI-compatible backends, and provides enterprise-ready control over history storage, MCP tooling, and privacy-sensitive workflows.

<p align="center">
  <img src="docs/assets/images/architecture-animated.svg" alt="SMG Architecture" width="100%">
</p>

## Why SMG?

| | |
|:--|:--|
| **🚀 Maximize GPU Utilization** | Cache-aware routing understands your inference engine's KV cache state—whether vLLM, SGLang, or TensorRT-LLM—to reuse prefixes and reduce redundant computation. |
| **🔌 One API, Any Backend** | Route to self-hosted models (vLLM, SGLang, TensorRT-LLM) or cloud providers (OpenAI, Anthropic, Gemini, Bedrock, and more) through a single unified endpoint. |
| **⚡ Built for Speed** | Native Rust with gRPC pipelines, sub-millisecond routing decisions, and zero-copy tokenization. Circuit breakers and automatic failover keep things running. |
| **🔒 Enterprise Control** | Multi-tenant rate limiting with OIDC, WebAssembly plugins for custom logic, and a privacy boundary that keeps conversation history within your infrastructure. |
| **📊 Full Observability** | 40+ Prometheus metrics, OpenTelemetry tracing, and structured JSON logs with request correlation—know exactly what's happening at every layer. |

**API Coverage:** OpenAI Chat/Completions/Embeddings, Responses API for agents, Anthropic Messages, and MCP tool execution.

## Quick Start

**Install** — pick your preferred method:

```bash
# Docker
docker pull lightseekorg/smg:latest

# Python
pip install smg

# Rust
cargo install smg
```

**Run** — point SMG at your inference workers:

```bash
# Single worker
smg --worker-urls http://localhost:8000

# Multiple workers with cache-aware routing
smg --worker-urls http://gpu1:8000 http://gpu2:8000 --policy cache_aware

# With high availability mesh
smg --worker-urls http://gpu1:8000 --ha-mesh --seeds 10.0.0.2:30001,10.0.0.3:30001
```

**Use** — send requests to the gateway:

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3", "messages": [{"role": "user", "content": "Hello!"}]}'
```

That's it. SMG is now load-balancing requests across your workers.

## Supported Backends

| Self-Hosted | Cloud Providers |
|-------------|-----------------|
| vLLM | OpenAI |
| SGLang | Anthropic |
| TensorRT-LLM | Google Gemini |
| Ollama | AWS Bedrock |
| Any OpenAI-compatible server | Azure OpenAI |

## Features

| Feature | Description |
|---------|-------------|
| **[8 Routing Policies](docs/configuration/routing.md)** | cache_aware, round_robin, power_of_two, consistent_hashing, prefix_hash, manual, random, bucket |
| **[gRPC Pipeline](docs/configuration/grpc-pipeline.md)** | Native gRPC with streaming, reasoning extraction, and tool call parsing |
| **[MCP Integration](docs/configuration/mcp.md)** | Connect external tool servers via Model Context Protocol |
| **[High Availability](docs/configuration/high-availability.md)** | Mesh networking with SWIM protocol for multi-node deployments |
| **[Chat History](docs/configuration/storage.md)** | Pluggable storage: PostgreSQL, Oracle, Redis, or in-memory |
| **[WASM Plugins](docs/configuration/wasm-plugins.md)** | Extend with custom WebAssembly logic |
| **[Resilience](docs/configuration/resilience.md)** | Circuit breakers, retries with backoff, rate limiting |

## Documentation

| | |
|:--|:--|
| [Getting Started](docs/getting-started/quickstart.md) | Installation and first steps |
| [Architecture](docs/concepts/architecture/overview.md) | How SMG works |
| [Configuration](docs/reference/configuration.md) | CLI reference and options |
| [API Reference](docs/reference/api/openai.md) | OpenAI-compatible endpoints |
| [Deployment](docs/tasks/deployment/kubernetes.md) | Production setup |

## Contributing

We welcome contributions! See [Contributing Guide](docs/contributing/index.md) for details.

- [Development Setup](docs/contributing/development.md)
- [Code Style](docs/contributing/code-style.md)

