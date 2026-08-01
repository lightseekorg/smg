<p align="center">
  <img alt="SMG Logo" src="https://raw.githubusercontent.com/smg-project/smg/main/assets/images/logomark.svg" width="80">
</p>

<h1 align="center">Shepherd Model Gateway</h1>

<p align="center">
  <a href="https://github.com/smg-project/smg/releases/latest"><img src="https://img.shields.io/github/v/release/smg-project/smg?logo=github&label=Release" alt="Release"></a>
  <a href="https://github.com/orgs/lightseekorg/packages/container/package/smg"><img src="https://img.shields.io/badge/ghcr.io-lightseekorg%2Fsmg-blue?logo=docker" alt="Docker"></a>
  <a href="https://pypi.org/project/smg/"><img src="https://img.shields.io/pypi/v/smg?logo=pypi&logoColor=white&label=PyPI" alt="PyPI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://smg-project.github.io/smg"><img src="https://img.shields.io/badge/docs-latest-brightgreen.svg" alt="Docs"></a>
  <a href="https://discord.lightseek.org"><img src="https://img.shields.io/badge/Discord-Join%20Us-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://slack.lightseek.org"><img src="https://img.shields.io/badge/Slack-Join%20Us-4A154B?logo=slack&logoColor=white" alt="Slack"></a>
  <a href="https://deepwiki.com/smg-project/smg"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></a>
  <a href="https://pytorch.org/blog/lightseek-smg/"><img src="https://img.shields.io/badge/PyTorch-Technical%20Blog-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch Blog"></a>
</p>

Engine-agnostic, high-performance model-routing gateway for large-scale LLM deployments. Centralizes worker lifecycle management, balances traffic across HTTP/gRPC/OpenAI-compatible backends, and provides enterprise-ready control over history storage, MCP tooling, and privacy-sensitive workflows.

<p align="center">
  <img src="https://raw.githubusercontent.com/smg-project/smg/main/assets/images/architecture.svg" alt="SMG architecture: clients flow through the gateway layer and router layer to gRPC workers, HTTP workers, and external APIs" width="100%">
</p>

## Why SMG?

|                                 |                                                                                                                                                                  |
|:--------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **🚀 Maximize GPU Utilization** | Cache-aware routing understands your inference engine's KV cache state—whether vLLM, TensorRT-LLM, TokenSpeed, or SGLang—to reuse prefixes and reduce redundant computation. |
| **🔌 One API, Any Backend**     | Route to self-hosted models (vLLM, TensorRT-LLM, TokenSpeed, SGLang) or cloud providers (OpenAI, Anthropic, Gemini, Bedrock, and more) through a single unified endpoint. |
| **⚡ Built for Speed**           | Native Rust with gRPC pipelines, sub-millisecond routing decisions, and zero-copy tokenization. Circuit breakers and automatic failover keep things running.     |
| **🔒 Enterprise Control**       | Multi-tenant rate limiting with OIDC, WebAssembly plugins for custom logic, and a privacy boundary that keeps conversation history within your infrastructure.   |
| **📊 Full Observability**       | 40+ Prometheus metrics, OpenTelemetry tracing, and structured JSON logs with request correlation—know exactly what's happening at every layer.                   |

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
smg launch --worker-urls http://localhost:8000

# Multiple workers with cache-aware routing
smg launch --worker-urls http://gpu1:8000 http://gpu2:8000 --policy cache_aware

# With high availability mesh
smg launch --worker-urls http://gpu1:8000 --enable-mesh \
  --mesh-advertise-host 10.0.0.1 --mesh-peer-urls 10.0.0.2:39527
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
| TensorRT-LLM | Anthropic |
| TokenSpeed | Google Gemini |
| SGLang | OCI Generative AI Service |
| Ollama | Azure OpenAI |
| Any OpenAI-compatible server | Any OpenAI-compatible provider |

## Features

| Feature | Description |
|---------|-------------|
| **[8 Routing Policies](https://github.com/smg-project/smg-docs/blob/main/src/lib/content/concepts/routing/load-balancing.md)** | cache_aware, round_robin, power_of_two, consistent_hashing, prefix_hash, manual, random, bucket |
| **[gRPC Pipeline](https://github.com/smg-project/smg-docs/blob/main/src/lib/content/concepts/architecture/grpc-pipeline.md)** | Native gRPC with streaming, reasoning extraction, and tool call parsing |
| **[MCP Integration](https://github.com/smg-project/smg-docs/blob/main/src/lib/content/concepts/extensibility/mcp.md)** | Connect external tool servers via Model Context Protocol |
| **[High Availability](https://github.com/smg-project/smg-docs/blob/main/src/lib/content/concepts/architecture/high-availability.md)** | Mesh networking with SWIM protocol for multi-node deployments |
| **[Chat History](https://github.com/smg-project/smg-docs/blob/main/src/lib/content/concepts/data/chat-history.md)** | Pluggable storage: PostgreSQL, Oracle, Redis, or in-memory |
| **[WASM Plugins](https://github.com/smg-project/smg-docs/blob/main/src/lib/content/concepts/extensibility/wasm-plugins.md)** | Extend with custom WebAssembly logic |
| **[Resilience](https://github.com/smg-project/smg-docs/blob/main/src/lib/content/concepts/reliability/index.md)** | Circuit breakers, retries with backoff, rate limiting |

## Documentation

| | |
|:--|:--|
| [Getting Started](https://github.com/smg-project/smg-docs/blob/main/src/lib/content/getting-started/index.md) | Installation and first steps |
| [Architecture](https://github.com/smg-project/smg-docs/blob/main/src/lib/content/concepts/architecture/overview.md) | How SMG works |
| [Configuration](https://github.com/smg-project/smg-docs/blob/main/src/lib/content/reference/configuration.md) | CLI reference and options |
| [API Reference](https://github.com/smg-project/smg-docs/blob/main/src/lib/content/reference/api/openai.md) | OpenAI-compatible endpoints |
| [Kubernetes Setup](https://github.com/smg-project/smg-docs/blob/main/src/lib/content/getting-started/service-discovery.md) | In-cluster discovery and production setup |

## Contributing

We welcome contributions! See [Contributing Guide](https://github.com/smg-project/smg-docs/blob/main/src/lib/content/contributing/index.md) for details.

- [Development Setup](https://github.com/smg-project/smg-docs/blob/main/src/lib/content/contributing/development.md)
- [Code Style](https://github.com/smg-project/smg-docs/blob/main/src/lib/content/contributing/code-style.md)
