---
title: Architecture Overview
---

# Architecture Overview

SMG is a high-performance inference gateway that sits between your applications and LLM workers. It provides unified routing, enterprise features, and full observability across heterogeneous model deployments.

---

## System Architecture

<div class="architecture-diagram">
  <img src="../../../assets/images/architecture-detailed.svg" alt="SMG Architecture">
</div>

---

## Registries & State

Registries hold the configuration and state needed for request processing.

| Registry | Purpose | Used By |
|----------|---------|---------|
| **Model Registry** | Maps model names to backends and capabilities | Router Manager |
| **LB Policy Registry** | Load balancing configurations per model | All routing paths |
| **Tokenizer Registry** | Tokenizers for gateway-side processing | gRPC path |
| **Chat History** | Multi-turn conversation context | Responses API |
| **WASM Plugins** | Custom request/response transformations | Middleware |

---

## API Endpoints

SMG exposes three categories of endpoints:

### Inference APIs

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | OpenAI-compatible chat completions |
| `POST /v1/completions` | Text completions |
| `POST /v1/responses` | Agentic workflows with tool execution |
| `POST /v1/embeddings` | Embedding generation |
| `POST /v1/rerank` | Reranking API |
| `POST /messages` | Anthropic Messages API |

### Utility APIs

| Endpoint | Description |
|----------|-------------|
| `POST /tokenize` | Tokenize text using model's tokenizer |
| `POST /detokenize` | Convert token IDs back to text |
| `POST /v1/parser/tool` | Parse tool calls from text |
| `POST /v1/parser/reasoning` | Parse reasoning chains |

### Admin APIs

| Endpoint | Description |
|----------|-------------|
| `GET/POST /workers` | Worker management |
| `GET/POST /tokenizers` | Tokenizer management |
| `GET/POST /wasm` | WASM plugin management |
| `GET/POST /mcp` | MCP server management |

---

## Gateway Layer

The gateway layer handles cross-cutting concerns before requests reach the router.

### Middleware Pipeline

| Component | Function |
|-----------|----------|
| **Rate Limiter** | Multi-tenant token bucket with per-user quotas |
| **OIDC Auth** | JWT validation and tenant extraction |
| **WASM Plugins** | Custom request transformation logic |
| **Request ID** | Assigns unique ID for tracing |
| **Metrics** | Records latency, throughput, error rates |
| **OpenTelemetry** | Distributed tracing spans |

---

## Router Layer

The router layer handles LLM-specific request processing. It selects one of three routing paths based on worker type.

### Router Manager

| Worker Type | Path Selected | Gateway Behavior |
|-------------|---------------|------------------|
| gRPC workers | gRPC Path | Full server - tokenization, chat templates, tool parsing |
| HTTP workers | HTTP Path | Smart proxy - load balancing, PD disaggregation |
| External APIs | 3rd Party Path | Unified router - provider abstraction |

---

## gRPC Path (Token-Level Streaming)

The gRPC path provides maximum performance by handling all text processing at the gateway.

### Pipeline Stages

| Stage | Function |
|-------|----------|
| **Chat Template** | Apply model-specific chat template (Jinja2) |
| **Tokenization** | Convert text to token IDs using model tokenizer |
| **Token Cache** | Cache tokenized prefixes for reuse |
| **Load Balance** | Select worker using cache-aware policy |
| **Detokenize** | Convert streaming tokens back to text |
| **Reasoning Parser** | Extract thinking/reasoning from output (DeepSeek-R1, etc.) |
| **Tool Parser** | Parse function/tool calls from output |

### Supported Backends

- SGLang (gRPC)
- vLLM (gRPC)
- TensorRT-LLM (gRPC)

---

## HTTP Path (OpenAI Compatible)

The HTTP path supports two modes for OpenAI-compatible backends.

### Regular HTTP Mode

Standard load balancing across HTTP workers running full inference.

### PD (Prefill-Decode) Mode

Disaggregated inference with separate prefill and decode workers:

1. **Find P/D Pair** - Select a prefill worker and decode worker pair
2. **Mutate Headers** - Add routing headers for KV cache transfer
3. **Prefill Worker** - Processes prompt, transfers KV cache
4. **Decode Worker** - Generates tokens using transferred KV cache

### Supported Backends

- SGLang (HTTP)
- vLLM (HTTP)
- TensorRT-LLM (HTTP)

---

## 3rd Party Path

The 3rd party path routes to external LLM providers through a unified interface.

### Model Discovery

The gateway discovers available models from external providers and exposes them through `/v1/models`.

### Supported Providers

| Provider | API Style |
|----------|-----------|
| OpenAI | OpenAI |
| Anthropic | Messages |
| Google Gemini | Gemini |
| xAI Grok | OpenAI |
| Together AI | OpenAI |
| OpenRouter | OpenAI |
| AWS Bedrock | Bedrock |
| OCI Generative AI | OCI |

---

## Response Processing

All paths converge at response processing for tool handling and MCP execution.

### Components

| Component | Function |
|-----------|----------|
| **Tool Parser** | Extracts function/tool calls from model output |
| **MCP Handler** | Executes tools via Model Context Protocol servers |
| **Response Builder** | Assembles final response with tool results |

### MCP Loop

When the model requests tool execution:

1. Tool parser extracts the tool call
2. MCP handler executes the tool
3. Result is re-routed through the router for continued generation
4. Loop continues until model produces final response

---

## Load Balancing

All paths use the same load balancing infrastructure with multiple policies.

| Policy | Algorithm | Best For |
|--------|-----------|----------|
| `random` | Uniform random | Simple deployments |
| `round_robin` | Sequential cycling | Even distribution |
| `power_of_two` | Sample two, pick lighter | Balanced load |
| `cache_aware` | Prefix locality + load | Production (default) |

### Cache-Aware Routing

The cache-aware policy optimizes for KV cache reuse:

1. Hash the tokenized prefix
2. Find workers with cached prefix
3. Among matches, select by current load
4. Falls back to least-loaded if no cache hit

This integrates with vLLM, SGLang, and TensorRT-LLM's native KV cache management.

---

## Resilience

Built-in resilience features protect against failures.

| Feature | Function |
|---------|----------|
| **Circuit Breaker** | Stops routing to failing workers |
| **Retry Handler** | Retries failed requests with exponential backoff |
| **Health Checker** | Periodic worker health probes |
| **Timeout Manager** | Request and connection timeouts |

---

## What's Next?

- [Control Plane](control-plane.md) - Worker management and service discovery
- [Data Plane](data-plane.md) - Request routing implementation details
- [Load Balancing](../routing/load-balancing.md) - Routing policy deep dive
- [Cache-Aware Routing](../routing/cache-aware.md) - KV cache optimization
