---
title: External Providers
---

# External Providers

SMG can route requests to external LLM provider APIs (OpenAI, Anthropic, xAI, Google Gemini), acting as a unified gateway. This enables provider-agnostic applications, load balancing across providers, and centralized observability.

<div class="prerequisites" markdown>

#### Before you begin

- Completed the [Getting Started](index.md) guide
- An API key for at least one external provider

</div>

---

## Supported Providers

SMG auto-detects the provider from the model name in each request and applies the correct API transformations:

| Provider | Auto-Detection | Header Format |
|----------|---------------|---------------|
| OpenAI | `gpt-*`, `o1-*`, `o3-*` models | `Authorization: Bearer` |
| Anthropic | `claude-*` models | `x-api-key` (plus `anthropic-version`) |
| xAI | `grok-*` models | `Authorization: Bearer` |
| Google Gemini | `gemini-*` models | `x-goog-api-key` |

---

## Quick Start

Register an external worker via IGW mode:

```bash
# Start SMG in IGW mode
smg --enable-igw
```

```bash
# Register an OpenAI worker
curl -X POST http://localhost:30000/workers \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://api.openai.com/v1",
    "api_key": "sk-...",
    "runtime_type": "external",
    "provider": "openai"
  }'
```

Send a request through the gateway:

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## Model Discovery

SMG supports fan-out model discovery across all registered external workers. When a caller sends a `GET /v1/models` request with a bearer token, SMG:

1. Fans out the request to all healthy external workers concurrently
2. Forwards the caller's token to each upstream provider
3. Returns the first non-empty model inventory from the fanned-out upstream responses

```bash
curl http://localhost:30000/v1/models \
  -H "Authorization: Bearer sk-..."
```

This supports BYOK (bring your own key) — the caller's token is forwarded to the upstream provider, so each caller can discover models available under their own account.

---

## API Key Handling

SMG supports two methods for providing API keys to external providers:

- **Stored key** — Set at worker registration time via the `api_key` field
- **Caller key (BYOK)** — Passed by the caller via the `Authorization` header at request time

If both a stored key and a caller key are present, the caller's key takes precedence.

!!! note "Provider-aware routing"
    SMG routes each request to the correct provider based on the model name. API keys are only forwarded to the matching provider, preventing keys from leaking across providers.

---

## Multiple Providers

Register workers for multiple providers to route across them by model name:

```bash
# Register OpenAI
curl -X POST http://localhost:30000/workers \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://api.openai.com/v1",
    "provider": "openai",
    "api_key": "sk-..."
  }'

# Register Anthropic
curl -X POST http://localhost:30000/workers \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://api.anthropic.com/v1",
    "provider": "anthropic",
    "api_key": "sk-ant-..."
  }'
```

SMG picks the right provider based on the model name in each request:

```bash
# Routes to OpenAI
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello"}]
  }'

# Routes to Anthropic
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

---

## Next Steps

- [Multiple Workers](multiple-workers.md) — Load balancing, worker types, and IGW configuration options
- [Load Balancing](load-balancing.md) — Routing policies for distributing traffic across workers
- [Monitoring](monitoring.md) — Track request rates, latency, and worker health across providers
