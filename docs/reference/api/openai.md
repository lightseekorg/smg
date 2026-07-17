---
title: OpenAI-Compatible API
---

# OpenAI-Compatible API Reference

SMG provides a fully OpenAI-compatible API, allowing you to use existing OpenAI client libraries with your self-hosted inference workers.

---

## Base URL

```
http://localhost:30000/v1
```

---

## Authentication

SMG supports optional API key authentication:

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '...'
```

Enable authentication with `--api-key`:

```bash
smg --worker-urls http://worker:8000 --api-key "your-api-key"
```

For real multi-tenant separation (e.g. per-tenant rate limiting), configure one key per
tenant instead with `--tenant-api-key tenant_id:key` (repeatable). Each key resolves to
its own tenant identity; `--api-key` remains available as a single shared fallback key
whose callers all share one identity.

```bash
smg --worker-urls http://worker:8000 \
  --tenant-api-key team-red:red-secret \
  --tenant-api-key team-blue:blue-secret
```

Callers authenticate the same way as with `--api-key` — a `Bearer` token in the
`Authorization` header — just using their own tenant's key instead of the shared one:

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Authorization: Bearer red-secret" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

This request is attributed to `team-red`, distinct from a request authenticated with
`blue-secret`. A key that doesn't match any configured `--api-key` or `--tenant-api-key`
value is rejected with `401 Unauthorized`. Tenant keys authenticate serving endpoints
only — `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, etc. They do not grant
access to admin/management routes (`/workers`, `/flush_cache`, and similar); that surface
requires either the shared `--api-key` or control-plane authentication.

---

## Endpoints

### Chat Completions

Create a chat completion.

```
POST /v1/chat/completions
```

#### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | Yes | Model identifier |
| `messages` | array | Yes | Array of message objects |
| `max_completion_tokens` | integer | No | Upper bound on generated completion tokens |
| `max_tokens` | integer | No | Deprecated — use `max_completion_tokens`. Still accepted and transparently migrated |
| `temperature` | number | No | Sampling temperature (0-2) |
| `top_p` | number | No | Nucleus sampling parameter |
| `n` | integer | No | Number of completions to generate (1-10) |
| `stream` | boolean | No | Enable streaming responses |
| `stop` | string/array | No | Stop sequences |
| `presence_penalty` | number | No | Presence penalty (-2 to 2) |
| `frequency_penalty` | number | No | Frequency penalty (-2 to 2) |
| `user` | string | No | End-user identifier |

#### Message Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `role` | string | Yes | `system`, `user`, `assistant`, `tool`, `function`, or `developer` |
| `content` | string | Yes | Message content |

#### Example Request

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

#### Response

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1705312345,
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
    "prompt_tokens": 25,
    "completion_tokens": 8,
    "total_tokens": 33
  }
}
```

#### Streaming Response

With `"stream": true`, responses are sent as Server-Sent Events:

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

Response:

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"}}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"delta":{"content":"!"}}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

---

### Completions

Create a text completion (legacy API).

```
POST /v1/completions
```

#### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | Yes | Model identifier |
| `prompt` | string/array | Yes | Text prompt(s) |
| `max_tokens` | integer | No | Maximum tokens to generate |
| `temperature` | number | No | Sampling temperature (0-2) |
| `top_p` | number | No | Nucleus sampling parameter |
| `n` | integer | No | Number of completions |
| `stream` | boolean | No | Enable streaming |
| `stop` | string/array | No | Stop sequences |
| `echo` | boolean | No | Echo prompt in response |

#### Example Request

```bash
curl http://localhost:30000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "The quick brown fox",
    "max_tokens": 50
  }'
```

#### Response

```json
{
  "id": "cmpl-abc123",
  "object": "text_completion",
  "created": 1705312345,
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "choices": [
    {
      "text": " jumps over the lazy dog.",
      "index": 0,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 4,
    "completion_tokens": 7,
    "total_tokens": 11
  }
}
```

---

### List Models

List available models.

```
GET /v1/models
```

#### Example Request

```bash
curl http://localhost:30000/v1/models
```

#### Response

```json
{
  "object": "list",
  "data": [
    {
      "id": "meta-llama/Llama-3.1-8B-Instruct",
      "object": "model",
      "created": 0,
      "owned_by": "self_hosted"
    }
  ]
}
```

`owned_by` is `self_hosted` for locally hosted workers, or the provider name (for example `openai`, `anthropic`, `xai`, `gemini`) for upstream providers.

---

### Audio Transcriptions

Transcribe an audio file (batch). Requires a worker serving an ASR model.

```
POST /v1/audio/transcriptions
```

Sent as `multipart/form-data` with fields `file` (the audio) and `model`, plus optional
`language`, `prompt`, `response_format`, `temperature`, and `stream`.

```bash
curl http://localhost:30000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=Qwen/Qwen3-ASR-1.7B
```

---

### Realtime API

SMG proxies the OpenAI Realtime API to a realtime-capable worker. Both the OpenAI router
(to an upstream provider) and the HTTP router (to a **local** worker labeled
[`realtime: "true"`](../../getting-started/multiple-workers.md#realtime-capable-workers))
support it. SMG relays frames verbatim, so the worker must speak the OpenAI Realtime
protocol — for local workers, for example vLLM serving an ASR model with the realtime task.

| Endpoint | Transport | Purpose |
|----------|-----------|---------|
| `GET /v1/realtime` | WebSocket | Bidirectional realtime session (e.g. live streaming transcription) |
| `POST /v1/realtime/calls` | WebRTC (SDP) | Browser/WebRTC realtime session |
| `POST /v1/realtime/sessions` | HTTP | Create a realtime session |
| `POST /v1/realtime/client_secrets` | HTTP | Mint an ephemeral client secret |
| `POST /v1/realtime/transcription_sessions` | HTTP | Create a realtime transcription session |

#### WebSocket example

```python
# pip install websockets
import asyncio, websockets

async def main():
    url = "ws://localhost:30000/v1/realtime?model=Qwen/Qwen3-ASR-1.7B"
    headers = {"Authorization": "Bearer your-api-key"}
    async with websockets.connect(url, additional_headers=headers) as ws:
        # Send realtime events (session.update, input_audio_buffer.append, ...)
        # and receive transcription/response events from the worker.
        ...

asyncio.run(main())
```

---

## Error Responses

### Error Format

```json
{
  "error": {
    "message": "Error description",
    "type": "error_type",
    "code": "error_code"
  }
}
```

### Error Codes

| HTTP Status | Type | Description |
|-------------|------|-------------|
| 400 | `invalid_request_error` | Malformed request |
| 401 | `authentication_error` | Invalid or missing API key |
| 404 | `not_found_error` | Model or endpoint not found |
| 408 | `timeout_error` | Request timed out in queue |
| 429 | `rate_limit_error` | Rate limit exceeded |
| 500 | `internal_error` | Server error |
| 503 | `service_unavailable` | No healthy workers |

### Example Error Response

```json
{
  "error": {
    "message": "Rate limit exceeded. Please retry later.",
    "type": "rate_limit_error",
    "code": "rate_limit_exceeded"
  }
}
```

---

## Client Libraries

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="your-api-key"  # or "not-needed" if auth disabled
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

### JavaScript/TypeScript

```typescript
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'http://localhost:30000/v1',
  apiKey: 'your-api-key'
});

const response = await client.chat.completions.create({
  model: 'meta-llama/Llama-3.1-8B-Instruct',
  messages: [
    { role: 'user', content: 'Hello!' }
  ]
});

console.log(response.choices[0].message.content);
```

### cURL

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## Request Headers

| Header | Required | Description |
|--------|----------|-------------|
| `Content-Type` | Yes | Must be `application/json` |
| `Authorization` | Conditional | `Bearer {api-key}` if auth enabled |
| `X-Request-ID` | No | Custom request ID for tracing |

---

## Rate Limiting

When rate limited, responses include:

| Header | Description |
|--------|-------------|
| `Retry-After` | Seconds to wait before retrying |
| `X-RateLimit-Limit` | Request limit |
| `X-RateLimit-Remaining` | Remaining requests |
| `X-RateLimit-Reset` | Unix timestamp when limit resets |
