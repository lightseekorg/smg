---
title: Anthropic Messages API
---

# Anthropic Messages API

SMG supports the Anthropic Messages API (`/v1/messages`), enabling applications to use Claude models through the gateway. Both HTTP proxy mode (forwarding to Anthropic's API) and gRPC mode (routing to local inference backends) are supported.

---

## Endpoint

Create a message.

```
POST /v1/messages
```

For streaming responses, set `"stream": true` in the request body.

---

## Request Example

```bash
curl http://localhost:30000/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": "What is the meaning of life?"}
    ]
  }'
```

---

## Streaming

To receive responses as Server-Sent Events, set `"stream": true`:

```bash
curl http://localhost:30000/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": "What is the meaning of life?"}
    ],
    "stream": true
  }'
```

---

## gRPC Backend

The Messages API works with gRPC backends such as SGLang and vLLM. When routing to a gRPC backend, SMG translates the Anthropic message format to the backend's native format and translates the response back.

!!! note
    When using the Messages API with gRPC backends, SMG handles format translation automatically. The backend receives requests in its native format.

---

## Connection Modes

| Mode | Backend | Description |
|------|---------|-------------|
| HTTP (proxy) | Anthropic API | Forward requests to `api.anthropic.com` |
| gRPC | SGLang/vLLM | Translate and route to local inference |

---

## Features

- Streaming and non-streaming responses
- Tool use (via MCP integration)
- Extended thinking
- Multi-turn conversations
