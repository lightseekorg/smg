---
title: Control Plane Operations
---

# Control Plane Operations

This guide covers day-2 admin workflows backed by control-plane endpoints in `model_gateway/src/server.rs`: workers, tokenizers, WASM modules, parser utilities, and cache/load operations.

<div class="prerequisites" markdown>

#### Before you begin

- Completed [Control Plane Auth](control-plane-auth.md)
- Set an admin bearer token (JWT or API key), for example:
  `export ADMIN_TOKEN=super-secret-key`

</div>

---

## Auth Header

Use the same header for all control-plane calls:

```bash
-H "Authorization: Bearer ${ADMIN_TOKEN}"
```

Control-plane middleware requires admin role for these operations.

---

## 1. Worker Management

Create worker:

```bash
curl http://localhost:30000/workers \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${ADMIN_TOKEN}" \
  -d '{
    "url": "grpc://localhost:50051",
    "model_id": "meta-llama/Llama-3.1-8B-Instruct",
    "worker_type": "regular",
    "runtime": "sglang"
  }'
```

List workers:

```bash
curl http://localhost:30000/workers \
  -H "Authorization: Bearer ${ADMIN_TOKEN}"
```

Get/update/delete by worker ID:

```bash
curl http://localhost:30000/workers/<worker_id> \
  -H "Authorization: Bearer ${ADMIN_TOKEN}"

curl -X PUT http://localhost:30000/workers/<worker_id> \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${ADMIN_TOKEN}" \
  -d '{"priority": 100}'

curl -X DELETE http://localhost:30000/workers/<worker_id> \
  -H "Authorization: Bearer ${ADMIN_TOKEN}"
```

---

## 2. Tokenizer Registry

Add tokenizer:

```bash
curl -X POST http://localhost:30000/v1/tokenizers \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${ADMIN_TOKEN}" \
  -d '{
    "name": "llama3-main",
    "source": "meta-llama/Llama-3.1-8B-Instruct"
  }'
```

List/get/status/delete:

```bash
curl http://localhost:30000/v1/tokenizers \
  -H "Authorization: Bearer ${ADMIN_TOKEN}"

curl http://localhost:30000/v1/tokenizers/<tokenizer_id> \
  -H "Authorization: Bearer ${ADMIN_TOKEN}"

curl http://localhost:30000/v1/tokenizers/<tokenizer_id>/status \
  -H "Authorization: Bearer ${ADMIN_TOKEN}"

curl -X DELETE http://localhost:30000/v1/tokenizers/<tokenizer_id> \
  -H "Authorization: Bearer ${ADMIN_TOKEN}"
```

---

## 3. WASM Module Management

Enable WASM support at startup:

```bash
smg launch \
  --worker-urls http://worker:8000 \
  --enable-wasm
```

Register module:

```bash
curl -X POST http://localhost:30000/wasm \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${ADMIN_TOKEN}" \
  -d '{
    "modules": [
      {
        "name": "audit-middleware",
        "file_path": "/opt/wasm/audit.wasm",
        "module_type": "Middleware",
        "attach_points": [{"Middleware":"OnRequest"}]
      }
    ]
  }'
```

List/remove modules:

```bash
curl http://localhost:30000/wasm \
  -H "Authorization: Bearer ${ADMIN_TOKEN}"

curl -X DELETE http://localhost:30000/wasm/<module_uuid> \
  -H "Authorization: Bearer ${ADMIN_TOKEN}"
```

---

## 4. Parser Utilities

Function call parsing:

```bash
curl -X POST http://localhost:30000/parse/function_call \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${ADMIN_TOKEN}" \
  -d '{
    "text": "{\"name\":\"get_weather\",\"arguments\":{\"city\":\"SF\"}}",
    "tool_call_parser": "json",
    "tools": []
  }'
```

Reasoning separation:

```bash
curl -X POST http://localhost:30000/parse/reasoning \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${ADMIN_TOKEN}" \
  -d '{
    "text": "<think>internal</think>answer",
    "reasoning_parser": "deepseek_r1"
  }'
```

---

## 5. Cache and Load Operations

Flush worker KV caches:

```bash
curl -X POST http://localhost:30000/flush_cache \
  -H "Authorization: Bearer ${ADMIN_TOKEN}"
```

Inspect worker loads:

```bash
curl http://localhost:30000/get_loads \
  -H "Authorization: Bearer ${ADMIN_TOKEN}"
```

---

## Next Steps

- [Admin API Reference](../reference/api/admin.md)
- [Extension API Reference](../reference/api/extensions.md)
- [Configuration Reference](../reference/configuration.md#control-plane-authentication)
