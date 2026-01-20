# Admin API Reference

SMG provides administrative endpoints for managing tokenizers, workers, cache, and cluster operations.

!!! tip "Related Documentation"
    For health checks, worker status, and monitoring endpoints, see [Gateway Extensions](extensions.md).

---

## Tokenizer Management

Manage tokenizers for text processing and tokenization.

!!! note "Authentication Required"
    These endpoints require admin authentication via API key or control plane credentials.

### Add Tokenizer

```
POST /v1/tokenizers
```

Adds a new tokenizer from a local path or HuggingFace model ID.

**Request Body:**
```json
{
  "name": "llama3-tokenizer",
  "source": "meta-llama/Meta-Llama-3-8B",
  "chat_template_path": "/path/to/template.jinja"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Unique tokenizer identifier |
| `source` | string | Yes | HuggingFace model ID or local path |
| `chat_template_path` | string | No | Path to custom Jinja2 chat template |

**Response:** `202 Accepted`
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Tokenizer loading initiated"
}
```

---

### List Tokenizers

```
GET /v1/tokenizers
```

Returns all registered tokenizers.

**Response:** `200 OK`
```json
{
  "tokenizers": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "name": "llama3-tokenizer",
      "source": "meta-llama/Meta-Llama-3-8B",
      "vocab_size": 128256
    }
  ]
}
```

---

### Get Tokenizer

```
GET /v1/tokenizers/{tokenizer_id}
```

Returns details for a specific tokenizer.

**Response:** `200 OK`
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "llama3-tokenizer",
  "source": "meta-llama/Meta-Llama-3-8B",
  "vocab_size": 128256
}
```

**Response:** `404 Not Found`
```json
{
  "error": {
    "message": "Tokenizer not found",
    "type": "not_found"
  }
}
```

---

### Get Tokenizer Status

```
GET /v1/tokenizers/{tokenizer_id}/status
```

Returns the loading status of a tokenizer.

**Response:** `200 OK`
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "message": "Tokenizer loaded successfully",
  "vocab_size": 128256
}
```

| Status | Description |
|--------|-------------|
| `pending` | Tokenizer loading queued |
| `processing` | Tokenizer currently loading |
| `completed` | Tokenizer ready for use |
| `failed` | Loading failed (see message) |

---

### Remove Tokenizer

```
DELETE /v1/tokenizers/{tokenizer_id}
```

Removes a tokenizer.

**Response:** `200 OK`
```json
{
  "success": true,
  "message": "Tokenizer removed successfully"
}
```

---

## Worker Management

Manage backend inference workers.

!!! tip
    For listing workers and viewing metrics, see [Gateway Extensions](extensions.md#worker-management).

### Create Worker

```
POST /workers
```

Registers a new backend worker.

**Request Body:**
```json
{
  "name": "gpu-worker-1",
  "url": "http://gpu1:8000",
  "model_name": "llama3-70b",
  "api_key": "worker-secret-key"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Worker identifier |
| `url` | string | Yes | Worker base URL |
| `model_name` | string | No | Model served by worker |
| `api_key` | string | No | API key for worker auth |

**Response:** `201 Created`
```json
{
  "id": "worker-1",
  "name": "gpu-worker-1",
  "url": "http://gpu1:8000",
  "status": "healthy"
}
```

---

### Update Worker

```
PUT /workers/{worker_id}
```

Updates worker configuration.

**Request Body:**
```json
{
  "name": "gpu-worker-1-updated",
  "api_key": "new-api-key"
}
```

**Response:** `200 OK`

---

### Delete Worker

```
DELETE /workers/{worker_id}
```

Removes a worker from the pool.

**Response:** `200 OK`
```json
{
  "success": true,
  "message": "Worker removed successfully"
}
```

---

## Cache Management

Manage the routing cache and load information.

### Flush Cache

```
POST /flush_cache
```

Flushes the KV cache on all workers.

**Response:** `200 OK`
```json
{
  "flushed_workers": 3,
  "success": true
}
```

---

### Get Loads

```
GET /get_loads
```

Returns current load distribution across workers.

**Response:** `200 OK`
```json
{
  "loads": [
    {
      "worker_id": "worker-1",
      "url": "http://gpu1:8000",
      "active_requests": 5,
      "queue_depth": 2,
      "cache_utilization": 0.75
    }
  ]
}
```

---

## Model Information

Query model and server information.

### List Models

```
GET /v1/models
```

Returns available models (proxied to workers).

**Response:** `200 OK`
```json
{
  "object": "list",
  "data": [
    {
      "id": "llama3-70b",
      "object": "model",
      "created": 1700000000,
      "owned_by": "meta"
    }
  ]
}
```

---

### Get Model Info

```
GET /get_model_info
```

Returns detailed model information (proxied to workers).

**Response:** `200 OK`
```json
{
  "model_name": "llama3-70b",
  "max_tokens": 8192,
  "vocab_size": 128256
}
```

---

### Get Server Info

```
GET /get_server_info
```

Returns server information (proxied to workers).

**Response:** `200 OK`
```json
{
  "version": "0.1.0",
  "backend": "vllm",
  "gpu_count": 8
}
```

---

## WASM Module Management

Manage WebAssembly plugins.

### Add WASM Module

```
POST /wasm
```

Uploads and registers a WASM module.

**Request:** Multipart form with WASM binary

**Response:** `201 Created`
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "custom-filter",
  "status": "loaded"
}
```

---

### List WASM Modules

```
GET /wasm
```

Returns all registered WASM modules.

**Response:** `200 OK`
```json
{
  "modules": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "name": "custom-filter",
      "status": "loaded"
    }
  ]
}
```

---

### Remove WASM Module

```
DELETE /wasm/{module_uuid}
```

Removes a WASM module.

**Response:** `200 OK`
```json
{
  "success": true,
  "message": "Module removed successfully"
}
```

---

## Error Responses

All endpoints return errors in a consistent format:

```json
{
  "error": {
    "message": "Detailed error description",
    "type": "error_type"
  }
}
```

| HTTP Status | Error Type | Description |
|-------------|------------|-------------|
| `400` | `bad_request` | Invalid request format or parameters |
| `401` | `unauthorized` | Missing or invalid authentication |
| `403` | `forbidden` | Insufficient permissions |
| `404` | `not_found` | Resource not found |
| `409` | `conflict` | Resource already exists |
| `503` | `service_unavailable` | No healthy workers available |

---

## Authentication

Admin endpoints require authentication via one of:

1. **API Key**: Pass via `Authorization: Bearer <api-key>` header
2. **Control Plane Key**: For cluster management operations

Public endpoints (health checks, model info) do not require authentication.
