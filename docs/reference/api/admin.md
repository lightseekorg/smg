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
  "message": "Tokenizer 'llama3-tokenizer' registration job submitted. Loading from: meta-llama/Meta-Llama-3-8B"
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
    "message": "Tokenizer 'llama3-tokenizer' not found",
    "type": "tokenizer_not_found"
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
  "message": "Tokenizer 'llama3-tokenizer' is loaded and ready",
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
  "message": "Tokenizer 'llama3-tokenizer' removed successfully"
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
  "url": "http://gpu1:8000",
  "models": [{ "id": "llama3-70b" }],
  "api_key": "worker-secret-key"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `url` | string | Yes | Worker base URL |
| `worker_type` | string | No | `regular`, `prefill`, or `decode` (default: `regular`) |
| `connection_mode` | string | No | `http` or `grpc` (default: `http`) |
| `runtime_type` | string | No | `sglang`, `vllm`, `trtllm`, `mlx`, `external`, or `unspecified` (default: `unspecified`, which triggers auto-detection) |
| `models` | array | No | Model cards served by this worker (empty = wildcard) |
| `api_key` | string | No | API key for worker authentication |
| `priority` | integer | No | Routing priority (higher = preferred, default: 50) |

**Response:** `202 Accepted`
```json
{
  "status": "accepted",
  "worker_id": "worker-abc123",
  "url": "http://gpu1:8000",
  "location": "/workers/worker-abc123",
  "message": "Worker addition queued for background processing"
}
```

---

### Update Worker (partial)

```
PATCH /workers/{worker_id}
```

Partially updates worker configuration. Only the fields you include are changed.

**Request Body:**
```json
{
  "priority": 75,
  "api_key": "new-api-key"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `priority` | integer | New routing priority |
| `cost` | number | New cost factor |
| `labels` | object | Updated labels |
| `api_key` | string | New API key (for key rotation) |
| `health` | object | Partial health-check overrides (`timeout_secs`, `check_interval_secs`, `success_threshold`, `failure_threshold`, `disable_health_check`) |

**Response:** `202 Accepted`
```json
{
  "status": "accepted",
  "worker_id": "worker-abc123",
  "message": "Worker update queued for background processing"
}
```

---

### Replace Worker (full)

```
PUT /workers/{worker_id}
```

Re-runs the full worker registration workflow (model discovery and all). The request body must be a complete `WorkerSpec` whose `url` matches the existing worker's URL — URL changes are not supported via `PUT`; use `DELETE` + `POST` instead.

**Response:** `202 Accepted` with the same shape as `PATCH`.

---

### Delete Worker

```
DELETE /workers/{worker_id}
```

Removes a worker from the pool.

**Response:** `202 Accepted`
```json
{
  "status": "accepted",
  "worker_id": "worker-abc123",
  "message": "Worker removal queued for background processing"
}
```

---

## Cache Management

Manage the routing cache and load information.

### Flush Cache

```
POST /flush_cache
```

Flushes the KV cache on all HTTP workers. gRPC workers are skipped. The response status is `200 OK` on full success and `206 Partial Content` when some workers fail.

**Response:** `200 OK`
```json
{
  "status": "success",
  "message": "Successfully flushed cache on all 3 HTTP workers",
  "workers_flushed": 3,
  "total_http_workers": 3,
  "total_workers": 3
}
```

On partial failure, the response additionally includes `successful` (list of worker URLs) and `failed` (list of `{worker, error}` entries), and `status` becomes `"partial_success"`.

---

### Get Loads

```
GET /get_loads
```

Returns the current load distribution across workers. The gateway fans out to every registered worker (HTTP and gRPC) and returns whatever each backend reports. The `load` field is the total number of KV-cache tokens in use across all data-parallel ranks for that worker; `-1` indicates the worker failed to respond.

**Response:** `200 OK`
```json
{
  "workers": [
    {
      "worker": "http://gpu1:8000",
      "load": 1234,
      "details": {
        "timestamp": "2024-01-15T12:00:00Z",
        "dp_rank_count": 1,
        "loads": [
          {
            "dp_rank": 0,
            "num_running_reqs": 5,
            "num_waiting_reqs": 2,
            "num_total_reqs": 7,
            "num_used_tokens": 1234,
            "max_total_num_tokens": 16384,
            "token_usage": 0.075,
            "gen_throughput": 45.2,
            "cache_hit_rate": 0.82,
            "utilization": 0.31,
            "max_running_requests": 256
          }
        ]
      }
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

Manage WebAssembly plugins. Modules are registered from files accessible to the gateway process; the request body contains descriptors with paths, not binary payloads.

### Add WASM Module

```
POST /wasm
```

Registers one or more WASM modules.

**Request Body:** JSON `WasmModuleAddRequest`
```json
{
  "modules": [
    {
      "name": "custom-middleware",
      "file_path": "/etc/smg/wasm/custom-middleware.wasm",
      "module_type": "Middleware",
      "attach_points": [
        {"Middleware": "OnRequest"},
        {"Middleware": "OnResponse"}
      ]
    }
  ]
}
```

The only supported `module_type` today is `Middleware`. Valid `Middleware` attach points are `OnRequest`, `OnResponse`, and `OnError`.

**Response:** `200 OK` on full success, `400 Bad Request` if any module failed to register. The response body echoes every requested module with an `add_result` field indicating success (carrying the assigned UUID) or failure (carrying the error message).

```json
{
  "modules": [
    {
      "name": "custom-middleware",
      "file_path": "/etc/smg/wasm/custom-middleware.wasm",
      "module_type": "Middleware",
      "attach_points": [
        {"Middleware": "OnRequest"},
        {"Middleware": "OnResponse"}
      ],
      "add_result": {
        "Success": "550e8400-e29b-41d4-a716-446655440000"
      }
    }
  ]
}
```

---

### List WASM Modules

```
GET /wasm
```

Returns all registered WASM modules together with aggregate execution metrics.

**Response:** `200 OK`
```json
{
  "modules": [
    {
      "module_uuid": "550e8400-e29b-41d4-a716-446655440000",
      "module_meta": {
        "name": "custom-middleware",
        "file_path": "/etc/smg/wasm/custom-middleware.wasm",
        "sha256_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "size_bytes": 65536,
        "created_at": "2024-01-15T12:00:00.000000000Z",
        "last_accessed_at": "2024-01-15T12:05:00.000000000Z",
        "access_count": 42,
        "attach_points": [
          {"Middleware": "OnRequest"}
        ]
      }
    }
  ],
  "metrics": {
    "total_executions": 42,
    "successful_executions": 42,
    "failed_executions": 0,
    "total_execution_time_ms": 125,
    "max_execution_time_ms": 8,
    "average_execution_time_ms": 2.97
  }
}
```

---

### Remove WASM Module

```
DELETE /wasm/{module_uuid}
```

Removes a WASM module. The body is a plain text status message, not JSON.

**Response:** `200 OK`
```
Module removed successfully
```

On failure returns `400 Bad Request` with the error text as the body.

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
