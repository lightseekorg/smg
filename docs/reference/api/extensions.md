---
title: Extension API
---

# Extension API Reference

SMG provides additional endpoints beyond the OpenAI-compatible API for gateway management, health checks, and operational tasks.

---

## Health Endpoints

### Health Check

Basic health check endpoint.

```
GET /health
```

#### Response

```json
{
  "status": "ok"
}
```

#### Status Codes

| Status | Description |
|--------|-------------|
| 200 | Gateway is healthy |
| 503 | Gateway is unhealthy |

---

### Readiness Check

Kubernetes readiness probe endpoint. Returns healthy only when at least one worker is available.

```
GET /readiness
```

#### Response (Ready)

```json
{
  "status": "ready",
  "healthy_workers": 3,
  "total_workers": 3
}
```

#### Response (Not Ready)

```json
{
  "status": "not_ready",
  "healthy_workers": 0,
  "total_workers": 3,
  "reason": "No healthy workers available"
}
```

#### Status Codes

| Status | Description |
|--------|-------------|
| 200 | Ready to serve traffic |
| 503 | Not ready (no healthy workers) |

---

### Liveness Check

Kubernetes liveness probe endpoint.

```
GET /liveness
```

#### Response

```json
{
  "status": "alive"
}
```

---

## Worker Management

### List Workers

Get information about all configured workers.

```
GET /workers
```

#### Response

```json
{
  "workers": [
    {
      "url": "http://worker1:8000",
      "healthy": true,
      "model": "meta-llama/Llama-3.1-8B-Instruct",
      "last_health_check": "2024-01-15T10:30:00Z",
      "requests_total": 15234,
      "requests_active": 5,
      "requests_failed": 12,
      "latency_p50_ms": 150,
      "latency_p99_ms": 450,
      "circuit_state": "closed"
    },
    {
      "url": "http://worker2:8000",
      "healthy": true,
      "model": "meta-llama/Llama-3.1-8B-Instruct",
      "last_health_check": "2024-01-15T10:30:00Z",
      "requests_total": 14892,
      "requests_active": 3,
      "requests_failed": 8,
      "latency_p50_ms": 145,
      "latency_p99_ms": 420,
      "circuit_state": "closed"
    }
  ],
  "total": 2,
  "healthy": 2
}
```

#### Worker Object

| Field | Type | Description |
|-------|------|-------------|
| `url` | string | Worker URL |
| `healthy` | boolean | Current health status |
| `model` | string | Model served by worker |
| `last_health_check` | string | ISO 8601 timestamp of last health check |
| `requests_total` | integer | Total requests sent to worker |
| `requests_active` | integer | Currently processing requests |
| `requests_failed` | integer | Failed requests |
| `latency_p50_ms` | integer | Median latency in milliseconds |
| `latency_p99_ms` | integer | 99th percentile latency |
| `circuit_state` | string | Circuit breaker state: `closed`, `open`, `half_open` |

---

### Get Worker Details

Get detailed information about a specific worker.

```
GET /workers/{worker_url}
```

!!! note
    The worker URL must be URL-encoded (e.g., `http%3A%2F%2Fworker1%3A8000`).

#### Example

```bash
curl http://localhost:30000/workers/http%3A%2F%2Fworker1%3A8000
```

#### Response

```json
{
  "url": "http://worker1:8000",
  "healthy": true,
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "last_health_check": "2024-01-15T10:30:00Z",
  "health_check_history": [
    {"timestamp": "2024-01-15T10:30:00Z", "success": true, "latency_ms": 5},
    {"timestamp": "2024-01-15T10:29:50Z", "success": true, "latency_ms": 4}
  ],
  "requests_total": 15234,
  "requests_active": 5,
  "requests_failed": 12,
  "latency_histogram": {
    "p50_ms": 150,
    "p75_ms": 220,
    "p90_ms": 350,
    "p95_ms": 400,
    "p99_ms": 450
  },
  "circuit_breaker": {
    "state": "closed",
    "failure_count": 0,
    "success_count": 1523,
    "last_failure": null,
    "last_state_change": "2024-01-15T08:00:00Z"
  }
}
```

---

### Worker Health Check

Trigger an immediate health check for a specific worker.

```
POST /workers/{worker_url}/health-check
```

#### Response

```json
{
  "url": "http://worker1:8000",
  "healthy": true,
  "latency_ms": 5,
  "checked_at": "2024-01-15T10:35:00Z"
}
```

---

## Gateway Configuration

### Get Configuration

Get current gateway configuration.

```
GET /config
```

#### Response

```json
{
  "policy": "cache_aware",
  "max_concurrent_requests": 100,
  "rate_limit_tokens_per_second": 512,
  "queue_size": 128,
  "queue_timeout_secs": 30,
  "circuit_breaker": {
    "threshold": 5,
    "timeout_secs": 30
  },
  "health_check": {
    "interval_secs": 10,
    "timeout_secs": 5,
    "path": "/health"
  },
  "service_discovery": {
    "enabled": true,
    "selector": "app=sglang-worker",
    "namespace": "inference",
    "port": 8000
  }
}
```

---

## Tokenization Endpoints

Direct tokenization and detokenization operations.

### Tokenize

```
POST /v1/tokenize
```

Converts text to token IDs.

#### Request Body

```json
{
  "model": "llama3",
  "prompt": "Hello, world!"
}
```

#### Request Body (batch)

```json
{
  "model": "llama3",
  "prompt": ["Hello, world!", "How are you?"]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | No | Model/tokenizer name |
| `prompt` | string or string[] | Yes | Text to tokenize |

#### Response

```json
{
  "tokens": [15496, 11, 1917, 0],
  "count": 4,
  "char_count": 13
}
```

#### Response (batch)

```json
{
  "tokens": [[15496, 11, 1917, 0], [2437, 527, 499, 30]],
  "count": [4, 4],
  "char_count": [13, 12]
}
```

---

### Detokenize

```
POST /v1/detokenize
```

Converts token IDs back to text.

#### Request Body

```json
{
  "model": "llama3",
  "tokens": [15496, 11, 1917, 0],
  "skip_special_tokens": true
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | Yes | Model/tokenizer name |
| `tokens` | int[] or int[][] | Yes | Token IDs to decode |
| `skip_special_tokens` | boolean | No | Skip special tokens (default: true) |

#### Response

```json
{
  "text": "Hello, world!"
}
```

---

## Parser Endpoints

Parse model outputs for tool calls and reasoning chains.

### Parse Function Call

```
POST /parse/function_call
```

Extracts tool/function calls from model output.

#### Request Body

```json
{
  "text": "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"London\"}}</tool_call>",
  "tool_call_parser": "json",
  "tools": [
    {
      "name": "get_weather",
      "description": "Get weather for a city",
      "parameters": {
        "type": "object",
        "properties": {
          "city": {"type": "string"}
        }
      }
    }
  ]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | Yes | Model output to parse |
| `tool_call_parser` | string | Yes | Parser type (see below) |
| `tools` | array | No | Tool definitions for validation |

#### Supported Parsers

| Parser | Description |
|--------|-------------|
| `json` | Generic JSON tool call format |
| `pythonic` | Python function call syntax |
| `mistral` | Mistral tool call format |
| `qwen` | Qwen tool call format |
| `llama` | Llama tool call format |
| `deepseek` | DeepSeek tool call format |

#### Response

```json
{
  "remaining_text": "",
  "tool_calls": [
    {
      "name": "get_weather",
      "arguments": {"city": "London"}
    }
  ],
  "success": true
}
```

---

### Parse Reasoning

```
POST /parse/reasoning
```

Separates reasoning/thinking from model output.

#### Request Body

```json
{
  "text": "<think>Let me analyze this step by step...</think>The answer is 42.",
  "reasoning_parser": "deepseek_r1"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | Yes | Model output to parse |
| `reasoning_parser` | string | Yes | Parser type (see below) |

#### Supported Parsers

| Parser | Description |
|--------|-------------|
| `deepseek_r1` | DeepSeek-R1 thinking format |
| `qwen3` | Qwen3 reasoning format |

#### Response

```json
{
  "normal_text": "The answer is 42.",
  "reasoning_text": "Let me analyze this step by step...",
  "success": true
}
```

---

## Metrics Endpoint

### Prometheus Metrics

Get Prometheus-formatted metrics.

```
GET /metrics
```

Available on the metrics port (default: 29000).

#### Example

```bash
curl http://localhost:29000/metrics
```

#### Response

```
# HELP smg_requests_total Total number of requests
# TYPE smg_requests_total counter
smg_requests_total{method="POST",path="/v1/chat/completions",status="200"} 15234

# HELP smg_request_duration_seconds Request duration in seconds
# TYPE smg_request_duration_seconds histogram
smg_request_duration_seconds_bucket{le="0.1"} 5000
smg_request_duration_seconds_bucket{le="0.5"} 12000
smg_request_duration_seconds_bucket{le="1.0"} 14500
smg_request_duration_seconds_bucket{le="+Inf"} 15234

# HELP smg_worker_health Worker health status
# TYPE smg_worker_health gauge
smg_worker_health{worker="http://worker1:8000"} 1
smg_worker_health{worker="http://worker2:8000"} 1
```

See [Metrics Reference](../metrics.md) for complete metrics documentation.

---

## Debug Endpoints

!!! warning
    Debug endpoints should be disabled in production or protected by authentication.

### Request Trace

Get detailed trace information for a request.

```
GET /debug/trace/{request_id}
```

#### Response

```json
{
  "request_id": "abc123",
  "received_at": "2024-01-15T10:30:00.000Z",
  "completed_at": "2024-01-15T10:30:00.250Z",
  "duration_ms": 250,
  "worker": "http://worker1:8000",
  "policy_decision": {
    "policy": "cache_aware",
    "candidates": ["http://worker1:8000", "http://worker2:8000"],
    "selected": "http://worker1:8000",
    "reason": "highest_prefix_match",
    "prefix_match_ratio": 0.85
  },
  "queue_wait_ms": 5,
  "worker_latency_ms": 245
}
```

### Gateway Stats

Get internal gateway statistics.

```
GET /debug/stats
```

#### Response

```json
{
  "uptime_secs": 86400,
  "requests_total": 1523400,
  "requests_active": 45,
  "queue_depth": 12,
  "workers_total": 4,
  "workers_healthy": 4,
  "memory_usage_bytes": 52428800,
  "goroutines": 150
}
```

---

## Error Responses

All extension endpoints return errors in a consistent format:

```json
{
  "error": {
    "message": "Worker not found",
    "type": "not_found_error",
    "code": "worker_not_found"
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `worker_not_found` | 404 | Specified worker does not exist |
| `invalid_request` | 400 | Malformed request |
| `internal_error` | 500 | Internal server error |
| `service_unavailable` | 503 | Service temporarily unavailable |
