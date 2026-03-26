---
title: Metrics Reference
---

# Metrics Reference

Complete reference for Prometheus metrics exposed by SMG. Metrics are organized in six layers matching the request lifecycle.

---

## Metrics Endpoint

Metrics are exposed on the Prometheus port (default: `29000`):

```bash
curl http://localhost:29000/metrics
```

Configure via CLI:

```bash
smg --prometheus-port 29000 --prometheus-host 0.0.0.0
```

---

## Layer 1: HTTP Metrics

Metrics for incoming HTTP requests at the gateway edge.

### `smg_http_requests_total`

Total HTTP requests received by the gateway.

| Type | Labels |
|------|--------|
| Counter | `method`, `path` |

```promql
# Request rate by endpoint
sum by (path) (rate(smg_http_requests_total[5m]))

# Total request rate
sum(rate(smg_http_requests_total[5m]))
```

---

### `smg_http_request_duration_seconds`

HTTP request duration from receipt to response.

| Type | Labels |
|------|--------|
| Histogram | `method`, `path` |

```promql
# P99 latency by endpoint
histogram_quantile(0.99, sum by (path, le) (rate(smg_http_request_duration_seconds_bucket[5m])))

# Average latency
rate(smg_http_request_duration_seconds_sum[5m]) / rate(smg_http_request_duration_seconds_count[5m])
```

---

### `smg_http_responses_total`

HTTP responses by status and error code.

| Type | Labels |
|------|--------|
| Counter | `status_code`, `error_code` |

```promql
# Error rate (5xx responses)
sum(rate(smg_http_responses_total{status_code=~"5.."}[5m])) / sum(rate(smg_http_responses_total[5m]))

# Success rate
sum(rate(smg_http_responses_total{status_code="200"}[5m])) / sum(rate(smg_http_responses_total[5m]))
```

---

### `smg_http_connections_active`

Currently active HTTP connections.

| Type | Labels |
|------|--------|
| Gauge | None |

---

### `smg_http_inflight_request_age_count`

Distribution of in-flight request ages for Grafana heatmaps.

| Type | Labels |
|------|--------|
| Gauge | `gt`, `le` |

Age buckets (seconds): 30, 60, 180, 300, 600, 1200, 3600, 7200, 14400, 28800, 86400

---

### `smg_http_rate_limit_total`

Rate limiting decisions.

| Type | Labels |
|------|--------|
| Counter | `decision` |

Values: `allowed`, `rejected`

```promql
# Rejection rate
rate(smg_http_rate_limit_total{decision="rejected"}[5m]) / sum(rate(smg_http_rate_limit_total[5m]))
```

---

## Layer 2: Router Metrics

Metrics for request routing and processing.

### `smg_router_requests_total`

Requests processed by the router.

| Type | Labels |
|------|--------|
| Counter | `router_type`, `backend_type`, `connection_mode`, `model`, `endpoint`, `streaming` |

Router types: `grpc`, `http`, `third_party`
Backend types: `grpc`, `http`, `external`
Endpoints: `chat`, `generate`, `embeddings`, `rerank`, `responses`

```promql
# Request rate by model
sum by (model) (rate(smg_router_requests_total[5m]))

# Streaming vs non-streaming
sum by (streaming) (rate(smg_router_requests_total[5m]))
```

---

### `smg_router_request_duration_seconds`

Total router request duration.

| Type | Labels |
|------|--------|
| Histogram | `router_type`, `backend_type`, `model`, `endpoint` |

---

### `smg_router_request_errors_total`

Router errors by type.

| Type | Labels |
|------|--------|
| Counter | `router_type`, `error_type` |

Error types: `timeout`, `connection`, `upstream`, `internal`, `validation`

```promql
# Error rate by type
sum by (error_type) (rate(smg_router_request_errors_total[5m]))
```

---

### `smg_router_stage_duration_seconds`

Duration of individual pipeline stages (gRPC mode only).

| Type | Labels |
|------|--------|
| Histogram | `stage` |

Stages: `tokenize`, `chat_template`, `route`, `inference`, `detokenize`, `tool_parse`

```promql
# Tokenization latency
histogram_quantile(0.99, rate(smg_router_stage_duration_seconds_bucket{stage="tokenize"}[5m]))
```

---

### `smg_router_ttft_seconds`

Time to first token (gRPC streaming only).

| Type | Labels |
|------|--------|
| Histogram | `model` |

```promql
# P50 TTFT by model
histogram_quantile(0.5, sum by (model, le) (rate(smg_router_ttft_seconds_bucket[5m])))
```

---

### `smg_router_tpot_seconds`

Time per output token (gRPC streaming only).

| Type | Labels |
|------|--------|
| Histogram | `model` |

```promql
# Average TPOT
rate(smg_router_tpot_seconds_sum[5m]) / rate(smg_router_tpot_seconds_count[5m])
```

---

### `smg_router_tokens_total`

Token counts by type.

| Type | Labels |
|------|--------|
| Counter | `type`, `model` |

Types: `input`, `output`

```promql
# Tokens per second
sum by (type) (rate(smg_router_tokens_total[5m]))

# Input/output ratio
sum(rate(smg_router_tokens_total{type="output"}[5m])) / sum(rate(smg_router_tokens_total{type="input"}[5m]))
```

---

### `smg_router_generation_duration_seconds`

Total generation time (first token to last token).

| Type | Labels |
|------|--------|
| Histogram | `model` |

---

### `smg_router_upstream_responses_total`

HTTP responses from upstream workers.

| Type | Labels |
|------|--------|
| Counter | `status_code` |

---

## Layer 3: Worker Metrics

Metrics for worker pool management and resilience.

### `smg_worker_pool_size`

Number of workers in the pool.

| Type | Labels |
|------|--------|
| Gauge | None |

---

### `smg_worker_connections_active`

Active connections per worker.

| Type | Labels |
|------|--------|
| Gauge | `worker` |

---

### `smg_worker_requests_active`

Active requests per worker.

| Type | Labels |
|------|--------|
| Gauge | `worker` |

```promql
# Load distribution across workers
smg_worker_requests_active / ignoring(worker) group_left sum(smg_worker_requests_active)
```

---

### `smg_worker_health`

Worker health status.

| Type | Labels | Values |
|------|--------|--------|
| Gauge | `worker` | `1` = healthy, `0` = unhealthy |

```promql
# Count healthy workers
sum(smg_worker_health)

# Alert on unhealthy workers
smg_worker_health == 0
```

---

### `smg_worker_health_checks_total`

Health check results.

| Type | Labels |
|------|--------|
| Counter | `worker`, `result` |

Results: `success`, `failure`

---

### `smg_worker_selection_total`

Worker selection events by load balancer.

| Type | Labels |
|------|--------|
| Counter | `worker`, `policy` |

---

### `smg_worker_errors_total`

Worker errors by type.

| Type | Labels |
|------|--------|
| Counter | `worker`, `error_type` |

---

### Circuit Breaker Metrics

#### `smg_worker_cb_state`

Circuit breaker state per worker.

| Type | Labels | Values |
|------|--------|--------|
| Gauge | `worker` | `0` = closed, `1` = open, `2` = half-open |

```promql
# Workers with open circuits
count(smg_worker_cb_state == 1)
```

#### `smg_worker_cb_transitions_total`

Circuit breaker state transitions.

| Type | Labels |
|------|--------|
| Counter | `worker`, `from`, `to` |

#### `smg_worker_cb_outcomes_total`

Request outcomes tracked by circuit breaker.

| Type | Labels |
|------|--------|
| Counter | `worker`, `outcome` |

Outcomes: `success`, `failure`

#### `smg_worker_cb_consecutive_failures`

Consecutive failures per worker.

| Type | Labels |
|------|--------|
| Gauge | `worker` |

#### `smg_worker_cb_consecutive_successes`

Consecutive successes per worker.

| Type | Labels |
|------|--------|
| Gauge | `worker` |

---

### Retry Metrics

#### `smg_worker_retries_total`

Retry attempts.

| Type | Labels |
|------|--------|
| Counter | `worker`, `attempt` |

#### `smg_worker_retries_exhausted_total`

Requests that exhausted all retries.

| Type | Labels |
|------|--------|
| Counter | `worker` |

#### `smg_worker_retry_backoff_seconds`

Retry backoff durations.

| Type | Labels |
|------|--------|
| Histogram | `worker` |

---

## Layer 4: Discovery Metrics

Metrics for service discovery.

### `smg_discovery_registrations_total`

Worker registrations.

| Type | Labels |
|------|--------|
| Counter | `source`, `result` |

Sources: `static`, `kubernetes`, `consul`, `manual`

---

### `smg_discovery_deregistrations_total`

Worker deregistrations.

| Type | Labels |
|------|--------|
| Counter | `source`, `reason` |

---

### `smg_discovery_sync_duration_seconds`

Discovery sync duration.

| Type | Labels |
|------|--------|
| Histogram | `source` |

---

### `smg_discovery_workers_discovered`

Workers discovered per source.

| Type | Labels |
|------|--------|
| Gauge | `source` |

---

## Layer 5: MCP Tool Metrics

Metrics for Model Context Protocol tool execution.

### `smg_mcp_tool_calls_total`

MCP tool invocations.

| Type | Labels |
|------|--------|
| Counter | `model`, `tool_name`, `result` |

Results: `success`, `error`

```promql
# Tool success rate
sum(rate(smg_mcp_tool_calls_total{result="success"}[5m])) / sum(rate(smg_mcp_tool_calls_total[5m]))

# Most used tools
topk(10, sum by (tool_name) (rate(smg_mcp_tool_calls_total[5m])))
```

---

### `smg_mcp_tool_duration_seconds`

Tool execution duration.

| Type | Labels |
|------|--------|
| Histogram | `model`, `tool_name` |

---

### `smg_mcp_servers_active`

Active MCP servers.

| Type | Labels |
|------|--------|
| Gauge | None |

---

### `smg_mcp_tool_iterations_total`

Tool loop iterations in Responses API.

| Type | Labels |
|------|--------|
| Counter | `model` |

---

## Layer 6: Database Metrics

Metrics for storage operations.

### `smg_db_operations_total`

Database operations.

| Type | Labels |
|------|--------|
| Counter | `storage_type`, `operation`, `result` |

Storage types: `response`, `conversation`, `conversation_item`
Operations: `read`, `write`, `delete`

---

### `smg_db_operation_duration_seconds`

Database operation duration.

| Type | Labels |
|------|--------|
| Histogram | `storage_type`, `operation` |

---

### `smg_db_connections_active`

Active database connections.

| Type | Labels |
|------|--------|
| Gauge | `storage_type` |

---

### `smg_db_items_stored`

Items stored in database.

| Type | Labels |
|------|--------|
| Counter | `storage_type` |

---

## Cache Routing Metrics

### `smg_manual_policy_cache_entries`

Entries in the cache-aware routing cache.

| Type | Labels |
|------|--------|
| Gauge | None |

---

## Dashboard Queries Summary

| Metric | Query |
|--------|-------|
| Request rate | `sum(rate(smg_http_requests_total[5m]))` |
| Error rate | `sum(rate(smg_http_responses_total{status_code=~"5.."}[5m])) / sum(rate(smg_http_responses_total[5m]))` |
| P99 latency | `histogram_quantile(0.99, rate(smg_http_request_duration_seconds_bucket[5m]))` |
| TTFT P50 | `histogram_quantile(0.5, rate(smg_router_ttft_seconds_bucket[5m]))` |
| Tokens/sec | `sum(rate(smg_router_tokens_total[5m]))` |
| Healthy workers | `sum(smg_worker_health)` |
| Open circuits | `count(smg_worker_cb_state == 1)` |
| Rate limit rejections | `rate(smg_http_rate_limit_total{decision="rejected"}[5m])` |
| MCP tool success rate | `sum(rate(smg_mcp_tool_calls_total{result="success"}[5m])) / sum(rate(smg_mcp_tool_calls_total[5m]))` |

---

## Histogram Buckets

Default histogram buckets (20 buckets from 1ms to 240s):

```
0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
10.0, 15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0, 240.0
```

Configure custom buckets via CLI:

```bash
smg --prometheus-duration-buckets 0.01 0.1 0.5 1 5 10
```
