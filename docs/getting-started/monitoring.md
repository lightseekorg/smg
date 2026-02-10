---
title: Monitoring
---

# Monitoring

Set up Prometheus monitoring, OpenTelemetry tracing, and Grafana dashboards for SMG.

<div class="prerequisites" markdown>

#### Before you begin

- Completed the [Getting Started](index.md) guide
- Prometheus server (or follow steps below to deploy)
- Grafana (optional, for dashboards)
- OTLP collector (optional, for distributed tracing)

</div>

---

## Enable Metrics

SMG exposes Prometheus metrics on a dedicated port with a 6-layer metric hierarchy.

### Start SMG with metrics

```bash
smg \
  --worker-urls http://worker:8000 \
  --prometheus-port 29000 \
  --prometheus-host 0.0.0.0
```

### Verify metrics endpoint

```bash
curl http://localhost:29000/metrics
```

You should see Prometheus-formatted metrics:

```
# HELP smg_http_requests_total Total HTTP requests
# TYPE smg_http_requests_total counter
smg_http_requests_total{method="POST",path="/v1/chat/completions"} 1234
...
```

---

## OpenTelemetry Tracing

SMG supports distributed tracing via OpenTelemetry.

### Enable tracing

```bash
smg \
  --worker-urls http://worker:8000 \
  --enable-trace \
  --otlp-traces-endpoint localhost:4317
```

### Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--enable-trace` | `false` | Enable OpenTelemetry tracing |
| `--otlp-traces-endpoint` | `localhost:4317` | OTLP gRPC collector endpoint |

### Trace propagation

SMG automatically propagates W3C TraceContext headers to workers:

- `traceparent` — Trace ID and span ID
- `tracestate` — Vendor-specific trace data

---

## Prometheus Configuration

### Basic configuration

```yaml title="prometheus.yml"
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'smg'
    static_configs:
      - targets: ['localhost:29000']
    metrics_path: /metrics
```

### Kubernetes ServiceMonitor

For Prometheus Operator:

```yaml title="smg-servicemonitor.yaml"
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: smg
  namespace: inference
  labels:
    app: smg
spec:
  selector:
    matchLabels:
      app: smg
  endpoints:
    - port: metrics
      interval: 15s
      path: /metrics
  namespaceSelector:
    matchNames:
      - inference
```

---

## Key Metrics by Layer

### Layer 1: HTTP Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `smg_http_requests_total` | Counter | Requests by method, path |
| `smg_http_request_duration_seconds` | Histogram | Request latency |
| `smg_http_responses_total` | Counter | Responses by status_code, error_code |
| `smg_http_connections_active` | Gauge | Active connections |
| `smg_http_rate_limit_total` | Counter | Rate limit decisions |

### Layer 2: Router Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `smg_router_requests_total` | Counter | Requests by router_type, model, endpoint |
| `smg_router_ttft_seconds` | Histogram | Time to first token (gRPC) |
| `smg_router_tpot_seconds` | Histogram | Time per output token (gRPC) |
| `smg_router_tokens_total` | Counter | Tokens by type (input/output) |
| `smg_router_stage_duration_seconds` | Histogram | Pipeline stage durations |

### Layer 3: Worker Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `smg_worker_health` | Gauge | Health status (1=healthy, 0=unhealthy) |
| `smg_worker_requests_active` | Gauge | Active requests per worker |
| `smg_worker_cb_state` | Gauge | Circuit breaker state |
| `smg_worker_retries_total` | Counter | Retry attempts |

### Layer 5: MCP Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `smg_mcp_tool_calls_total` | Counter | Tool invocations by tool_name, result |
| `smg_mcp_tool_duration_seconds` | Histogram | Tool execution time |
| `smg_mcp_servers_active` | Gauge | Active MCP servers |

[View all metrics →](../reference/metrics.md)

---

## Grafana Dashboards

### Essential panels

**Request Rate**
```promql
sum(rate(smg_http_requests_total[5m]))
```

**P99 Latency**
```promql
histogram_quantile(0.99, rate(smg_http_request_duration_seconds_bucket[5m]))
```

**Error Rate**
```promql
sum(rate(smg_http_responses_total{status_code=~"5.."}[5m]))
/ sum(rate(smg_http_responses_total[5m]))
```

**Time to First Token (TTFT)**
```promql
histogram_quantile(0.5, rate(smg_router_ttft_seconds_bucket[5m]))
```

**Tokens per Second**
```promql
sum(rate(smg_router_tokens_total[5m]))
```

**Worker Health**
```promql
sum(smg_worker_health)
```

---

## Alerting Rules

```yaml title="smg-alerts.yaml"
groups:
  - name: smg
    rules:
      - alert: SMGHighErrorRate
        expr: |
          sum(rate(smg_http_responses_total{status_code=~"5.."}[5m]))
          / sum(rate(smg_http_responses_total[5m])) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate on SMG"
          description: "Error rate is {{ $value | humanizePercentage }}"

      - alert: SMGWorkerUnhealthy
        expr: smg_worker_health == 0
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "SMG worker unhealthy"
          description: "Worker {{ $labels.worker }} is unhealthy"

      - alert: SMGHighLatency
        expr: |
          histogram_quantile(0.99, rate(smg_http_request_duration_seconds_bucket[5m])) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency on SMG"
          description: "P99 latency is {{ $value }}s"

      - alert: SMGCircuitBreakerOpen
        expr: smg_worker_cb_state == 1
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Circuit breaker open"
          description: "Circuit breaker for {{ $labels.worker }} is open"

      - alert: SMGHighTTFT
        expr: |
          histogram_quantile(0.95, rate(smg_router_ttft_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High time to first token"
          description: "P95 TTFT is {{ $value }}s"

      - alert: SMGRateLimitRejections
        expr: rate(smg_http_rate_limit_total{decision="rejected"}[5m]) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High rate limit rejections"
          description: "{{ $value }} rejections/sec"
```

---

## Useful Queries

### Request analysis

```promql
# Request rate by endpoint
sum by (path) (rate(smg_http_requests_total[5m]))

# Success rate
sum(rate(smg_http_responses_total{status_code="200"}[5m]))
/ sum(rate(smg_http_responses_total[5m]))

# Latency percentiles
histogram_quantile(0.50, rate(smg_http_request_duration_seconds_bucket[5m]))
histogram_quantile(0.95, rate(smg_http_request_duration_seconds_bucket[5m]))
histogram_quantile(0.99, rate(smg_http_request_duration_seconds_bucket[5m]))
```

### LLM performance

```promql
# Tokens per second by model
sum by (model) (rate(smg_router_tokens_total[5m]))

# TTFT by model
histogram_quantile(0.5, sum by (model, le) (rate(smg_router_ttft_seconds_bucket[5m])))

# Input/output token ratio
sum(rate(smg_router_tokens_total{type="output"}[5m]))
/ sum(rate(smg_router_tokens_total{type="input"}[5m]))
```

### Worker analysis

```promql
# Load distribution
smg_worker_requests_active / ignoring(worker) group_left sum(smg_worker_requests_active)

# Unhealthy workers
count(smg_worker_health == 0)

# Circuit breaker states
count by (worker) (smg_worker_cb_state == 1)
```

### MCP tool analysis

```promql
# Tool success rate
sum(rate(smg_mcp_tool_calls_total{result="success"}[5m]))
/ sum(rate(smg_mcp_tool_calls_total[5m]))

# Most used tools
topk(10, sum by (tool_name) (rate(smg_mcp_tool_calls_total[5m])))

# Slowest tools
topk(5, histogram_quantile(0.95, sum by (tool_name, le) (rate(smg_mcp_tool_duration_seconds_bucket[5m]))))
```

---

## Verification

```bash
# Check metrics are being scraped
curl -s http://prometheus:9090/api/v1/targets | jq '.data.activeTargets[] | select(.labels.job=="smg")'

# Query a metric
curl -s 'http://prometheus:9090/api/v1/query?query=smg_http_requests_total' | jq

# Check alerts
curl -s http://prometheus:9090/api/v1/alerts | jq
```

---

## Troubleshooting

??? question "Metrics endpoint not responding"

    1. Verify SMG is running with `--prometheus-port`:
    ```bash
    ps aux | grep smg
    ```

    2. Check the port is listening:
    ```bash
    netstat -tlnp | grep 29000
    ```

    3. Check firewall rules allow access

??? question "Traces not appearing"

    1. Verify OTLP endpoint is reachable:
    ```bash
    curl http://localhost:4317
    ```

    2. Check SMG was started with `--enable-trace`

    3. Verify collector is receiving spans

??? question "Missing metrics"

    1. Ensure the feature generating metrics is enabled

    2. Some metrics only appear for specific router types (e.g., TTFT is gRPC-only)

    3. Verify metric name spelling in queries

---

## What's Next?

- [Configure Logging](logging.md) — Structured log aggregation
- [Configure TLS](tls.md) — Secure client-to-gateway traffic
- [Metrics Reference](../reference/metrics.md) — Complete metrics documentation
