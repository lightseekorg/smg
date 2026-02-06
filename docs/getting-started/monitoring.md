---
title: Monitoring
---

# Monitoring

This guide covers basic Prometheus monitoring for SMG without Kubernetes. You'll enable metrics, learn which ones matter, and optionally connect Grafana.

<div class="prerequisites" markdown>

#### Before you begin

- Completed the [Getting Started](index.md) guide
- (Optional) [Prometheus](https://prometheus.io/download/) for scraping
- (Optional) [Grafana](https://grafana.com/grafana/download) for dashboards

</div>

---

## Enable Metrics

Add `--prometheus-port` when starting SMG:

```bash
smg \
  --worker-urls http://worker:8000 \
  --prometheus-port 29000
```

Verify the endpoint:

```bash
curl http://localhost:29000/metrics
```

You should see Prometheus-formatted metrics:

```
# HELP smg_http_requests_total Total HTTP requests
# TYPE smg_http_requests_total counter
smg_http_requests_total{method="POST",path="/v1/chat/completions"} 42
```

---

## Key Metrics

### Requests and Latency

| Metric | Type | What it tells you |
|--------|------|-------------------|
| `smg_http_requests_total` | Counter | Total requests by method and path |
| `smg_http_request_duration_seconds` | Histogram | End-to-end request latency |
| `smg_http_responses_total` | Counter | Responses by status code |

### Workers

| Metric | Type | What it tells you |
|--------|------|-------------------|
| `smg_worker_health` | Gauge | Worker health (1 = healthy, 0 = unhealthy) |
| `smg_worker_requests_active` | Gauge | In-flight requests per worker |
| `smg_worker_cb_state` | Gauge | Circuit breaker state (0 = closed, 1 = open) |

### LLM-Specific (gRPC mode)

| Metric | Type | What it tells you |
|--------|------|-------------------|
| `smg_router_ttft_seconds` | Histogram | Time to first token |
| `smg_router_tpot_seconds` | Histogram | Time per output token |
| `smg_router_tokens_total` | Counter | Tokens processed (input/output) |

For the complete metrics list, see the [Metrics Reference](../reference/metrics.md).

---

## Useful PromQL Queries

### Request rate

```promql
sum(rate(smg_http_requests_total[5m]))
```

### P99 latency

```promql
histogram_quantile(0.99, rate(smg_http_request_duration_seconds_bucket[5m]))
```

### Error rate

```promql
sum(rate(smg_http_responses_total{status_code=~"5.."}[5m]))
/ sum(rate(smg_http_responses_total[5m]))
```

### Healthy workers

```promql
sum(smg_worker_health)
```

### Tokens per second

```promql
sum(rate(smg_router_tokens_total[5m]))
```

---

## Set Up Prometheus Scraping

Create a minimal `prometheus.yml`:

```yaml title="prometheus.yml"
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'smg'
    static_configs:
      - targets: ['localhost:29000']
    metrics_path: /metrics
```

Start Prometheus:

```bash
prometheus --config.file=prometheus.yml
```

Verify the target is being scraped at `http://localhost:9090/targets`.

---

## Optional: Grafana Dashboard

1. Start Grafana (default port 3000):

    ```bash
    grafana-server
    ```

2. Add Prometheus as a data source at `http://localhost:9090`.

3. Create a dashboard with these panels:

    | Panel | Query |
    |-------|-------|
    | Request Rate | `sum(rate(smg_http_requests_total[5m]))` |
    | P99 Latency | `histogram_quantile(0.99, rate(smg_http_request_duration_seconds_bucket[5m]))` |
    | Error Rate | `sum(rate(smg_http_responses_total{status_code=~"5.."}[5m])) / sum(rate(smg_http_responses_total[5m]))` |
    | Worker Health | `sum(smg_worker_health)` |
    | TTFT (P50) | `histogram_quantile(0.5, rate(smg_router_ttft_seconds_bucket[5m]))` |
    | Tokens/sec | `sum(rate(smg_router_tokens_total[5m]))` |

---

## Next Steps

- [Metrics Reference](../reference/metrics.md) — Complete list of all SMG metrics
- [Monitor with Prometheus](../tasks/operations/monitoring.md) — Full monitoring setup including alerting rules, OpenTelemetry tracing, and Kubernetes ServiceMonitor
- [Configure Logging](../tasks/operations/logging.md) — Structured log aggregation
