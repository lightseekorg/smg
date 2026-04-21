---
title: Rate Limiting
---

# Rate Limiting

Rate limiting protects workers from being overwhelmed by too many concurrent requests. SMG uses a token bucket algorithm with optional request queuing.

---

## Overview

<div class="grid" markdown>

<div class="card" markdown>

### :material-bucket: Token Bucket

Smooth rate limiting with burst capacity using the token bucket algorithm.

</div>

<div class="card" markdown>

### :material-tray-full: Request Queuing

Queue excess requests instead of rejecting them immediately.

</div>

<div class="card" markdown>

### :material-timer-outline: Configurable Timeouts

Bound request and queue wait times to maintain system responsiveness.

</div>

<div class="card" markdown>

### :material-chart-line: Observable

Full Prometheus metrics for queue depth, wait times, and rejection rates.

</div>

</div>

---

## Why Rate Limit?

Without rate limiting:

1. **Worker overload**: Too many concurrent requests degrade performance
2. **Memory exhaustion**: Workers run out of GPU memory
3. **Cascading timeouts**: Slow responses cause client timeouts
4. **Poor user experience**: Some users get fast responses, others wait forever

Rate limiting ensures **fair access** and **predictable performance**.

---

## How It Works

SMG uses a **token bucket** algorithm:

<div class="architecture-diagram">
  <img src="../../../assets/images/rate-limiting.svg" alt="Token Bucket Rate Limiting">
</div>

### Token Bucket

- **Bucket capacity**: Maximum concurrent requests (`--max-concurrent-requests`)
- **Refill rate**: Tokens added per second (`--rate-limit-tokens-per-second`)
- **Request cost**: Each request consumes one token

### Request Queue

When no tokens are available, requests can wait in a queue:

- **Queue size**: Maximum waiting requests (`--queue-size`)
- **Queue timeout**: Maximum wait time (`--queue-timeout-secs`)

---

## Configuration

```bash
smg \
  --worker-urls http://w1:8000 http://w2:8000 \
  --max-concurrent-requests 100 \
  --rate-limit-tokens-per-second 50 \
  --queue-size 200 \
  --queue-timeout-secs 30
```

### Rate Limit Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-concurrent-requests` | `-1` (disabled) | Token bucket capacity. When `<= 0` the limiter is disabled entirely and requests pass through. |
| `--rate-limit-tokens-per-second` | unset (refills at `max_concurrent_requests`) | Token bucket refill rate in tokens per second. |
| `--queue-size` | `100` | Maximum queued requests |
| `--queue-timeout-secs` | `60` | Maximum queue wait time |

### Timeout Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--request-timeout-secs` | `1800` (30 min) | Maximum time for a request to complete |
| `--queue-timeout-secs` | `60` | Maximum time a request waits in queue |
| `--worker-startup-timeout-secs` | `1800` (30 min) | Timeout for worker startup/model loading |

!!! note "Concurrency vs. Rate Limiting"
    Setting `--max-concurrent-requests` alone creates a token bucket whose capacity *and* refill rate both equal `max_concurrent_requests`, so it enforces both burst capacity and a sustained rate. Set `--rate-limit-tokens-per-second` when you want the sustained rate to differ from the burst capacity (for example, capacity `100` with refill `50` allows short bursts of 100 while sustaining 50 req/s).

---

## Response Codes

| Code | Meaning | When |
|------|---------|------|
| **429** | Too Many Requests | Queue is full, or queuing is disabled and no token is available |
| **408** | Request Timeout | Queue wait exceeded timeout |

The local rate limiter returns a status-only response with no JSON body (clients should read the HTTP status and `X-Request-Id` to distinguish cases). SMG does not currently emit a `Retry-After` header with the response.

When the mesh global rate limit is enabled and exceeded, the 429 response carries a JSON body:

```json
{
  "error": "Rate limit exceeded",
  "current_count": 123,
  "limit": 100
}
```

---

## Sizing Guidelines

### Concurrent Requests

Base on worker capacity:

```
max_concurrent_requests = num_workers × requests_per_worker
```

| Worker Type | Requests per Worker |
|-------------|---------------------|
| Small GPU (16GB) | 4-8 |
| Medium GPU (40GB) | 8-16 |
| Large GPU (80GB) | 16-32 |

### Queue Size

Base on acceptable latency:

```
queue_size = max_concurrent_requests × queue_depth_factor
```

| Latency Tolerance | Queue Depth Factor |
|-------------------|-------------------|
| Low (interactive) | 0.5-1x |
| Medium (batch) | 2-4x |
| High (async) | 4-8x |

### Token Refill Rate

Base on sustainable throughput:

```
tokens_per_second = expected_requests_per_second × 1.2
```

The 1.2 factor provides headroom for bursts.

---

## Example Configurations

=== "Interactive API"

    Low latency, reject excess traffic:

    ```bash
    smg \
      --max-concurrent-requests 50 \
      --queue-size 25 \
      --queue-timeout-secs 5
    ```

=== "Batch Processing"

    Higher throughput, longer queues:

    ```bash
    smg \
      --max-concurrent-requests 200 \
      --queue-size 500 \
      --queue-timeout-secs 60
    ```

=== "No Rate Limiting"

    Trust upstream rate limiting:

    ```bash
    smg \
      --max-concurrent-requests -1
    ```

---

## Monitoring

### Metrics

| Metric | Description |
|--------|-------------|
| `smg_http_rate_limit_total` | Rate limit decisions by result (allowed/rejected) |
| `smg_http_request_duration_seconds` | Request duration histogram |

### Useful PromQL Queries

<div class="grid" markdown>

<div class="card" markdown>

#### Rate Limit Decisions

```promql
# Rate limit decisions per second
rate(smg_http_rate_limit_total[5m])

# By decision type (allowed/rejected)
sum by (result) (
  rate(smg_http_rate_limit_total[5m])
)
```

</div>

<div class="card" markdown>

#### Request Duration

```promql
# 99th percentile request duration
histogram_quantile(0.99,
  rate(smg_http_request_duration_seconds_bucket[5m]))
```

</div>

</div>

### Alert Thresholds

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| Queue utilization | >70% | >90% | Increase queue size or capacity |
| Rejection rate | >5% | >20% | Increase limits or scale workers |
| Avg queue wait | >10s | >30s | Reduce load or increase capacity |
| Queue timeouts | >1/min | >10/min | Investigate bottlenecks |

---

## Client-Side Handling

### Retry Strategy

Clients should implement exponential backoff when receiving 429. SMG does not set a `Retry-After` header today, so clients must compute their own wait:

```python
import time
import requests

def request_with_retry(url, data, max_retries=5):
    for attempt in range(max_retries):
        response = requests.post(url, json=data)

        if response.status_code == 429:
            # SMG does not emit Retry-After; fall back to exponential backoff.
            time.sleep(2 ** attempt)
            continue

        return response

    raise Exception("Max retries exceeded")
```

### Adaptive Rate

Monitor 429 responses and adjust request rate:

```python
class AdaptiveClient:
    def __init__(self, base_rate=10):
        self.rate = base_rate

    def on_success(self):
        self.rate = min(self.rate * 1.1, 100)  # Increase slowly

    def on_rate_limit(self):
        self.rate = self.rate * 0.5  # Decrease quickly
```

---

## What's Next?

<div class="grid" markdown>

<div class="card" markdown>

### :material-electric-switch: Circuit Breakers

Isolate failing workers to prevent cascade failures.

[Circuit Breakers →](circuit-breakers.md)

</div>

<div class="card" markdown>

### :material-refresh: Retries

Automatic retry with exponential backoff for transient failures.

[Retries →](retries.md)

</div>

<div class="card" markdown>

### :material-heart-pulse: Health Checks

Proactive worker monitoring and failure detection.

[Health Checks →](health-checks.md)

</div>

<div class="card" markdown>

### :material-chart-box: Metrics Reference

Complete list of rate limiting metrics.

[Metrics Reference →](../../reference/metrics.md)

</div>

</div>
