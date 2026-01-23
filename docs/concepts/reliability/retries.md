---
title: Retries
---

# Retries

SMG implements automatic retries with exponential backoff to handle transient failures gracefully without overwhelming recovering services.

---

## Overview

<div class="grid" markdown>

<div class="card" markdown>

### :material-refresh: Automatic Retries

Transparently retry failed requests to different workers without client intervention.

</div>

<div class="card" markdown>

### :material-chart-timeline: Exponential Backoff

Space out retry attempts with increasing delays to give services time to recover.

</div>

<div class="card" markdown>

### :material-shuffle-variant: Jitter

Add randomness to backoff timing to prevent thundering herd problems.

</div>

<div class="card" markdown>

### :material-filter: Smart Selection

Only retry on transient error codes that are likely to succeed on retry.

</div>

</div>

---

## Why Retries?

Transient failures are common in distributed systems:

- **Network timeouts**: Temporary network congestion or packet loss
- **Worker overload**: Temporary capacity limits (429 responses)
- **Intermittent errors**: Brief service interruptions during deployments
- **Connection issues**: Worker restart or network partition

Without retries, every transient failure becomes a client-visible error. With retries, SMG handles these automatically.

---

## Exponential Backoff with Jitter

SMG uses exponential backoff with jitter to space out retry attempts:

```
delay = initial_backoff_ms * (backoff_multiplier ^ attempt)
delay = min(delay, max_backoff_ms)
delay = delay * (1 + random(-jitter_factor, +jitter_factor))
```

### Example Progression

With default settings (no jitter for clarity):

| Attempt | Calculated Delay |
|---------|------------------|
| 1 | 50ms |
| 2 | 75ms |
| 3 | 112ms |
| 4 | 168ms |
| 5 | 253ms |

!!! note "Zero-based indexing"
    The `attempt` variable uses 0-based indexing internally. Attempt 1 in the table corresponds to `attempt=0` in the calculation.

### Why Jitter?

Without jitter, if multiple requests fail simultaneously, they all retry at exactly the same time—potentially overwhelming the recovering service. Jitter spreads out retries randomly to prevent this "thundering herd" problem.

---

## Retryable Status Codes

SMG automatically retries requests that fail with these status codes:

| Code | Meaning | Why Retryable |
|------|---------|---------------|
| `408` | Request Timeout | Temporary network issue |
| `429` | Too Many Requests | Worker temporarily overloaded |
| `500` | Internal Server Error | Transient server issue |
| `502` | Bad Gateway | Upstream temporarily unavailable |
| `503` | Service Unavailable | Service temporarily down |
| `504` | Gateway Timeout | Upstream timeout |

Requests with other status codes (e.g., 400 Bad Request, 401 Unauthorized) are **not retried** because they would likely fail again.

---

## Configuration

```bash
smg \
  --worker-urls http://w1:8000 http://w2:8000 \
  --retry-max-retries 5 \
  --retry-initial-backoff-ms 50 \
  --retry-max-backoff-ms 30000 \
  --retry-backoff-multiplier 1.5 \
  --retry-jitter-factor 0.2
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--retry-max-retries` | `5` | Maximum number of retry attempts |
| `--retry-initial-backoff-ms` | `50` | Initial delay before first retry (milliseconds) |
| `--retry-max-backoff-ms` | `30000` | Maximum backoff delay (milliseconds) |
| `--retry-backoff-multiplier` | `1.5` | Multiplier applied to delay after each retry |
| `--retry-jitter-factor` | `0.2` | Random jitter factor (0.0-1.0) to prevent thundering herd |
| `--disable-retries` | `false` | Disable automatic retries entirely |

---

## Recommended Configurations

<div class="grid" markdown>

<div class="card" markdown>

### :material-lightning-bolt: Latency-Sensitive

Minimal retries for interactive applications.

```bash
smg \
  --retry-max-retries 2 \
  --retry-initial-backoff-ms 10 \
  --retry-max-backoff-ms 100
```

**Use when**: Real-time chat, interactive UIs

</div>

<div class="card" markdown>

### :material-server-network: High-Availability

Balanced retries for production workloads.

```bash
smg \
  --retry-max-retries 3 \
  --retry-initial-backoff-ms 100 \
  --retry-backoff-multiplier 2.0
```

**Use when**: Production APIs, multi-worker deployments

</div>

<div class="card" markdown>

### :material-cog: Batch Processing

Aggressive retries for offline workloads.

```bash
smg \
  --retry-max-retries 10 \
  --retry-initial-backoff-ms 100 \
  --retry-max-backoff-ms 60000 \
  --retry-backoff-multiplier 2.0
```

**Use when**: Batch inference, non-interactive pipelines

</div>

<div class="card" markdown>

### :material-close-circle: No Retries

Disable retries entirely.

```bash
smg --disable-retries
```

**Use when**: Client handles retries, testing failure scenarios

</div>

</div>

---

## Interaction with Circuit Breakers

Retries and circuit breakers work together:

| Circuit State | Retry Behavior |
|---------------|----------------|
| **Closed** | Normal retries to the worker |
| **Open** | Worker skipped; retry goes to different worker |
| **Half-Open** | Limited test requests; failures don't count against retry budget |

When a circuit is **open**:

- Requests are rejected immediately (no retry to that worker)
- If other healthy workers exist, the retry goes to them
- If all circuits are open, the request fails

---

## Monitoring

### Metrics

| Metric | Description |
|--------|-------------|
| `smg_retry_attempts_total` | Total retry attempts by status |
| `smg_retry_backoff_seconds` | Histogram of backoff delays |

### Useful PromQL Queries

<div class="grid" markdown>

<div class="card" markdown>

#### Retry Rate

```promql
# Retries per second
rate(smg_retry_attempts_total[5m])

# Retry success rate
rate(smg_retry_attempts_total{status="success"}[5m]) /
rate(smg_retry_attempts_total[5m])
```

</div>

<div class="card" markdown>

#### Backoff Distribution

```promql
# Average backoff delay
rate(smg_retry_backoff_seconds_sum[5m]) /
rate(smg_retry_backoff_seconds_count[5m])

# 99th percentile backoff
histogram_quantile(0.99, smg_retry_backoff_seconds_bucket)
```

</div>

</div>

### Alert Thresholds

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| Retry rate | >10/sec | >50/sec | Investigate worker health |
| Retry success rate | <80% | <50% | Check for persistent failures |
| Avg backoff | >5s | >15s | Workers may be overloaded |

---

## Tuning Guidelines

| Symptom | Potential Adjustment |
|---------|---------------------|
| Excessive latency from retries | Reduce `--retry-max-retries`, decrease backoff times |
| Thundering herd on recovery | Increase `--retry-jitter-factor` |
| Retries exhausted too quickly | Increase `--retry-max-retries`, `--retry-max-backoff-ms` |
| Clients seeing too many errors | Increase retry count, check worker health |

---

## What's Next?

<div class="grid" markdown>

<div class="card" markdown>

### :material-electric-switch: Circuit Breakers

Isolate failing workers to prevent cascade failures.

[Circuit Breakers →](circuit-breakers.md)

</div>

<div class="card" markdown>

### :material-heart-pulse: Health Checks

Proactive worker monitoring and failure detection.

[Health Checks →](health-checks.md)

</div>

<div class="card" markdown>

### :material-traffic-light: Rate Limiting

Protect workers from overload with token bucket rate limiting.

[Rate Limiting →](rate-limiting.md)

</div>

</div>
