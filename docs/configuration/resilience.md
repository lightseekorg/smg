---
title: Resilience Configuration
---

# Resilience Configuration

This guide covers resilience and fault tolerance configuration for SMG, including retry policies, circuit breakers, health checks, timeouts, and graceful shutdown behavior.

---

## Overview

SMG implements a defense-in-depth approach to resilience with multiple layers of protection.

<div class="grid" markdown>

<div class="card" markdown>

### :material-refresh: Automatic Retries

Retry transient failures with exponential backoff and jitter to avoid overwhelming recovering services.

**Default**: Enabled, 5 retries

</div>

<div class="card" markdown>

### :material-electric-switch: Circuit Breaker

Automatically isolate unhealthy workers to prevent cascade failures across your inference fleet.

**Default**: 10 failures to trip

</div>

<div class="card" markdown>

### :material-heart-pulse: Health Checks

Continuous background monitoring removes unhealthy workers before they cause request failures.

**Default**: 60s interval

</div>

<div class="card" markdown>

### :material-timer-outline: Timeouts

Bound request and queue wait times to maintain system responsiveness under load.

**Default**: 30min request, 60s queue

</div>

</div>

### How They Work Together

1. **Request arrives** - Timeout clock starts
2. **Worker selection** - Only workers passing health checks and circuit breaker checks are considered
3. **Request sent** - If failure occurs and is retryable, retry with exponential backoff
4. **Circuit breaker updates** - Success/failure recorded; circuit opens if threshold exceeded
5. **Health checks continue** - Background monitoring marks workers healthy/unhealthy
6. **Shutdown signal** - Grace period allows in-flight requests to complete

---

## Circuit Breaker

The circuit breaker prevents cascade failures by temporarily stopping requests to workers that are experiencing repeated failures.

<div class="architecture-diagram">
  <img src="../../assets/images/circuit-breaker.svg" alt="Circuit Breaker State Machine">
</div>

<div class="grid" markdown>

<div class="card" markdown>

### :material-check-circle: Closed State

**Normal operation** - requests flow through.

- Failures increment counter
- Success resets counter to zero
- Opens when failures ≥ threshold

</div>

<div class="card" markdown>

### :material-close-circle: Open State

**Circuit tripped** - requests rejected immediately.

- Worker isolated from pool
- No traffic sent to worker
- Transitions to half-open after timeout

</div>

<div class="card" markdown>

### :material-help-circle: Half-Open State

**Testing recovery** - limited probe requests allowed.

- Success → close circuit
- Any failure → reopen circuit
- Gradual traffic restoration

</div>

</div>

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--cb-failure-threshold` | `10` | Consecutive failures before circuit opens |
| `--cb-success-threshold` | `3` | Successes in half-open state to close circuit |
| `--cb-timeout-duration-secs` | `60` | Seconds before open circuit transitions to half-open |
| `--cb-window-duration-secs` | `120` | Sliding window for counting failures |
| `--disable-circuit-breaker` | `false` | Disable circuit breaker entirely |

### Configuration Examples

<div class="grid" markdown>

<div class="card" markdown>

**Fast Circuit Opening**

Sensitive to failures - isolate quickly.

```bash
smg \
  --cb-failure-threshold 3 \
  --cb-timeout-duration-secs 30
```

</div>

<div class="card" markdown>

**Tolerant Configuration**

Allow occasional failures before tripping.

```bash
smg \
  --cb-failure-threshold 20 \
  --cb-success-threshold 5 \
  --cb-timeout-duration-secs 120
```

</div>

</div>

---

## Retry Configuration

Automatic retries protect against transient failures such as network timeouts, temporary overload (429), and intermittent server errors.

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--retry-max-retries` | `5` | Maximum number of retry attempts |
| `--retry-initial-backoff-ms` | `50` | Initial delay before first retry (milliseconds) |
| `--retry-max-backoff-ms` | `30000` | Maximum backoff delay (milliseconds) |
| `--retry-backoff-multiplier` | `1.5` | Multiplier applied to delay after each retry |
| `--retry-jitter-factor` | `0.2` | Random jitter factor (0.0-1.0) to prevent thundering herd |
| `--disable-retries` | `false` | Disable automatic retries entirely |

### Exponential Backoff with Jitter

SMG uses exponential backoff with jitter to space out retry attempts:

```
delay = initial_backoff_ms * (backoff_multiplier ^ attempt)
delay = min(delay, max_backoff_ms)
delay = delay * (1 + random(-jitter_factor, +jitter_factor))
```

**Example progression** (with defaults, no jitter):

| Attempt | Calculated Delay |
|---------|------------------|
| 1 | 50ms |
| 2 | 75ms |
| 3 | 112ms |
| 4 | 168ms |
| 5 | 253ms |

!!! note "Zero-based indexing"
    The `attempt` variable in the formula uses 0-based indexing internally. Attempt 1 in the table corresponds to `attempt=0` in the calculation, attempt 2 corresponds to `attempt=1`, and so on.

### Retryable Status Codes

SMG automatically retries requests that fail with:

- `408` - Request Timeout
- `429` - Too Many Requests
- `500` - Internal Server Error
- `502` - Bad Gateway
- `503` - Service Unavailable
- `504` - Gateway Timeout

### Configuration Examples

<div class="grid" markdown>

<div class="card" markdown>

**Latency-Sensitive**

Minimal retries for interactive applications.

```bash
smg \
  --retry-max-retries 2 \
  --retry-initial-backoff-ms 10 \
  --retry-max-backoff-ms 100
```

</div>

<div class="card" markdown>

**Batch Processing**

Aggressive retries for offline workloads.

```bash
smg \
  --retry-max-retries 10 \
  --retry-initial-backoff-ms 100 \
  --retry-max-backoff-ms 60000 \
  --retry-backoff-multiplier 2.0
```

</div>

</div>

---

## Health Checks

Background health checks continuously monitor worker availability, removing unhealthy workers from the selection pool before they can cause request failures.

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--health-failure-threshold` | `3` | Consecutive failures before marking unhealthy |
| `--health-success-threshold` | `2` | Consecutive successes to mark healthy again |
| `--health-check-timeout-secs` | `5` | Timeout for each health check request |
| `--health-check-interval-secs` | `60` | Interval between health checks |
| `--health-check-endpoint` | `/health` | Endpoint path for health checks |
| `--disable-health-check` | `false` | Disable background health checks |

### Configuration Examples

<div class="grid" markdown>

<div class="card" markdown>

**Fast Detection**

Sensitive to failures - quick marking.

```bash
smg \
  --health-check-interval-secs 10 \
  --health-failure-threshold 2 \
  --health-check-timeout-secs 3
```

</div>

<div class="card" markdown>

**Conservative Detection**

Tolerant of network blips.

```bash
smg \
  --health-check-interval-secs 120 \
  --health-failure-threshold 5 \
  --health-success-threshold 3
```

</div>

</div>

---

## Timeouts

Timeouts prevent requests from waiting indefinitely and help maintain system responsiveness under load.

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--request-timeout-secs` | `1800` (30 min) | Maximum time for a request to complete |
| `--queue-timeout-secs` | `60` | Maximum time a request waits in queue |
| `--worker-startup-timeout-secs` | `1800` (30 min) | Timeout for worker startup/model loading |

### Request Timeout

The request timeout bounds the total time from request arrival to response completion. Set high by default (30 minutes) to accommodate long-running LLM inference:

```bash
# Shorter timeout for interactive applications
smg --request-timeout-secs 120

# Longer timeout for batch processing
smg --request-timeout-secs 3600
```

### Queue Timeout

When concurrency limiting is enabled (`--max-concurrent-requests`), requests exceeding the limit are queued. The queue timeout prevents indefinite waiting.

!!! note "Concurrency vs. rate limiting"
    Setting `--max-concurrent-requests` alone enables **concurrency limiting** (bounds simultaneous requests). To enable **rate limiting** (bounds requests per second using a token bucket), you must explicitly set `--rate-limit-tokens-per-second`.

```bash
smg \
  --max-concurrent-requests 100 \
  --queue-size 200 \
  --queue-timeout-secs 30
```

If a request waits longer than `queue-timeout-secs`, it receives a `429 Too Many Requests` response.

---

## Graceful Shutdown

Graceful shutdown allows in-flight requests to complete before the gateway terminates, preventing request failures during deployments and restarts.

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--shutdown-grace-period-secs` | `180` (3 min) | Time to wait for in-flight requests |

### How It Works

1. **Shutdown signal received** (SIGTERM, SIGINT, or API call)
2. **Stop accepting new requests** - New connections are rejected
3. **Drain in-flight requests** - Existing requests continue processing
4. **Grace period timer starts** - After `shutdown-grace-period-secs`, force shutdown
5. **Clean exit** - Once all requests complete (or grace period expires)

### Triggering Graceful Shutdown

```bash
# Via signal
kill -TERM <pid>

# Via API
curl -X POST http://gateway:3001/ha/shutdown
```

---

## Production Configurations

<div class="grid" markdown>

<div class="card" markdown>

### :material-server-network: High-Availability

Multiple workers with balanced resilience.

```bash
smg \
  --worker-urls http://w1:8000,http://w2:8000,http://w3:8000 \
  --retry-max-retries 3 \
  --retry-initial-backoff-ms 100 \
  --cb-failure-threshold 5 \
  --cb-timeout-duration-secs 30 \
  --health-check-interval-secs 30 \
  --shutdown-grace-period-secs 180
```

</div>

<div class="card" markdown>

### :material-lightning-bolt: Latency-Sensitive

Minimal retries for interactive use.

```bash
smg \
  --worker-urls http://worker:8000 \
  --retry-max-retries 2 \
  --retry-initial-backoff-ms 10 \
  --cb-failure-threshold 3 \
  --cb-timeout-duration-secs 15 \
  --health-check-interval-secs 10 \
  --request-timeout-secs 60
```

</div>

<div class="card" markdown>

### :material-cog: Batch Processing

Aggressive retries for offline workloads.

```bash
smg \
  --worker-urls http://worker:8000 \
  --retry-max-retries 10 \
  --retry-initial-backoff-ms 500 \
  --retry-max-backoff-ms 60000 \
  --cb-failure-threshold 20 \
  --request-timeout-secs 7200 \
  --queue-timeout-secs 300 \
  --shutdown-grace-period-secs 600
```

</div>

</div>

---

## Observability

SMG exposes metrics for monitoring resilience mechanisms:

### Retry Metrics

| Metric | Description |
|--------|-------------|
| `smg_retry_attempts_total` | Total retry attempts by status |
| `smg_retry_backoff_seconds` | Histogram of backoff delays |

### Circuit Breaker Metrics

| Metric | Description |
|--------|-------------|
| `smg_circuit_breaker_state` | Current state per worker (0=closed, 1=open, 2=half-open) |
| `smg_circuit_breaker_transitions_total` | State transitions by worker and direction |
| `smg_circuit_breaker_consecutive_failures` | Current failure count per worker |
| `smg_circuit_breaker_consecutive_successes` | Current success count per worker |

### Health Check Metrics

| Metric | Description |
|--------|-------------|
| `smg_health_check_total` | Health check results by worker and status |
| `smg_worker_health_status` | Current health status per worker |

### Timeout Metrics

| Metric | Description |
|--------|-------------|
| `smg_request_duration_seconds` | Request duration histogram |
| `smg_queue_wait_seconds` | Queue wait time histogram |
| `smg_queue_timeout_total` | Requests that timed out in queue |

---

## Tuning Guidelines

| Symptom | Potential Adjustment |
|---------|---------------------|
| Excessive latency from retries | Reduce `--retry-max-retries`, decrease backoff times |
| Workers marked unhealthy too quickly | Increase `--health-failure-threshold`, `--cb-failure-threshold` |
| Slow failure detection | Decrease `--health-check-interval-secs`, `--cb-timeout-duration-secs` |
| Request timeouts during load | Increase `--request-timeout-secs`, `--queue-timeout-secs` |
| Cascade failures | Lower `--cb-failure-threshold` for faster isolation |
| Requests fail during deployment | Increase `--shutdown-grace-period-secs` |
