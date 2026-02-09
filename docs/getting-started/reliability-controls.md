---
title: Reliability Controls
---

# Reliability Controls

This guide provides command-first setup for request protection and failure handling: concurrency limits, retries, and circuit breakers.

<div class="prerequisites" markdown>

#### Before you begin

- Completed the [Getting Started](index.md) guide
- Two or more workers recommended for retry failover

</div>

---

## 1. Concurrency and Queue Limits

Start with bounded concurrency and queueing:

```bash
smg \
  --worker-urls http://w1:8000 http://w2:8000 \
  --max-concurrent-requests 100 \
  --queue-size 200 \
  --queue-timeout-secs 30
```

Optional token refill rate:

```bash
smg \
  --worker-urls http://w1:8000 http://w2:8000 \
  --max-concurrent-requests 100 \
  --rate-limit-tokens-per-second 100 \
  --queue-size 200 \
  --queue-timeout-secs 30
```

---

## 2. Retries

Enable retries with explicit backoff settings:

```bash
smg \
  --worker-urls http://w1:8000 http://w2:8000 \
  --retry-max-retries 5 \
  --retry-initial-backoff-ms 50 \
  --retry-max-backoff-ms 30000 \
  --retry-backoff-multiplier 1.5 \
  --retry-jitter-factor 0.2
```

Disable retries when client handles them:

```bash
smg \
  --worker-urls http://w1:8000 http://w2:8000 \
  --disable-retries
```

---

## 3. Circuit Breakers

Protect traffic from repeatedly failing workers:

```bash
smg \
  --worker-urls http://w1:8000 http://w2:8000 \
  --cb-failure-threshold 10 \
  --cb-success-threshold 3 \
  --cb-timeout-duration-secs 60 \
  --cb-window-duration-secs 120
```

Disable only for controlled testing:

```bash
smg \
  --worker-urls http://w1:8000 http://w2:8000 \
  --disable-circuit-breaker
```

---

## Production Baseline

A practical starting profile:

```bash
smg \
  --worker-urls http://w1:8000 http://w2:8000 http://w3:8000 \
  --max-concurrent-requests 150 \
  --queue-size 300 \
  --queue-timeout-secs 30 \
  --retry-max-retries 3 \
  --retry-initial-backoff-ms 50 \
  --retry-max-backoff-ms 5000 \
  --retry-backoff-multiplier 2.0 \
  --retry-jitter-factor 0.2 \
  --cb-failure-threshold 10 \
  --cb-success-threshold 3 \
  --cb-timeout-duration-secs 60 \
  --cb-window-duration-secs 120
```

---

## Verify

```bash
curl http://localhost:30000/health
curl http://localhost:30000/workers
```

With metrics enabled, inspect reliability metrics at `/metrics`.

---

## Next Steps

- [Rate Limiting Concepts](../concepts/reliability/rate-limiting.md)
- [Retries Concepts](../concepts/reliability/retries.md)
- [Circuit Breakers Concepts](../concepts/reliability/circuit-breakers.md)
- [Configuration Reference](../reference/configuration.md#rate-limiting-configuration)
