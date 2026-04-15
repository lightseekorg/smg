---
title: Graceful Shutdown
---

# Graceful Shutdown

Graceful shutdown allows in-flight requests to complete before the gateway terminates, preventing request failures during deployments and restarts.

---

## Overview

<div class="grid" markdown>

<div class="card" markdown>

### :material-power: Clean Termination

Allow existing requests to finish rather than abruptly closing connections.

</div>

<div class="card" markdown>

### :material-rocket-launch: Zero-Downtime Deployments

Deploy updates without causing client-visible errors.

</div>

<div class="card" markdown>

### :material-timer: Configurable Grace Period

Control how long to wait for in-flight requests.

</div>

<div class="card" markdown>

### :material-api: API Control

Trigger shutdown programmatically via HTTP API.

</div>

</div>

---

## Why Graceful Shutdown?

Without graceful shutdown:

- **Abrupt termination**: Active requests are immediately disconnected
- **Client errors**: In-flight requests return connection errors
- **Data loss**: Streaming responses may be truncated
- **Deployment failures**: Rolling updates cause visible errors

With graceful shutdown:

- **Request completion**: Active requests finish normally
- **No client errors**: Users don't see deployment-related failures
- **Clean streaming**: Streaming responses complete before shutdown
- **Smooth deployments**: Zero-downtime rolling updates

---

## How It Works

<div class="architecture-diagram">
  <img src="../../../assets/images/graceful-shutdown.svg" alt="Graceful Shutdown Sequence">
</div>

### Shutdown Sequence

1. **Shutdown signal received** (SIGTERM or SIGINT). The mesh-only `/ha/shutdown` API triggers a separate mesh-level broadcast path and is not part of this signal-driven sequence.
2. **Stop accepting new connections** — `axum_server`'s handle stops the TCP accept loop and marks the in-flight tracker as draining; new connections are refused at the socket level rather than receiving a 503 response.
3. **Drain in-flight requests** — existing requests continue processing while the server waits on the in-flight tracker.
4. **Grace period timer starts** — after `--shutdown-grace-period-secs`, the drain wait times out and the server forces shutdown with any remaining requests still in-flight.
5. **Clean exit** — once all requests complete (or the grace period expires), background components (MCP orchestrator, etc.) are cleaned up and the process exits.

---

## Configuration

```bash
smg \
  --worker-urls http://w1:8000 http://w2:8000 \
  --shutdown-grace-period-secs 180
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--shutdown-grace-period-secs` | `180` (3 min) | Time to wait for in-flight requests |

---

## Recommended Configurations

<div class="grid" markdown>

<div class="card" markdown>

### :material-lightning-bolt: Fast Shutdown

Quick termination for development.

```bash
smg --shutdown-grace-period-secs 10
```

**Use when**: Development, testing, quick restarts

</div>

<div class="card" markdown>

### :material-server-network: Production Standard

Balanced grace period for typical workloads.

```bash
smg --shutdown-grace-period-secs 180
```

**Use when**: Standard production deployments

</div>

<div class="card" markdown>

### :material-cog: Batch Processing

Long grace period for long-running requests.

```bash
smg --shutdown-grace-period-secs 600
```

**Use when**: Batch inference, long-running generations

</div>

<div class="card" markdown>

### :material-clock-fast: Critical Low-Latency

Minimal grace for latency-sensitive systems.

```bash
smg --shutdown-grace-period-secs 30
```

**Use when**: Very short requests, rapid scaling

</div>

</div>

---

## Triggering Shutdown

### Via Signal

```bash
# Find the SMG process
pgrep -f smg

# Send SIGTERM for graceful shutdown
kill -TERM <pid>

# Or SIGINT (Ctrl+C in terminal)
kill -INT <pid>
```

### Via API

```bash
# Trigger graceful shutdown via HTTP (mesh mode only)
curl -X POST http://gateway:30000/ha/shutdown
```

The `/ha/shutdown` endpoint lives on the main gateway port (default `30000`) and requires mesh mode (`--mesh-*` flags). Without mesh enabled the endpoint returns `503 Service Unavailable`. The mesh handler broadcasts a `LEAVING` status to peer nodes and stops the mesh rate-limit task — it does not share the same in-flight drain path used by the signal handler.

### Kubernetes Integration

Kubernetes sends SIGTERM by default when terminating pods. Configure `terminationGracePeriodSeconds` to match or exceed your SMG grace period:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: smg
spec:
  template:
    spec:
      terminationGracePeriodSeconds: 210  # SMG grace + buffer
      containers:
        - name: smg
          args:
            - --shutdown-grace-period-secs=180
```

!!! warning "Kubernetes timeout"
    Kubernetes will force-kill the pod after `terminationGracePeriodSeconds`. Set this **higher** than `--shutdown-grace-period-secs` to ensure SMG has time to complete its graceful shutdown.

---

## Sizing the Grace Period

Consider these factors when setting the grace period:

| Factor | Impact on Grace Period |
|--------|------------------------|
| **Average request duration** | Grace period should exceed typical request time |
| **Longest expected request** | Batch jobs may need longer grace periods |
| **Streaming responses** | Long streams need extended grace periods |
| **Deployment frequency** | Frequent deployments may need shorter periods |
| **Scaling responsiveness** | Autoscaling may need faster termination |

### Calculation Guidelines

```
grace_period = max(
    avg_request_duration * 3,
    p99_request_duration * 1.5,
    max_streaming_duration
)
```

**Example**: If your average request is 30s, p99 is 60s, and max streaming is 120s:

```
grace_period = max(90, 90, 120) = 120 seconds
```

---

## Integration with Load Balancers

For zero-downtime deployments, coordinate with your load balancer:

### Pre-Stop Hook (Kubernetes)

Remove the pod from the load balancer before shutdown:

```yaml
spec:
  containers:
    - name: smg
      lifecycle:
        preStop:
          exec:
            command: ["/bin/sh", "-c", "sleep 5"]
```

The sleep allows the load balancer to stop sending new traffic before SMG begins its graceful shutdown.

### Health Check Coordination

SMG's `/health` (liveness) endpoint always returns `200 OK` — it does not switch to an unhealthy response during shutdown. To drain traffic cleanly, remove the pod from the load balancer before SMG begins its own drain (for example, with the Kubernetes `preStop` hook above, or an external control-plane deregister step).

```bash
curl http://gateway:30000/health
# Returns 200 OK both during normal operation and throughout the drain
```

`/readiness` can still return `503 Service Unavailable` when no healthy workers remain, but it reacts to worker state rather than to the shutdown signal itself.

---

## Monitoring

### Shutdown Events

Watch logs for shutdown-related messages:

```text
# Signal received
INFO Received Ctrl+C, starting graceful shutdown
# or
INFO Received terminate signal, starting graceful shutdown

# Gate — in-flight tracker is marked draining and the accept loop stops
INFO Beginning graceful shutdown: gating new connections in_flight=5

# Drain completes within the grace period
INFO All in-flight requests drained

# Or the grace period expires with requests still running
WARN Drain timed out, forcing shutdown with requests still in-flight remaining=2 timeout_secs=180

# Component teardown
INFO HTTP server stopped. Starting component cleanup...
INFO Cleanup complete. Process exiting.
```

### Metrics During Shutdown

| Metric | Observation |
|--------|-------------|
| `smg_worker_requests_active` | Should decrease towards 0 |
| `smg_http_requests_total` | New requests should stop |

---

## Tuning Guidelines

| Symptom | Potential Adjustment |
|---------|---------------------|
| Requests failing during deployment | Increase `--shutdown-grace-period-secs` |
| Slow scaling down | Decrease `--shutdown-grace-period-secs` |
| Kubernetes force-killing pods | Increase `terminationGracePeriodSeconds` |
| Streaming responses truncated | Match grace period to max stream duration |

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
