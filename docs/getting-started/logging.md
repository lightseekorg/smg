---
title: Configure Logging
---

# Configure Logging

Configure structured logging with multiple output formats and integrate with log aggregation systems.

<div class="prerequisites" markdown>

#### Before you begin

- SMG [installed](../../getting-started/index.md)
- Completed the [Getting Started](index.md) guide
- Log aggregation system (optional): Elasticsearch, Loki, or similar

</div>

---

## Configuration Options

SMG supports flexible logging configuration via CLI flags or environment variables.

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--log-level` | `info` | Log level: debug, info, warn, error |
| `--log-json` | `false` | Output logs as JSON |
| `--log-dir` | None | Directory for log files (enables file logging) |

### Environment Variable

```bash
# Override log level
RUST_LOG=debug smg --worker-urls http://worker:8000

# Per-module logging
RUST_LOG=smg=debug,hyper=warn smg --worker-urls http://worker:8000
```

---

## Log Levels

| Level | Description | Use Case |
|-------|-------------|----------|
| `error` | Error conditions only | Production (minimal) |
| `warn` | Warnings and errors | Production (recommended) |
| `info` | Informational messages | Production (verbose) |
| `debug` | Debug information | Development |

### Set Log Level

```bash
# Via flag
smg --worker-urls http://worker:8000 --log-level debug

# Via environment
RUST_LOG=info smg --worker-urls http://worker:8000
```

---

## Output Formats

### Plain Text (Default)

Human-readable format with optional ANSI colors:

```
2024-01-15 10:30:45 INFO  smg::routing Request routed to worker worker=http://worker1:8000 policy=cache_aware
```

### JSON Format

Machine-readable format for log aggregation:

```bash
smg --worker-urls http://worker:8000 --log-json
```

Output:

```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "level": "INFO",
  "target": "smg::routing",
  "message": "Request routed to worker",
  "fields": {
    "request_id": "abc123",
    "worker": "http://worker1:8000",
    "policy": "cache_aware",
    "latency_ms": 150
  }
}
```

---

## File Logging

Enable persistent log files with automatic daily rotation.

### Enable File Logging

```bash
smg \
  --worker-urls http://worker:8000 \
  --log-dir /var/log/smg \
  --log-level info
```

### Features

- **Daily rotation**: New file created each day
- **Non-blocking I/O**: Logging doesn't block request handling
- **File naming**: `smg.YYYY-MM-DD.log`

### Example Directory Structure

```
/var/log/smg/
├── smg.2024-01-13.log
├── smg.2024-01-14.log
└── smg.2024-01-15.log
```

---

## Docker Logging

### Basic Configuration

```yaml title="docker-compose.yml"
services:
  smg:
    image: smg:latest
    environment:
      - RUST_LOG=info
    command: >
      --worker-urls http://worker:8000
      --log-json
    logging:
      driver: json-file
      options:
        max-size: "100m"
        max-file: "3"
```

### Production Configuration

```yaml title="docker-compose.yml"
services:
  smg:
    image: smg:latest
    command: >
      --worker-urls http://worker:8000
      --log-level info
      --log-json
      --log-dir /var/log/smg
    volumes:
      - smg-logs:/var/log/smg
    logging:
      driver: json-file
      options:
        max-size: "50m"
        max-file: "5"

volumes:
  smg-logs:
```

---

## Kubernetes Logging

### Basic Pod Logging

```bash
# Follow logs
kubectl logs -n inference -l app=smg -f

# Previous container logs
kubectl logs -n inference -l app=smg --previous

# All containers in pod
kubectl logs -n inference <pod-name> --all-containers
```

### Deployment Configuration

```yaml title="smg-deployment.yaml"
apiVersion: apps/v1
kind: Deployment
metadata:
  name: smg
spec:
  template:
    spec:
      containers:
        - name: smg
          args:
            - --worker-urls=http://worker:8000
            - --log-level=info
            - --log-json
          env:
            - name: RUST_LOG
              value: "smg=info"
```

---

## Log Aggregation

### Grafana Loki

```yaml title="promtail-config.yaml"
server:
  http_listen_port: 9080

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: smg
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        regex: smg
        action: keep
      - source_labels: [__meta_kubernetes_namespace]
        target_label: namespace
      - source_labels: [__meta_kubernetes_pod_name]
        target_label: pod
    pipeline_stages:
      - json:
          expressions:
            level: level
            target: target
            request_id: fields.request_id
      - labels:
          level:
          target:
```

### Elasticsearch + Fluentd

```yaml title="fluentd-config.yaml"
<source>
  @type tail
  path /var/log/containers/smg*.log
  pos_file /var/log/fluentd-smg.pos
  tag smg.*
  <parse>
    @type json
    time_key timestamp
    time_format %Y-%m-%dT%H:%M:%S.%NZ
  </parse>
</source>

<filter smg.**>
  @type parser
  key_name log
  <parse>
    @type json
  </parse>
</filter>

<match smg.**>
  @type elasticsearch
  host elasticsearch
  port 9200
  index_name smg-logs
</match>
```

---

## Log Queries

### Loki/LogQL

```logql
# All SMG logs
{app="smg"}

# Error logs only
{app="smg"} | json | level="ERROR"

# Specific request
{app="smg"} | json | request_id="abc123"

# Routing decisions
{app="smg"} |= "routed" | json

# By module
{app="smg"} | json | target=~"smg::routing.*"
```

### Elasticsearch/Kibana

```json
{
  "query": {
    "bool": {
      "must": [
        { "match": { "target": "smg::routing" } },
        { "match": { "level": "ERROR" } }
      ],
      "filter": [
        { "range": { "timestamp": { "gte": "now-1h" } } }
      ]
    }
  }
}
```

---

## OpenTelemetry Integration

When OpenTelemetry is enabled, log levels are automatically adjusted:

| OTEL Enabled | Event Level | Behavior |
|--------------|-------------|----------|
| Yes | INFO | Exported to OTLP collector |
| No | DEBUG | Not exported (local only) |

This ensures request events are captured in traces when OTEL is active.

```bash
# Enable OTEL with logging
smg \
  --worker-urls http://worker:8000 \
  --enable-trace \
  --otlp-traces-endpoint localhost:4317 \
  --log-level info
```

---

## Request Correlation

### Request ID Propagation

SMG generates unique request IDs and propagates them through logs:

```bash
# Send request with custom ID
curl -H "X-Request-ID: my-trace-123" \
  http://localhost:30000/v1/chat/completions \
  -d '{"model": "llama", "messages": [{"role": "user", "content": "Hi"}]}'
```

### Trace a Request

```bash
# Find all logs for a request
kubectl logs -n inference -l app=smg | grep "my-trace-123"

# In Loki
{app="smg"} | json | request_id="my-trace-123"
```

---

## Log Rotation

### Linux logrotate

```conf title="/etc/logrotate.d/smg"
/var/log/smg/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0640 smg smg
}
```

### Docker

```yaml
logging:
  driver: json-file
  options:
    max-size: "100m"
    max-file: "5"
```

---

## Verification

```bash
# Check log output
smg --worker-urls http://worker:8000 --log-level debug 2>&1 | head -20

# Verify JSON format
smg --worker-urls http://worker:8000 --log-json 2>&1 | jq .

# Test log level filtering
smg --worker-urls http://worker:8000 --log-level warn 2>&1 | grep -c INFO
# Should output: 0

# Check file logging
smg --worker-urls http://worker:8000 --log-dir /tmp/smg-logs &
ls -la /tmp/smg-logs/
```

---

## Troubleshooting

??? question "No logs appearing"

    1. Check log level is not too restrictive:
    ```bash
    smg --log-level debug ...
    ```

    2. Verify logs are going to stderr:
    ```bash
    smg --worker-urls http://worker:8000 2>&1 | head
    ```

    3. Check RUST_LOG isn't overriding:
    ```bash
    unset RUST_LOG
    smg --log-level info ...
    ```

??? question "Logs not in JSON format"

    1. Ensure `--log-json` flag is set:
    ```bash
    smg --log-json --worker-urls http://worker:8000
    ```

    2. Verify you're reading stderr, not stdout

??? question "File logs not appearing"

    1. Check directory exists and is writable:
    ```bash
    mkdir -p /var/log/smg
    chmod 755 /var/log/smg
    ```

    2. Verify `--log-dir` flag is set correctly

??? question "Log aggregator not receiving logs"

    1. Verify JSON format is enabled
    2. Check network connectivity to aggregator
    3. Verify log format matches parser expectations

---

## What's Next?

- [Monitoring](monitoring.md) — Metrics, tracing, and alerting
- [Configuration Reference](../reference/configuration.md) — Full CLI options
