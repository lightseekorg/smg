---
title: Configuration
---

# Configuration Reference

Complete configuration reference for tuning SMG behavior.

---

## Configuration Methods

SMG can be configured through:

1. **Command-line arguments** (highest priority)
2. **Environment variables**
3. **Default values** (lowest priority)

---

## Worker Configuration

### Host

Network interface to bind to.

| Option | `--host` |
|--------|----------|
| Environment | - |
| Default | `0.0.0.0` |

| Value | Description |
|-------|-------------|
| `127.0.0.1` | Localhost only |
| `0.0.0.0` | All IPv4 interfaces |
| `::` | All IPv6 interfaces |
| `::1` | IPv6 localhost |

### Port

Port for the main API server.

| Option | `--port` |
|--------|----------|
| Environment | - |
| Default | `30000` |

### Worker URLs

List of worker URLs to route requests to.

| Option | `--worker-urls` |
|--------|-----------------|
| Environment | - |
| Default | Empty |
| Format | Space-separated URLs |

**Examples**:
```bash
--worker-urls http://worker1:8000 http://worker2:8000
--worker-urls http://[::1]:8000 http://192.168.1.1:8000  # IPv6 and IPv4
--worker-urls grpc://worker1:50051  # gRPC mode
```

---

## Routing Policy Configuration

### Load Balancing Policy

Controls how requests are distributed across workers.

| Option | `--policy` |
|--------|------------|
| Environment | - |
| Default | `cache_aware` |
| Values | `random`, `round_robin`, `cache_aware`, `power_of_two`, `prefix_hash`, `manual` |

**Policy Comparison**:

| Policy | Use Case | KV Cache | Load Balance |
|--------|----------|----------|--------------|
| `random` | Simple deployments | Poor | Fair |
| `round_robin` | Uniform workloads | Poor | Good |
| `power_of_two` | Variable workloads | Poor | Excellent |
| `cache_aware` | LLM inference | Excellent | Good |
| `prefix_hash` | Consistent routing by prefix | Good | Good |
| `manual` | Session affinity | Good | Manual |

**Recommendation**: Use `cache_aware` for LLM workloads to maximize KV cache hit rates.

### Cache-Aware Policy Options

| Option | Description | Default |
|--------|-------------|---------|
| `--cache-threshold` | Cache threshold (0.0-1.0) for cache-aware routing | `0.3` |
| `--balance-abs-threshold` | Absolute threshold for load balancing trigger | `64` |
| `--balance-rel-threshold` | Relative threshold for load balancing trigger | `1.5` |
| `--eviction-interval` | Interval in seconds between cache eviction operations | `120` |
| `--max-tree-size` | Maximum size of the approximation tree | `67108864` |

### Prefix Hash Policy Options

| Option | Description | Default |
|--------|-------------|---------|
| `--prefix-token-count` | Number of prefix tokens to use for hashing | `256` |
| `--prefix-hash-load-factor` | Load factor threshold for rebalancing | `1.25` |

### Manual Policy Options

| Option | Description | Default |
|--------|-------------|---------|
| `--max-idle-secs` | Maximum idle time before eviction | `14400` (4 hours) |
| `--assignment-mode` | Mode for new routing key assignment | `random` |

**Assignment Modes**:
- `random` - Assign to a random worker
- `min_load` - Assign to worker with fewest active requests
- `min_group` - Assign to worker with fewest routing keys

### Advanced Routing Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dp-aware` | Enable data parallelism aware scheduling | `false` |
| `--enable-igw` | Enable IGW (Inference Gateway) mode for multi-model support | `false` |

---

## PD Disaggregation Configuration

Prefill-Decode disaggregated mode separates prefill and decode operations across different workers.

### Enable PD Mode

| Option | `--pd-disaggregation` |
|--------|----------------------|
| Environment | - |
| Default | `false` |

### Prefill Servers

| Option | `--prefill` |
|--------|-------------|
| Format | `URL [BOOTSTRAP_PORT]` |
| Multiple | Yes (specify multiple times) |

**Examples**:
```bash
--prefill http://prefill1:30001 9001 \
--prefill http://prefill2:30002 9002 \
--prefill http://prefill3:30003 none  # No bootstrap port
```

### Decode Servers

| Option | `--decode` |
|--------|------------|
| Format | URL |
| Multiple | Yes (specify multiple times) |

**Example**:
```bash
--decode http://decode1:30003 \
--decode http://decode2:30004
```

### PD-Specific Policies

| Option | Description | Default |
|--------|-------------|---------|
| `--prefill-policy` | Specific policy for prefill nodes | Uses main `--policy` |
| `--decode-policy` | Specific policy for decode nodes | Uses main `--policy` |

### Worker Startup Configuration

| Option | Description | Default |
|--------|-------------|---------|
| `--worker-startup-timeout-secs` | Timeout for worker startup and registration | `1800` (30 min) |
| `--worker-startup-check-interval` | Interval between worker startup checks | `30` |

---

## Service Discovery (Kubernetes)

### Enable Service Discovery

| Option | `--service-discovery` |
|--------|----------------------|
| Environment | - |
| Default | `false` |

Note: Enabling service discovery automatically enables IGW mode.

### Label Selector

| Option | `--selector` |
|--------|--------------|
| Format | `key=value` (space-separated for multiple) |

**Example**:
```bash
--selector app=sglang-worker tier=inference
```

### Namespace

| Option | `--service-discovery-namespace` |
|--------|--------------------------------|
| Environment | - |
| Default | All namespaces |

### Worker Port

| Option | `--service-discovery-port` |
|--------|---------------------------|
| Environment | - |
| Default | `80` |

### PD Service Discovery Selectors

| Option | Description |
|--------|-------------|
| `--prefill-selector` | Label selector for prefill server pods |
| `--decode-selector` | Label selector for decode server pods |

---

## Tokenizer Configuration

### Model Path

| Option | `--model-path` |
|--------|----------------|
| Environment | - |
| Default | None |
| Description | HuggingFace model ID or local path for loading tokenizer |

### Tokenizer Path

| Option | `--tokenizer-path` |
|--------|-------------------|
| Environment | - |
| Default | None |
| Description | Explicit tokenizer path (overrides model_path tokenizer) |

### Chat Template

| Option | `--chat-template` |
|--------|-------------------|
| Environment | - |
| Default | None |
| Description | Path to chat template file |

### Tokenizer Cache (L0 - Exact Match)

| Option | Description | Default |
|--------|-------------|---------|
| `--tokenizer-cache-enable-l0` | Enable L0 exact match cache | `false` |
| `--tokenizer-cache-l0-max-entries` | Maximum entries in L0 cache | `10000` |

### Tokenizer Cache (L1 - Prefix Matching)

| Option | Description | Default |
|--------|-------------|---------|
| `--tokenizer-cache-enable-l1` | Enable L1 prefix matching cache | `false` |
| `--tokenizer-cache-l1-max-memory` | Maximum memory for L1 cache (bytes) | `52428800` (50MB) |

---

## Parser Configuration

### Reasoning Parser

| Option | `--reasoning-parser` |
|--------|---------------------|
| Environment | - |
| Default | None |
| Values | `deepseek-r1`, `qwen3`, etc. |
| Description | Parser for reasoning models with thinking tokens |

### Tool Call Parser

| Option | `--tool-call-parser` |
|--------|---------------------|
| Environment | - |
| Default | None |
| Values | `json`, `qwen`, etc. |
| Description | Parser for tool-call/function-calling interactions |

---

## MCP Configuration

### MCP Config Path

| Option | `--mcp-config-path` |
|--------|---------------------|
| Environment | - |
| Default | None |
| Description | Path to MCP (Model Context Protocol) server configuration file |

---

## Backend Configuration

### Backend Runtime

| Option | `--backend` |
|--------|-------------|
| Environment | - |
| Default | `sglang` |
| Values | `sglang`, `vllm`, `trtllm`, `openai`, `anthropic` |

### History Backend

| Option | `--history-backend` |
|--------|---------------------|
| Environment | - |
| Default | `memory` |
| Values | `memory`, `none`, `oracle`, `postgres`, `redis` |

---

## Storage Configuration

### Oracle Database

| Option | Environment | Description |
|--------|-------------|-------------|
| `--oracle-wallet-path` | `ATP_WALLET_PATH` | Path to Oracle ATP wallet directory |
| `--oracle-tns-alias` | `ATP_TNS_ALIAS` | Oracle TNS alias from tnsnames.ora |
| `--oracle-dsn` | `ATP_DSN` | Oracle connection descriptor/DSN |
| `--oracle-user` | `ATP_USER` | Oracle database username |
| `--oracle-password` | `ATP_PASSWORD` | Oracle database password |
| `--oracle-pool-min` | `ATP_POOL_MIN` | Minimum connection pool size (default: 1) |
| `--oracle-pool-max` | `ATP_POOL_MAX` | Maximum connection pool size (default: 16) |
| `--oracle-pool-timeout-secs` | `ATP_POOL_TIMEOUT_SECS` | Pool timeout in seconds (default: 30) |

### PostgreSQL Database

| Option | Environment | Description | Default |
|--------|-------------|-------------|---------|
| `--postgres-db-url` | `POSTGRES_DB_URL` | PostgreSQL connection URL | - |
| `--postgres-pool-max-size` | `POSTGRES_POOL_MAX` | Maximum pool size | `16` |

### Redis Database

| Option | Environment | Description | Default |
|--------|-------------|-------------|---------|
| `--redis-url` | `REDIS_URL` | Redis connection URL | - |
| `--redis-pool-max-size` | `REDIS_POOL_MAX` | Maximum pool size | `16` |
| `--redis-retention-days` | `REDIS_RETENTION_DAYS` | Data retention (-1 for persistent) | `30` |

---

## WASM Configuration

### Enable WebAssembly

| Option | `--enable-wasm` |
|--------|-----------------|
| Environment | - |
| Default | `false` |
| Description | Enable WebAssembly support |

---

## Mesh/HA Configuration

High-availability mesh networking for router coordination.

| Option | Description | Default |
|--------|-------------|---------|
| `--enable-mesh` | Enable mesh server for HA | `false` |
| `--mesh-server-name` | Unique name for this mesh node | Auto-generated |
| `--mesh-host` | Host address for mesh server | `0.0.0.0` |
| `--mesh-port` | Port for mesh server | `39527` |
| `--mesh-peer-urls` | URLs of peer mesh nodes | Empty |

**Example**:
```bash
smg \
  --enable-mesh \
  --mesh-server-name router-1 \
  --mesh-port 39527 \
  --mesh-peer-urls 192.168.1.10:39527
```

---

## Request Handling Configuration

### Request Timeout

| Option | `--request-timeout-secs` |
|--------|--------------------------|
| Environment | - |
| Default | `1800` (30 minutes) |
| Description | Maximum time for request processing |

### Shutdown Grace Period

| Option | `--shutdown-grace-period-secs` |
|--------|-------------------------------|
| Environment | - |
| Default | `180` (3 minutes) |
| Description | Time to wait for in-flight requests during shutdown |

### Maximum Payload Size

| Option | `--max-payload-size` |
|--------|----------------------|
| Environment | - |
| Default | `536870912` (512MB) |
| Description | Maximum request payload size in bytes |

### CORS Configuration

| Option | `--cors-allowed-origins` |
|--------|--------------------------|
| Environment | - |
| Default | Empty |
| Format | Space-separated URLs |

**Example**:
```bash
--cors-allowed-origins http://localhost:3000 https://example.com
```

### Request ID Headers

| Option | `--request-id-headers` |
|--------|------------------------|
| Environment | - |
| Default | None (uses common defaults) |
| Description | Custom HTTP headers to check for request IDs |

**Example**:
```bash
--request-id-headers x-request-id x-trace-id x-correlation-id
```

---

## Rate Limiting Configuration

### Concurrent Request Limit

| Option | `--max-concurrent-requests` |
|--------|----------------------------|
| Environment | - |
| Default | `-1` (unlimited) |
| Range | `-1` or `1+` |

**Sizing Guide**:

```
max_concurrent_requests = num_workers * requests_per_worker_capacity
```

| Worker GPU Memory | Suggested per Worker |
|-------------------|---------------------|
| 16GB | 4-8 |
| 40GB | 8-16 |
| 80GB | 16-32 |

### Queue Configuration

| Option | Description | Default |
|--------|-------------|---------|
| `--queue-size` | Maximum requests waiting when rate limit reached | `100` |
| `--queue-timeout-secs` | Maximum time a request can wait in queue | `60` |

### Token Bucket Rate Limiting

| Option | `--rate-limit-tokens-per-second` |
|--------|----------------------------------|
| Environment | - |
| Default | Same as `max-concurrent-requests` |
| Description | Token bucket refill rate |

---

## Retry Configuration

### Retry Options

| Option | Description | Default |
|--------|-------------|---------|
| `--retry-max-retries` | Maximum retry attempts | `5` |
| `--retry-initial-backoff-ms` | Initial backoff delay (ms) | `50` |
| `--retry-max-backoff-ms` | Maximum backoff delay (ms) | `30000` |
| `--retry-backoff-multiplier` | Exponential backoff multiplier | `1.5` |
| `--retry-jitter-factor` | Jitter factor (0.0-1.0) | `0.2` |
| `--disable-retries` | Disable automatic retries | `false` |

**Backoff Formula**:
```
delay = min(initial_backoff * multiplier^attempt, max_backoff) * (1 + random(0, jitter_factor))
```

---

## Circuit Breaker Configuration

| Option | Description | Default |
|--------|-------------|---------|
| `--cb-failure-threshold` | Failures before circuit opens | `10` |
| `--cb-success-threshold` | Successes needed to close in half-open state | `3` |
| `--cb-timeout-duration-secs` | Time before attempting recovery | `60` |
| `--cb-window-duration-secs` | Sliding window for tracking failures | `120` |
| `--disable-circuit-breaker` | Disable circuit breaker | `false` |

**Circuit Breaker States**:
- **Closed**: Normal operation, tracking failures
- **Open**: All requests fail fast, circuit tripped
- **Half-Open**: Testing if service recovered

---

## Health Check Configuration

| Option | Description | Default |
|--------|-------------|---------|
| `--health-failure-threshold` | Failures before marking unhealthy | `3` |
| `--health-success-threshold` | Successes before marking healthy | `2` |
| `--health-check-timeout-secs` | Timeout for health check requests | `5` |
| `--health-check-interval-secs` | Interval between health checks | `60` |
| `--health-check-endpoint` | Health check endpoint path | `/health` |
| `--disable-health-check` | Disable all health checks | `false` |

---

## Prometheus Metrics Configuration

### Metrics Server

| Option | Description | Default |
|--------|-------------|---------|
| `--prometheus-port` | Port for Prometheus metrics endpoint | `29000` |
| `--prometheus-host` | Host for Prometheus metrics server | `0.0.0.0` |
| `--prometheus-duration-buckets` | Custom histogram buckets | Default buckets |

**Example**:
```bash
--prometheus-duration-buckets 0.001 0.005 0.01 0.025 0.05 0.1 0.25 0.5 1.0 2.5 5.0 10.0
```

---

## OpenTelemetry Configuration

### Enable Tracing

| Option | `--enable-trace` |
|--------|------------------|
| Environment | - |
| Default | `false` |

### OTLP Endpoint

| Option | `--otlp-traces-endpoint` |
|--------|--------------------------|
| Environment | - |
| Default | `localhost:4317` |
| Format | `host:port` |

**Example**:
```bash
smg --enable-trace --otlp-traces-endpoint jaeger:4317
```

---

## TLS/mTLS Security Configuration

### Server TLS

For HTTPS on the gateway:

| Option | Description |
|--------|-------------|
| `--tls-cert-path` | Path to server certificate (PEM format) |
| `--tls-key-path` | Path to server private key (PEM format) |

### Client mTLS

For secure communication to workers (Python bindings):

| Option | Description |
|--------|-------------|
| `--client-cert-path` | Path to client certificate |
| `--client-key-path` | Path to client private key |
| `--ca-cert-paths` | Path(s) to CA certificate(s) |

---

## Control Plane Authentication

### API Key (Worker Authorization)

| Option | `--api-key` |
|--------|-------------|
| Environment | - |
| Default | None |
| Description | API key for worker authorization (useful with dp-aware scheduling) |

### Control Plane API Keys

| Option | `--control-plane-api-keys` |
|--------|---------------------------|
| Environment | `CONTROL_PLANE_API_KEYS` |
| Format | `id:name:role:key` |
| Multiple | Yes |

**Example**:
```bash
--control-plane-api-keys 'key1:Admin:admin:secret123' 'key2:ReadOnly:user:secret456'
```

### JWT/OIDC Authentication

| Option | Environment | Description |
|--------|-------------|-------------|
| `--jwt-issuer` | `JWT_ISSUER` | OIDC issuer URL |
| `--jwt-audience` | `JWT_AUDIENCE` | Expected audience claim |
| `--jwt-jwks-uri` | `JWT_JWKS_URI` | Explicit JWKS URI (auto-discovered if not set) |
| `--jwt-role-claim` | - | JWT claim containing role (default: `roles`) |
| `--jwt-role-mapping` | - | Role mapping from IDP to gateway role |

**JWT Role Mapping Example**:
```bash
--jwt-role-mapping 'Gateway.Admin=admin' 'Gateway.User=user'
```

### Audit Logging

| Option | `--disable-audit-logging` |
|--------|---------------------------|
| Environment | - |
| Default | `false` (audit logging enabled) |

---

## Logging Configuration

### Log Directory

| Option | `--log-dir` |
|--------|-------------|
| Environment | - |
| Default | None (console only) |
| Description | Directory to store log files |

### Log File Name

| Option | `--log-file-name` |
|--------|-------------------|
| Environment | - |
| Default | `smg` |
| Description | Current log file name |

### Log Format

| Option | `--log-format` |
|--------|----------------|
| Environment | - |
| Default | `plain` |
| Values | `plain`, `json` |

### Log Rotation Frequency

| Option | `--log-rotate-frequency` |
|--------|---------------------------|
| Environment | - |
| Default | `daily` |
| Values | `daily`, `hourly` |

### Log Rotation Count

| Option | `--log-rotate-count` |
|--------|----------------------|
| Environment | - |
| Default | `3` |
| Description | Number of archived log files to keep |

### Log Level

| Option | `--log-level` |
|--------|---------------|
| Environment | `RUST_LOG` |
| Default | `info` |
| Values | `debug`, `info`, `warn`, `error` |

**Per-Module Logging**:

```bash
RUST_LOG=smg=debug,hyper=warn smg ...
```

---

## Configuration Examples

### Minimal Configuration

```bash
smg --worker-urls http://localhost:8000
```

### High-Throughput Configuration

```bash
smg \
  --worker-urls http://w1:8000 http://w2:8000 http://w3:8000 http://w4:8000 \
  --policy cache_aware \
  --max-concurrent-requests 200 \
  --queue-size 400 \
  --queue-timeout-secs 60 \
  --retry-max-retries 3
```

### Low-Latency Configuration

```bash
smg \
  --worker-urls http://w1:8000 http://w2:8000 \
  --policy power_of_two \
  --max-concurrent-requests 50 \
  --queue-size 25 \
  --queue-timeout-secs 5 \
  --health-check-interval-secs 5 \
  --request-timeout-secs 30
```

### PD Disaggregated Mode

```bash
smg \
  --pd-disaggregation \
  --prefill http://prefill1:30001 9001 \
  --prefill http://prefill2:30002 9002 \
  --decode http://decode1:30003 \
  --decode http://decode2:30004 \
  --prefill-policy cache_aware \
  --decode-policy round_robin
```

### Kubernetes Service Discovery

```bash
smg \
  --service-discovery \
  --selector app=sglang-worker \
  --service-discovery-namespace inference \
  --service-discovery-port 8000 \
  --policy cache_aware
```

### High-Availability Mesh

```bash
# Router 1
smg \
  --enable-mesh \
  --mesh-server-name router-1 \
  --mesh-port 39527 \
  --mesh-peer-urls 192.168.1.11:39527 \
  --worker-urls http://worker1:8000

# Router 2
smg \
  --enable-mesh \
  --mesh-server-name router-2 \
  --mesh-port 39527 \
  --mesh-peer-urls 192.168.1.10:39527 \
  --worker-urls http://worker2:8000
```

### Secure Production Configuration

```bash
smg \
  --service-discovery \
  --selector app=sglang-worker \
  --service-discovery-namespace inference \
  --policy cache_aware \
  --max-concurrent-requests 100 \
  --tls-cert-path /etc/certs/server.crt \
  --tls-key-path /etc/certs/server.key \
  --jwt-issuer https://login.microsoftonline.com/tenant/v2.0 \
  --jwt-audience api://smg-gateway \
  --jwt-role-mapping 'Gateway.Admin=admin' 'Gateway.User=user' \
  --enable-trace \
  --otlp-traces-endpoint jaeger:4317 \
  --host 0.0.0.0 \
  --port 443
```

### With Tokenizer and Parsers

```bash
smg \
  --worker-urls http://localhost:8000 \
  --model-path meta-llama/Llama-3-8B-Instruct \
  --tokenizer-cache-enable-l0 \
  --tokenizer-cache-l0-max-entries 50000 \
  --reasoning-parser deepseek-r1 \
  --tool-call-parser json
```

### With Database Backend

```bash
# PostgreSQL
smg \
  --worker-urls http://localhost:8000 \
  --history-backend postgres \
  --postgres-db-url "postgres://user:pass@localhost:5432/smg" \
  --postgres-pool-max-size 32

# Redis
smg \
  --worker-urls http://localhost:8000 \
  --history-backend redis \
  --redis-url "redis://localhost:6379" \
  --redis-pool-max-size 32 \
  --redis-retention-days 7
```

---

## Environment Variable Reference

| Environment Variable | CLI Option | Description |
|---------------------|------------|-------------|
| `RUST_LOG` | `--log-level` | Log level |
| `ATP_WALLET_PATH` | `--oracle-wallet-path` | Oracle wallet path |
| `ATP_TNS_ALIAS` | `--oracle-tns-alias` | Oracle TNS alias |
| `ATP_DSN` | `--oracle-dsn` | Oracle DSN |
| `ATP_USER` | `--oracle-user` | Oracle username |
| `ATP_PASSWORD` | `--oracle-password` | Oracle password |
| `ATP_POOL_MIN` | `--oracle-pool-min` | Oracle min pool size |
| `ATP_POOL_MAX` | `--oracle-pool-max` | Oracle max pool size |
| `ATP_POOL_TIMEOUT_SECS` | `--oracle-pool-timeout-secs` | Oracle pool timeout |
| `POSTGRES_DB_URL` | `--postgres-db-url` | PostgreSQL URL |
| `POSTGRES_POOL_MAX` | `--postgres-pool-max-size` | PostgreSQL max pool |
| `REDIS_URL` | `--redis-url` | Redis URL |
| `REDIS_POOL_MAX` | `--redis-pool-max-size` | Redis max pool |
| `REDIS_RETENTION_DAYS` | `--redis-retention-days` | Redis retention |
| `JWT_ISSUER` | `--jwt-issuer` | JWT issuer URL |
| `JWT_AUDIENCE` | `--jwt-audience` | JWT audience |
| `JWT_JWKS_URI` | `--jwt-jwks-uri` | JWKS URI |
| `CONTROL_PLANE_API_KEYS` | `--control-plane-api-keys` | Control plane API keys |
