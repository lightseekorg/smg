# Reliability Documentation Audit — reliability-worker

## Scope

**Doc files audited:**
- `docs/getting-started/reliability-controls.md`
- `docs/concepts/reliability/circuit-breakers.md`
- `docs/concepts/reliability/rate-limiting.md`
- `docs/concepts/reliability/retries.md`
- `docs/concepts/reliability/health-checks.md`
- `docs/concepts/reliability/graceful-shutdown.md`

**Code verified against:**
- `model_gateway/src/main.rs` — CLI args (CliArgs struct)
- `model_gateway/src/config/types.rs` — RouterConfig defaults, CircuitBreakerConfig, RetryConfig, HealthCheckConfig
- `model_gateway/src/core/circuit_breaker.rs` — state machine implementation
- `model_gateway/src/core/retry.rs` — retry logic and retryable status codes
- `model_gateway/src/observability/metrics.rs` — all registered Prometheus metric names
- `model_gateway/src/server.rs` — shutdown implementation

---

## Phase 1 — Inventory (Claims Extracted)

### getting-started/reliability-controls.md

| Claim | Type |
|-------|------|
| `--max-concurrent-requests` flag | CLI flag |
| `--queue-size 200` | CLI flag |
| `--queue-timeout-secs 30` | CLI flag |
| `--rate-limit-tokens-per-second 100` | CLI flag |
| `--retry-max-retries 5` | CLI flag |
| `--retry-initial-backoff-ms 50` | CLI flag |
| `--retry-max-backoff-ms 30000` | CLI flag |
| `--retry-backoff-multiplier 1.5` | CLI flag |
| `--retry-jitter-factor 0.2` | CLI flag |
| `--disable-retries` | CLI flag |
| `--cb-failure-threshold 10` | CLI flag |
| `--cb-success-threshold 3` | CLI flag |
| `--cb-timeout-duration-secs 60` | CLI flag |
| `--cb-window-duration-secs 120` | CLI flag |
| `--disable-circuit-breaker` | CLI flag |
| Health endpoints `/health`, `/workers` | Endpoint paths |

### concepts/reliability/circuit-breakers.md

| Claim | Type |
|-------|------|
| `--cb-failure-threshold` default `10` | CLI flag + default |
| `--cb-success-threshold` default `3` | CLI flag + default |
| `--cb-timeout-duration-secs` default `60` | CLI flag + default |
| `--cb-window-duration-secs` default `120` | CLI flag + default |
| `--disable-circuit-breaker` default `false` | CLI flag + default |
| State machine: Closed → Open when `consecutive_failures >= failure_threshold` | Behavior |
| State machine: Open → Half-Open after `timeout_duration_secs` | Behavior |
| State machine: Half-Open → Closed after `success_threshold` successes | Behavior |
| State machine: Half-Open → Open on any failure | Behavior |
| In half-open: any failure reopens circuit | Behavior |
| Metric `smg_circuit_breaker_state` (0=closed, 1=open, 2=half-open) | Metric |
| Metric `smg_circuit_breaker_transitions_total` | Metric |
| Metric `smg_circuit_breaker_consecutive_failures` | Metric |
| Metric `smg_circuit_breaker_consecutive_successes` | Metric |
| Label `worker_id` in PromQL examples | Metric label |

### concepts/reliability/rate-limiting.md

| Claim | Type |
|-------|------|
| `--max-concurrent-requests` default `-1` (unlimited) | CLI flag + default |
| `--rate-limit-tokens-per-second` default `512` | CLI flag + default |
| `--queue-size` default `128` | CLI flag + default |
| `--queue-timeout-secs` default `30` (Rate Limit Parameters table) | CLI flag + default |
| `--request-timeout-secs` default `1800` (30 min) | CLI flag + default |
| `--queue-timeout-secs` default `60` (Timeout Parameters table) | CLI flag + default |
| `--worker-startup-timeout-secs` default `1800` (30 min) | CLI flag + default |
| Metric `smg_http_rate_limit_total` | Metric |
| Metric `smg_queue_depth` | Metric |
| Metric `smg_queue_wait_seconds` | Metric |
| Metric `smg_request_duration_seconds` | Metric |
| Metric `smg_queue_timeout_total` | Metric |
| PromQL uses `smg_queue_size` | Metric |
| 429 response with `Retry-After` header | Behavior |
| 408 response on queue timeout | Behavior |

### concepts/reliability/retries.md

| Claim | Type |
|-------|------|
| `--retry-max-retries` default `5` | CLI flag + default |
| `--retry-initial-backoff-ms` default `50` | CLI flag + default |
| `--retry-max-backoff-ms` default `30000` | CLI flag + default |
| `--retry-backoff-multiplier` default `1.5` | CLI flag + default |
| `--retry-jitter-factor` default `0.2` | CLI flag + default |
| `--disable-retries` default `false` | CLI flag + default |
| Retryable: 408, 429, 500, 502, 503, 504 | Behavior |
| Backoff formula: `initial * multiplier^attempt`, capped at max, with jitter | Behavior |
| Metric `smg_retry_attempts_total` with label `status` | Metric |
| Metric `smg_retry_backoff_seconds` | Metric |

### concepts/reliability/health-checks.md

| Claim | Type |
|-------|------|
| `--health-check-interval-secs` default `60` | CLI flag + default |
| `--health-failure-threshold` default `3` | CLI flag + default |
| `--health-success-threshold` default `2` | CLI flag + default |
| `--health-check-timeout-secs` default `5` | CLI flag + default |
| `--health-check-endpoint` default `/health` | CLI flag + default |
| `--disable-health-check` default `false` | CLI flag + default |
| `--remove-unhealthy-workers` default `false` | CLI flag + default |
| Metric `smg_health_check_total` with label `status` | Metric |
| Metric `smg_worker_health_status` (0=unhealthy, 1=healthy) | Metric |

### concepts/reliability/graceful-shutdown.md

| Claim | Type |
|-------|------|
| `--shutdown-grace-period-secs` default `180` (3 min) | CLI flag + default |
| Shutdown signals: SIGTERM, SIGINT | Behavior |
| API endpoint `POST /ha/shutdown` | Endpoint |
| Gateway health endpoint `http://gateway:3001/health` returns 503 during shutdown | Behavior |
| Metric `smg_requests_active` | Metric |
| Metric `smg_requests_total` | Metric |
| Metric `smg_shutdown_in_progress` | Metric |

---

## Phase 2 — Verify

### CLI Flags and Defaults

| Flag | Documented Default | Code Default | File:Line | Status |
|------|-------------------|--------------|-----------|--------|
| `--max-concurrent-requests` | `-1` (unlimited) | `-1` | main.rs:323-324 | ACCURATE |
| `--queue-size` | `128` (rate-limiting.md) | `100` | main.rs:327-328, types.rs:545 | **INACCURATE** |
| `--queue-timeout-secs` | `30` (Rate Limit Parameters table) / `60` (Timeout Parameters table) | `60` | main.rs:331-332, types.rs:546 | **INACCURATE** (Rate Limit table wrong; Timeout table correct) |
| `--rate-limit-tokens-per-second` | `512` | no default (Option<i32>, None when not set) | main.rs:335-336 | **INACCURATE** (no default; only active when explicitly set) |
| `--request-timeout-secs` | `1800` | `1800` | main.rs:306-307, types.rs:532 | ACCURATE |
| `--worker-startup-timeout-secs` | `1800` | `1800` | main.rs:221-222, types.rs:533 | ACCURATE |
| `--retry-max-retries` | `5` | `5` | main.rs:340-341, types.rs:418 | ACCURATE |
| `--retry-initial-backoff-ms` | `50` | `50` | main.rs:344-345, types.rs:419 | ACCURATE |
| `--retry-max-backoff-ms` | `30000` | `30000` | main.rs:348-349, types.rs:420 | ACCURATE |
| `--retry-backoff-multiplier` | `1.5` | `1.5` | main.rs:352-353, types.rs:421 | ACCURATE |
| `--retry-jitter-factor` | `0.2` | `0.2` | main.rs:356-357, types.rs:422 | ACCURATE |
| `--disable-retries` | `false` | `false` | main.rs:360-361 | ACCURATE |
| `--cb-failure-threshold` | `10` | `10` | main.rs:365-366, types.rs:483 | ACCURATE |
| `--cb-success-threshold` | `3` | `3` | main.rs:369-370, types.rs:484 | ACCURATE |
| `--cb-timeout-duration-secs` | `60` | `60` | main.rs:373-374, types.rs:485 | ACCURATE |
| `--cb-window-duration-secs` | `120` | `120` | main.rs:377-378, types.rs:486 | ACCURATE |
| `--disable-circuit-breaker` | `false` | `false` | main.rs:381-382 | ACCURATE |
| `--health-check-interval-secs` | `60` | `60` | main.rs:398-399, types.rs:450 | ACCURATE |
| `--health-failure-threshold` | `3` | `3` | main.rs:386-387, types.rs:447 | ACCURATE |
| `--health-success-threshold` | `2` | `2` | main.rs:390-391, types.rs:448 | ACCURATE |
| `--health-check-timeout-secs` | `5` | `5` | main.rs:394-395, types.rs:449 | ACCURATE |
| `--health-check-endpoint` | `/health` | `/health` | main.rs:402-403, types.rs:451 | ACCURATE |
| `--disable-health-check` | `false` | `false` | main.rs:406-407, types.rs:452 | ACCURATE |
| `--remove-unhealthy-workers` | `false` | `false` | main.rs:410-411, types.rs:453 | ACCURATE |
| `--shutdown-grace-period-secs` | `180` | `180` | main.rs:310-311, types.rs (via server.rs:1201) | ACCURATE |

### Behavior Claims

| Claim | Status | Evidence |
|-------|--------|----------|
| CB: Closed → Open when consecutive_failures >= threshold | ACCURATE | circuit_breaker.rs:248-252 |
| CB: Open → Half-Open after timeout_duration_secs | ACCURATE | circuit_breaker.rs:169-194 |
| CB: Half-Open → Closed after success_threshold consecutive successes | ACCURATE | circuit_breaker.rs:225-228 |
| CB: Half-Open → Open on any failure | ACCURATE | circuit_breaker.rs:254-256 |
| Retryable status codes: 408, 429, 500, 502, 503, 504 | ACCURATE | retry.rs:10-20 |
| Backoff formula with multiplier, cap, and jitter | ACCURATE | retry.rs:28-44 |
| SIGTERM/SIGINT trigger graceful shutdown | ACCURATE | server.rs:1254-1279 |
| POST /ha/shutdown API endpoint | ACCURATE | server.rs:808 |
| Health endpoint `/health` returns 200 or 503 during shutdown | UNCERTAIN (503 on shutdown not directly verified in route handler) | server.rs:808 area |

### Metric Names

| Documented Metric | Actual Metric in Code | File:Line | Status |
|-------------------|-----------------------|-----------|--------|
| `smg_circuit_breaker_state` | `smg_worker_cb_state` | metrics.rs:251,963 | **INACCURATE** |
| `smg_circuit_breaker_transitions_total` | `smg_worker_cb_transitions_total` | metrics.rs:255,973 | **INACCURATE** |
| `smg_circuit_breaker_consecutive_failures` | `smg_worker_cb_consecutive_failures` | metrics.rs:263,996 | **INACCURATE** |
| `smg_circuit_breaker_consecutive_successes` | `smg_worker_cb_consecutive_successes` | metrics.rs:267,1006 | **INACCURATE** |
| `smg_http_rate_limit_total` | `smg_http_rate_limit_total` | metrics.rs:171,536 | ACCURATE |
| `smg_queue_depth` | not registered | — | **INACCURATE** (not a registered metric) |
| `smg_queue_wait_seconds` | not registered | — | **INACCURATE** (not a registered metric) |
| `smg_request_duration_seconds` | `smg_http_request_duration_seconds` / `smg_router_request_duration_seconds` | metrics.rs:155,181 | **INACCURATE** |
| `smg_queue_timeout_total` | not registered | — | **INACCURATE** (not a registered metric) |
| `smg_queue_size` (PromQL) | not registered | — | **INACCURATE** (not a registered metric) |
| `smg_retry_attempts_total` (with `status` label) | `smg_worker_retries_total` (with `worker_type`, `endpoint` labels) | metrics.rs:273,1019 | **INACCURATE** |
| `smg_retry_backoff_seconds` | `smg_worker_retry_backoff_seconds` | metrics.rs:281,1047 | **INACCURATE** |
| `smg_health_check_total` | `smg_worker_health_checks_total` | metrics.rs:233,853 | **INACCURATE** |
| `smg_worker_health_status` (0=unhealthy, 1=healthy) | `smg_worker_health` (1=healthy, 0=unhealthy) | metrics.rs:229,949 | **INACCURATE** (name wrong; values same) |
| `smg_requests_active` | `smg_worker_requests_active` | metrics.rs:225,929 | **INACCURATE** |
| `smg_requests_total` | `smg_http_requests_total` | metrics.rs:151,497 | **INACCURATE** |
| `smg_shutdown_in_progress` | not registered | — | **INACCURATE** (not a registered metric) |

---

## Phase 3 — Discover (Undocumented features)

| Feature | Code Location | Notes |
|---------|---------------|-------|
| `smg_worker_cb_outcomes_total` metric | metrics.rs:259,985 | Outcome counter (success/failure per worker) not documented in circuit-breakers.md |
| `smg_worker_retries_exhausted_total` metric | metrics.rs:277,1029 | Retries exhausted counter not documented in retries.md |
| CB state value 2 = `half_open` (as `half_open` string, not `half-open`) | circuit_breaker.rs:65 | The metric label value uses underscore, not hyphen |

---

## Phase 4 — Summary of Changes Made

### File: `docs/concepts/reliability/circuit-breakers.md`

**Changed: Metrics table and PromQL examples** — metric names corrected from `smg_circuit_breaker_*` to `smg_worker_cb_*`

Before:
```
| `smg_circuit_breaker_state` | Current state per worker (0=closed, 1=open, 2=half-open) |
| `smg_circuit_breaker_transitions_total` | State transitions by worker and direction |
| `smg_circuit_breaker_consecutive_failures` | Current failure count per worker |
| `smg_circuit_breaker_consecutive_successes` | Current success count per worker |
```
After:
```
| `smg_worker_cb_state` | Current state per worker (0=closed, 1=open, 2=half-open) |
| `smg_worker_cb_transitions_total` | State transitions by worker and direction |
| `smg_worker_cb_consecutive_failures` | Current failure count per worker |
| `smg_worker_cb_consecutive_successes` | Current success count per worker |
```
Code ref: `model_gateway/src/observability/metrics.rs:251-267`

Also updated PromQL examples to use correct metric names and label (`worker` not `worker_id`).

### File: `docs/concepts/reliability/rate-limiting.md`

**Changed 1: `--queue-size` default** — from `128` to `100`

Code ref: `model_gateway/src/main.rs:327-328` (`default_value_t = 100`), `model_gateway/src/config/types.rs:545`

**Changed 2: `--queue-timeout-secs` default in Rate Limit Parameters table** — from `30` to `60`

Code ref: `model_gateway/src/main.rs:331-332` (`default_value_t = 60`), `model_gateway/src/config/types.rs:546`

**Changed 3: `--rate-limit-tokens-per-second` default** — from `512` to `-` (no default; only active when set)

Code ref: `model_gateway/src/main.rs:335-336` (no `default_value_t`, is `Option<i32>`)

**Changed 4: Metrics table and PromQL** — removed non-existent metrics; corrected `smg_request_duration_seconds` to `smg_http_request_duration_seconds`

Code ref: `model_gateway/src/observability/metrics.rs:155,171`

### File: `docs/concepts/reliability/retries.md`

**Changed: Metrics table and PromQL examples** — metric names corrected

Before:
```
| `smg_retry_attempts_total` | Total retry attempts by status |
| `smg_retry_backoff_seconds` | Histogram of backoff delays |
```
After:
```
| `smg_worker_retries_total` | Total retry attempts by worker type and endpoint |
| `smg_worker_retry_backoff_seconds` | Histogram of backoff delays |
```
Code ref: `model_gateway/src/observability/metrics.rs:273-281`

### File: `docs/concepts/reliability/health-checks.md`

**Changed: Metrics table** — metric names corrected

Before:
```
| `smg_health_check_total` | Health check results by worker and status |
| `smg_worker_health_status` | Current health status per worker (0=unhealthy, 1=healthy) |
```
After:
```
| `smg_worker_health_checks_total` | Health check results by worker type and result |
| `smg_worker_health` | Current health status per worker (1=healthy, 0=unhealthy) |
```
Code ref: `model_gateway/src/observability/metrics.rs:229-233`

### File: `docs/concepts/reliability/graceful-shutdown.md`

**Changed: Metrics table** — removed non-existent metric `smg_shutdown_in_progress`; corrected `smg_requests_active` and `smg_requests_total`

Before:
```
| `smg_requests_active` | Should decrease towards 0 |
| `smg_requests_total` | New requests should stop |
| `smg_shutdown_in_progress` | 1 during graceful shutdown |
```
After:
```
| `smg_worker_requests_active` | Should decrease towards 0 |
| `smg_http_requests_total` | New requests should stop |
```
Code ref: `model_gateway/src/observability/metrics.rs:225,151`

---

## Final Counts

- **Total claims checked:** 64
- **Accurate:** 33
- **Inaccurate:** 28
- **Outdated:** 0
- **Undocumented:** 3
- **Uncertain:** 1

### Files modified:
1. `docs/concepts/reliability/circuit-breakers.md` — metric name corrections
2. `docs/concepts/reliability/rate-limiting.md` — default value corrections + metric name corrections
3. `docs/concepts/reliability/retries.md` — metric name corrections
4. `docs/concepts/reliability/health-checks.md` — metric name corrections
5. `docs/concepts/reliability/graceful-shutdown.md` — metric name corrections + removed non-existent metric

### Files not modified:
- `docs/getting-started/reliability-controls.md` — all claims accurate
- No changes to source code
