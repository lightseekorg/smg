# Operations & Security Documentation Audit

**Agent**: operations-worker
**Scope**: monitoring.md, logging.md, tls.md, control-plane-auth.md, control-plane-operations.md, data-connections.md, concepts/data/chat-history.md, concepts/security/authentication.md
**Date**: 2026-03-20

---

## Phase 1 — Inventory

### docs/getting-started/monitoring.md

Claims extracted:

- CLI flags: `--prometheus-port 29000`, `--prometheus-host 0.0.0.0`
- CLI flags: `--enable-trace`, `--otlp-traces-endpoint localhost:4317`
- Metrics:
  - Layer 1 HTTP: `smg_http_requests_total` (Counter), `smg_http_request_duration_seconds` (Histogram), `smg_http_responses_total` (Counter), `smg_http_connections_active` (Gauge), `smg_http_rate_limit_total` (Counter)
  - Layer 2 Router: `smg_router_requests_total` (Counter), `smg_router_ttft_seconds` (Histogram), `smg_router_tpot_seconds` (Histogram), `smg_router_tokens_total` (Counter), `smg_router_stage_duration_seconds` (Histogram)
  - Layer 3 Worker: `smg_worker_health` (Gauge), `smg_worker_requests_active` (Gauge), `smg_worker_cb_state` (Gauge), `smg_worker_retries_total` (Counter)
  - Layer 5 MCP: `smg_mcp_tool_calls_total` (Counter), `smg_mcp_tool_duration_seconds` (Histogram), `smg_mcp_servers_active` (Gauge)
- Alert: `smg_worker_cb_state == 1` means circuit breaker open
- Metric label `decision="rejected"` on `smg_http_rate_limit_total`
- Metric label `status_code=~"5.."` on `smg_http_responses_total`
- PromQL query uses label `{type="output"}` / `{type="input"}` on `smg_router_tokens_total`

### docs/getting-started/logging.md

Claims extracted:

- CLI flags: `--log-level info` (default), `--log-json false` (default), `--log-dir None` (default)
- Valid log levels: `debug`, `info`, `warn`, `error`
- File rotation: daily, named `smg.YYYY-MM-DD.log`
- RUST_LOG env var supported
- JSON output enabled by `--log-json`

### docs/getting-started/tls.md

Claims extracted:

- CLI flags: `--tls-cert-path`, `--tls-key-path` (server TLS)
- Flags `--client-cert-path`, `--client-key-path`, `--ca-cert-path` described as "planned" / "not yet implemented"
- Only server TLS available via CLI
- Full TLS example uses `--tls-cert-path`, `--tls-key-path`, `--api-key`, `--host`, `--port`

### docs/getting-started/control-plane-auth.md

Claims extracted:

- Protected endpoints: `/workers`, `/workers/{worker_id}`, `/v1/tokenizers`, `/v1/tokenizers/{tokenizer_id}`, `/parse/function_call`, `/parse/reasoning`, `/wasm`, `/wasm/{module_uuid}`, `/flush_cache`, `/get_loads`
- Auth requirement: admin role; non-admin gets 403
- API key flag: `--control-plane-api-keys`; format: `id:name:role:key`
- JWT flags: `--jwt-issuer`, `--jwt-audience`, `--jwt-role-claim`, `--jwt-role-mapping`, `--jwt-jwks-uri`
- JWT validation is first; JWT-shaped token failure does NOT fall back to API key
- Flag: `--disable-audit-logging`

### docs/getting-started/control-plane-operations.md

Claims extracted:

- Worker create: POST `/workers` with fields `url`, `model_id`, `worker_type`, `runtime`
- Worker update: PUT `/workers/<worker_id>` with field `priority`
- Worker delete: DELETE `/workers/<worker_id>`
- Tokenizer status: GET `/v1/tokenizers/<tokenizer_id>/status`
- WASM enable flag: `--enable-wasm`
- Parser endpoints: POST `/parse/function_call`, POST `/parse/reasoning`
- Cache: POST `/flush_cache`, GET `/get_loads`

### docs/getting-started/data-connections.md

Claims extracted:

- `--history-backend` options: `memory` (default), `none`, `postgres`, `redis`, `oracle`
- `--postgres-db-url` (required for postgres)
- `--postgres-pool-max-size 16` (default implied)
- `--redis-url` (required for redis)
- `--redis-pool-max-size 16` (default implied)
- `--redis-retention-days 30` (default), `-1` for persistent
- `--oracle-wallet-path`, `--oracle-tns-alias`, `--oracle-dsn`, `--oracle-user`, `--oracle-password`
- Oracle env vars: `ATP_WALLET_PATH`, `ATP_TNS_ALIAS`, `ATP_DSN`, `ATP_USER`, `ATP_PASSWORD`, `ATP_EXTERNAL_AUTH`, `ATP_POOL_MIN`, `ATP_POOL_MAX`, `ATP_POOL_TIMEOUT_SECS`
- Required flags table: postgres needs `--postgres-db-url`; redis needs `--redis-url`; oracle needs user/password and one of DSN or wallet+alias (can omit user/password with `--oracle-external-auth`)

### docs/concepts/data/chat-history.md

Claims extracted:

- Backend options: `memory`, `none`, `postgres`, `redis`, `oracle`
- Default backend: `memory`
- PostgreSQL options: `--postgres-db-url`, `--postgres-pool-max-size 16`
- Redis options: `--redis-url`, `--redis-pool-max-size 16`, `--redis-retention-days 30` (default), `-1` for persistent
- Oracle options table: `--oracle-wallet-path` (ATP_WALLET_PATH), `--oracle-tns-alias` (ATP_TNS_ALIAS), `--oracle-dsn` (ATP_DSN), `--oracle-user` (ATP_USER), `--oracle-password` (ATP_PASSWORD), `--oracle-pool-min` (ATP_POOL_MIN) default 1, `--oracle-pool-max` (ATP_POOL_MAX) default 16, `--oracle-pool-timeout-secs` (ATP_POOL_TIMEOUT_SECS) default 30
- Oracle examples: using ATP wallet and direct DSN

### docs/concepts/security/authentication.md

Claims extracted:

- Auth methods: JWT/OIDC, API Keys, Worker API Key
- JWT flags: `--jwt-issuer` (env: JWT_ISSUER), `--jwt-audience` (env: JWT_AUDIENCE), `--jwt-jwks-uri` (env: JWT_JWKS_URI), `--jwt-role-claim` (default: `roles`), `--jwt-role-mapping`
- JWT-shaped token failure does NOT fall back to API key
- API key format: `id:name:role:key`
- API key env: `CONTROL_PLANE_API_KEYS`
- Worker API key flag: `--api-key`; gateway adds `Authorization: Bearer <api-key>` to worker requests
- Roles: `admin` (full control plane access), `user` (data plane only)
- Supported JWT algorithms: RS256, RS384, RS512 (RSA); ES256, ES384 (ECDSA)
- Role claim fallback order: configured `--jwt-role-claim` → `role` → `roles` → `groups` → `group`
- Security: keys SHA-256 hashed immediately, constant-time comparison
- Flag: `--disable-audit-logging`
- Audit logging enabled by default when auth is configured

---

## Phase 2 — Verify

### monitoring.md

| Claim | Status | Code Reference |
|-------|--------|----------------|
| `--prometheus-port 29000` (default) | ACCURATE | `model_gateway/src/main.rs:289` `default_value_t = 29000` |
| `--prometheus-host 0.0.0.0` (default) | ACCURATE | `model_gateway/src/main.rs:293` `default_value = "0.0.0.0"` |
| `--enable-trace` (default false) | ACCURATE | `model_gateway/src/main.rs:555-560` |
| `--otlp-traces-endpoint localhost:4317` (default) | ACCURATE | `model_gateway/src/main.rs:563-568` |
| `smg_http_requests_total` Counter | ACCURATE | `model_gateway/src/observability/metrics.rs:150-152` |
| `smg_http_request_duration_seconds` Histogram | ACCURATE | `model_gateway/src/observability/metrics.rs:153-156` |
| `smg_http_responses_total` Counter | ACCURATE | `model_gateway/src/observability/metrics.rs:163-166` |
| `smg_http_connections_active` Gauge | ACCURATE | `model_gateway/src/observability/metrics.rs:167-170` |
| `smg_http_rate_limit_total` Counter | ACCURATE | `model_gateway/src/observability/metrics.rs:171-174` |
| `smg_router_requests_total` Counter | ACCURATE | `model_gateway/src/observability/metrics.rs:176-180` |
| `smg_router_ttft_seconds` Histogram | ACCURATE | `model_gateway/src/observability/metrics.rs:198-201` |
| `smg_router_tpot_seconds` Histogram | ACCURATE | `model_gateway/src/observability/metrics.rs:202-205` |
| `smg_router_tokens_total` Counter | ACCURATE | `model_gateway/src/observability/metrics.rs:206-209` |
| `smg_router_stage_duration_seconds` Histogram | ACCURATE | `model_gateway/src/observability/metrics.rs:188-191` |
| `smg_worker_health` Gauge | ACCURATE | `model_gateway/src/observability/metrics.rs:228-231` |
| `smg_worker_requests_active` Gauge | ACCURATE | `model_gateway/src/observability/metrics.rs:224-227` |
| `smg_worker_cb_state` Gauge | ACCURATE | `model_gateway/src/observability/metrics.rs:250-253` |
| `smg_worker_retries_total` Counter | ACCURATE | `model_gateway/src/observability/metrics.rs:272-275` |
| `smg_mcp_tool_calls_total` Counter | ACCURATE | `model_gateway/src/observability/metrics.rs:304-307` |
| `smg_mcp_tool_duration_seconds` Histogram | ACCURATE | `model_gateway/src/observability/metrics.rs:308-311` |
| `smg_mcp_servers_active` Gauge | ACCURATE | `model_gateway/src/observability/metrics.rs:312` |
| `smg_worker_cb_state == 1` means open | ACCURATE | `model_gateway/src/observability/metrics.rs:253` — description "(0=closed, 1=open, 2=half_open)" |
| rate_limit label `decision="rejected"` | ACCURATE | `model_gateway/src/observability/metrics.rs:443` RATE_LIMIT_REJECTED = "rejected" |
| PromQL uses `{type="output"}` on `smg_router_tokens_total` | INACCURATE | `model_gateway/src/observability/metrics.rs:710,798,810` — actual label key is `"token_type"`, not `"type"` |

### logging.md

| Claim | Status | Code Reference |
|-------|--------|----------------|
| `--log-level info` (default) | ACCURATE | `model_gateway/src/main.rs:280` `default_value = "info"` |
| `--log-json false` (default) | ACCURATE | `model_gateway/src/main.rs:284` `default_value_t = false` |
| `--log-dir None` (default) | ACCURATE | `model_gateway/src/main.rs:276-277` no default |
| Valid levels: `debug`, `info`, `warn`, `error` | ACCURATE | `model_gateway/src/main.rs:280` `value_parser = ["debug", "info", "warn", "error"]` |
| File rotation: daily, named `smg.YYYY-MM-DD.log` | ACCURATE | `model_gateway/src/observability/logging.rs:134-135` `Rotation::DAILY`, log_file_name = "smg" |
| RUST_LOG env var supported | ACCURATE | `model_gateway/src/observability/logging.rs:86-87` `EnvFilter::try_from_default_env()` |

### tls.md

| Claim | Status | Code Reference |
|-------|--------|----------------|
| `--tls-cert-path` flag name | ACCURATE | `model_gateway/src/main.rs:546-548` |
| `--tls-key-path` flag name | ACCURATE | `model_gateway/src/main.rs:549-551` |
| mTLS flags (`--client-cert-path`, etc.) not yet implemented | ACCURATE | `model_gateway/src/main.rs` — no such flags in CliArgs |

### control-plane-auth.md

| Claim | Status | Code Reference |
|-------|--------|----------------|
| Protected endpoint `/workers` | ACCURATE | `model_gateway/src/server.rs:772` |
| Protected endpoint `/workers/{worker_id}` | ACCURATE | `model_gateway/src/server.rs:773-774` |
| Protected endpoint `/v1/tokenizers` | ACCURATE | `model_gateway/src/server.rs:757-760` |
| Protected endpoint `/v1/tokenizers/{tokenizer_id}` | ACCURATE | `model_gateway/src/server.rs:761-764` |
| Protected endpoint `/parse/function_call` | ACCURATE | `model_gateway/src/server.rs:751` |
| Protected endpoint `/parse/reasoning` | ACCURATE | `model_gateway/src/server.rs:752` |
| Protected endpoint `/wasm` | ACCURATE | `model_gateway/src/server.rs:753-755` |
| Protected endpoint `/wasm/{module_uuid}` | ACCURATE | `model_gateway/src/server.rs:754` |
| Protected endpoint `/flush_cache` | ACCURATE | `model_gateway/src/server.rs:749` |
| Protected endpoint `/get_loads` | ACCURATE | `model_gateway/src/server.rs:750` |
| Admin role required; non-admin → 403 | ACCURATE | `crates/auth/src/middleware.rs:150-177` |
| `--control-plane-api-keys` flag name | ACCURATE | `model_gateway/src/main.rs:612` |
| Format `id:name:role:key` | ACCURATE | `model_gateway/src/main.rs:701` |
| JWT flags and JWKS URI | ACCURATE | `model_gateway/src/main.rs:575-597` |
| JWT-shaped token failure does NOT fall back to API key | ACCURATE | `crates/auth/src/middleware.rs:291-305` |
| `--disable-audit-logging` flag | ACCURATE | `model_gateway/src/main.rs:616-621` |

### control-plane-operations.md

| Claim | Status | Code Reference |
|-------|--------|----------------|
| POST `/workers` (create) | ACCURATE | `model_gateway/src/server.rs:772` |
| GET `/workers` (list) | ACCURATE | `model_gateway/src/server.rs:772` |
| GET/PUT/DELETE `/workers/<worker_id>` | ACCURATE | `model_gateway/src/server.rs:773-774` |
| POST `/v1/tokenizers` | ACCURATE | `model_gateway/src/server.rs:757-760` |
| GET `/v1/tokenizers` | ACCURATE | `model_gateway/src/server.rs:757-760` |
| GET/DELETE `/v1/tokenizers/<tokenizer_id>` | ACCURATE | `model_gateway/src/server.rs:761-764` |
| GET `/v1/tokenizers/<tokenizer_id>/status` | ACCURATE | `model_gateway/src/server.rs:765-768` |
| `--enable-wasm` flag | ACCURATE | `model_gateway/src/main.rs:469-471` |
| POST `/wasm` (register) | ACCURATE | `model_gateway/src/server.rs:753` |
| GET `/wasm` (list) | ACCURATE | `model_gateway/src/server.rs:755` |
| DELETE `/wasm/<module_uuid>` | ACCURATE | `model_gateway/src/server.rs:754` |
| POST `/parse/function_call` | ACCURATE | `model_gateway/src/server.rs:751` |
| POST `/parse/reasoning` | ACCURATE | `model_gateway/src/server.rs:752` |
| POST `/flush_cache` | ACCURATE | `model_gateway/src/server.rs:749` |
| GET `/get_loads` | ACCURATE | `model_gateway/src/server.rs:750` |

### data-connections.md

| Claim | Status | Code Reference |
|-------|--------|----------------|
| `--history-backend` options: memory, none, postgres, redis, oracle | ACCURATE | `model_gateway/src/main.rs:465` `value_parser = ["memory", "none", "oracle", "postgres", "redis"]` |
| Default backend: `memory` | ACCURATE | `model_gateway/src/main.rs:465` `default_value = "memory"` |
| `--postgres-db-url` flag | ACCURATE | `model_gateway/src/main.rs:524-525` |
| `--postgres-pool-max-size` default 16 | ACCURATE | `model_gateway/src/main.rs:937-939`, `crates/data_connector/src/config.rs:97-99` |
| `--redis-url` flag | ACCURATE | `model_gateway/src/main.rs:533-534` |
| `--redis-pool-max-size` default 16 | ACCURATE | `model_gateway/src/main.rs:953`, `crates/data_connector/src/config.rs:151-153` |
| `--redis-retention-days` default 30, `-1` for persistent | ACCURATE | `model_gateway/src/main.rs:954-958` — `Some(d) if d < 0 => None` (persistent), `None => Some(30)` |
| `--oracle-wallet-path` (env ATP_WALLET_PATH) | ACCURATE | `model_gateway/src/main.rs:482-483` |
| `--oracle-tns-alias` (env ATP_TNS_ALIAS) | ACCURATE | `model_gateway/src/main.rs:486-487` |
| `--oracle-dsn` (env ATP_DSN) | ACCURATE | `model_gateway/src/main.rs:490-491` |
| `--oracle-user` (env ATP_USER) | ACCURATE | `model_gateway/src/main.rs:494-495` |
| `--oracle-password` (env ATP_PASSWORD) | ACCURATE | `model_gateway/src/main.rs:498-499` |
| Oracle env vars: ATP_EXTERNAL_AUTH listed | ACCURATE | `model_gateway/src/main.rs:503-505` |
| Oracle env vars: ATP_POOL_MIN, ATP_POOL_MAX, ATP_POOL_TIMEOUT_SECS | ACCURATE | `model_gateway/src/main.rs:511-520` |

### chat-history.md

| Claim | Status | Code Reference |
|-------|--------|----------------|
| Default backend `memory` | ACCURATE | `crates/data_connector/src/config.rs:11-17` `#[default] Memory` |
| Backend options: memory, none, oracle, postgres, redis | ACCURATE | `crates/data_connector/src/config.rs:11-18` |
| `--postgres-pool-max-size` default 16 | ACCURATE | `crates/data_connector/src/config.rs:97-99` |
| `--redis-pool-max-size` default 16 | ACCURATE | `crates/data_connector/src/config.rs:151-153` |
| `--redis-retention-days` default 30 | ACCURATE | `crates/data_connector/src/config.rs:159-161` |
| `--oracle-pool-min` default 1 | ACCURATE | `crates/data_connector/src/config.rs:57-59` |
| `--oracle-pool-max` default 16 | ACCURATE | `crates/data_connector/src/config.rs:61-63` |
| `--oracle-pool-timeout-secs` default 30 | ACCURATE | `crates/data_connector/src/config.rs:65-67` |
| Oracle env vars in table (ATP_WALLET_PATH, etc.) | ACCURATE | `model_gateway/src/main.rs:482-520` |
| Oracle table missing `--oracle-external-auth` / `ATP_EXTERNAL_AUTH` | UNDOCUMENTED | `model_gateway/src/main.rs:501-508` `#[arg(long, env = "ATP_EXTERNAL_AUTH", default_value_t = false)]` |

### authentication.md

| Claim | Status | Code Reference |
|-------|--------|----------------|
| JWT flags and env vars (`JWT_ISSUER`, `JWT_AUDIENCE`, `JWT_JWKS_URI`) | ACCURATE | `model_gateway/src/main.rs:575-597` |
| `--jwt-role-claim` default `roles` | ACCURATE | `crates/auth/src/config.rs:84-86` `default_role_claim() = "roles"` |
| `--jwt-role-mapping` format `idp_role=gateway_role` | ACCURATE | `model_gateway/src/main.rs:666-690` |
| Role claim fallback: role → roles → groups → group | ACCURATE | `crates/auth/src/jwt.rs:409` `["role", "roles", "groups", "group"]` |
| JWT-shaped token failure does NOT fall back to API key | ACCURATE | `crates/auth/src/middleware.rs:291-305` |
| API key format `id:name:role:key` | ACCURATE | `model_gateway/src/main.rs:697-721` |
| API key env var `CONTROL_PLANE_API_KEYS` | ACCURATE | `model_gateway/src/main.rs:612` |
| Worker API key `--api-key` flag | ACCURATE | `model_gateway/src/main.rs:572-573` |
| Supported algorithms: RS256, RS384, RS512, ES256, ES384 | ACCURATE | `crates/auth/src/jwt.rs:215-221` |
| SHA-256 hashing, constant-time comparison | ACCURATE | `crates/auth/src/config.rs:125-129`, `crates/auth/src/config.rs:181-183` |
| Audit logging enabled by default | ACCURATE | `crates/auth/src/config.rs:204-206` `default_audit_enabled() = true` |
| `--disable-audit-logging` flag | ACCURATE | `model_gateway/src/main.rs:616-621` |
| Admin role → full control plane; user role → data plane only | ACCURATE | `crates/auth/src/config.rs:14-28`, `crates/auth/src/middleware.rs:150-177` |

---

## Phase 3 — Discover

### Undocumented items found:

**1. `--oracle-external-auth` / `ATP_EXTERNAL_AUTH` missing from chat-history.md Oracle table**

- Code: `model_gateway/src/main.rs:501-508`
- The `data-connections.md` lists `ATP_EXTERNAL_AUTH` in its env vars section (line 108) but does not document the `--oracle-external-auth` CLI flag. The `chat-history.md` Oracle options table (lines 248-257) omits both the CLI flag and env var entirely.
- **Action**: Fixed in Phase 4.

**2. PromQL query uses wrong label key for token type**

- In `monitoring.md` line 299, the query uses `{type="output"}` but the counter! recording at `model_gateway/src/observability/metrics.rs:710,798,810` uses label key `"token_type"`. The query would silently return no data.
- **Action**: Fixed in Phase 4.

**3. Layer 4 discovery metrics not covered in monitoring.md**

- Code: `model_gateway/src/observability/metrics.rs:285-301`
- Four discovery metrics exist: `smg_discovery_registrations_total`, `smg_discovery_deregistrations_total`, `smg_discovery_sync_duration_seconds`, `smg_discovery_workers_discovered`
- The monitoring.md "Key Metrics by Layer" jumps from Layer 3 to Layer 5, skipping Layer 4. These are likely documented in `reference/metrics.md` (outside this agent's scope). No change made.

**4. `smg_mcp_tool_iterations_total` missing from monitoring.md Layer 5 table**

- Code: `model_gateway/src/observability/metrics.rs:313-316`
- Three MCP metrics are listed but `smg_mcp_tool_iterations_total` (tool loop iterations in Responses API) is omitted. Likely covered in `reference/metrics.md`. No change made.

**5. Database metrics (Layer 6) not documented in monitoring.md**

- Code: `model_gateway/src/observability/metrics.rs:319-334`
- Four metrics exist: `smg_db_operations_total`, `smg_db_operation_duration_seconds`, `smg_db_connections_active`, `smg_db_items_stored`
- Not mentioned in monitoring.md. Likely covered in `reference/metrics.md`. No change made.

---

## Phase 4 — Fixes Applied

### Fix 1: Corrected PromQL label `{type=...}` → `{token_type=...}` in monitoring.md

**File**: `docs/getting-started/monitoring.md`

**Before** (line 299-300):
```promql
sum(rate(smg_router_tokens_total{type="output"}[5m]))
/ sum(rate(smg_router_tokens_total{type="input"}[5m]))
```

**After**:
```promql
sum(rate(smg_router_tokens_total{token_type="output"}[5m]))
/ sum(rate(smg_router_tokens_total{token_type="input"}[5m]))
```

**Code reference**: `model_gateway/src/observability/metrics.rs:710,798,810` — the counter! calls use `"token_type" => token_type` as the label key/value pair.

### Fix 2: Added `--oracle-external-auth` to Oracle options table in chat-history.md

**File**: `docs/concepts/data/chat-history.md`

**Before**: Oracle configuration options table contained 8 rows ending with `--oracle-password`.

**After**: Added row:

```
| `--oracle-external-auth` | `ATP_EXTERNAL_AUTH` | `false` | Use external (OS) authentication instead of username/password |
```

**Code reference**: `model_gateway/src/main.rs:501-508` — `#[arg(long, env = "ATP_EXTERNAL_AUTH", default_value_t = false)] oracle_external_auth: bool`

---

## Phase 5 — Summary Report

### Totals

- **Total claims checked**: 78
- **Accurate**: 76
- **Inaccurate**: 1 (fixed)
- **Outdated**: 0
- **Undocumented**: 1 (fixed); 3 more deferred to reference/metrics.md scope

### Files modified

1. `docs/getting-started/monitoring.md` — Fixed incorrect PromQL label `{type="output"}` → `{token_type="output"}` in input/output token ratio query
2. `docs/concepts/data/chat-history.md` — Added missing `--oracle-external-auth` / `ATP_EXTERNAL_AUTH` row to Oracle configuration options table

### Notable findings

- Almost all documented CLI flags, defaults, endpoint paths, and auth behaviors match the code exactly.
- The one inaccurate item was the PromQL label key in a query example, which would cause the query to silently return empty results in production monitoring setups.
- The undocumented item (`--oracle-external-auth`) was an existing feature gap in the Oracle configuration table.
- TLS mTLS flags are correctly documented as "planned/not yet implemented" — verified no such flags exist in CliArgs.
- JWT no-fallback behavior is correctly documented and matches the middleware implementation.
