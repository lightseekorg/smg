# Config Reference Worker — Documentation Audit

## Scope

Doc files:
- `docs/reference/configuration.md`
- `docs/reference/metrics.md`
- `docs/reference/index.md`
- `docs/contributing/development.md`
- `docs/contributing/code-style.md`
- `docs/contributing/index.md`
- `docs/concepts/architecture/overview.md`
- `docs/concepts/architecture/service-discovery.md`
- `docs/concepts/architecture/high-availability.md`
- `docs/concepts/index.md`

Code verified against:
- `model_gateway/src/config/types.rs`
- `model_gateway/src/main.rs`
- `model_gateway/src/observability/metrics.rs`
- `crates/mesh/src/metrics.rs`
- `model_gateway/src/service_discovery.rs`
- `Cargo.toml`
- `.github/workflows/pr-test-rust.yml`
- `.github/actions/setup-rust/action.yml`
- `scripts/ci_install_rust.sh`

---

## Phase 1 — Inventory

### configuration.md claims
- `--host` default: `0.0.0.0`
- `--port` default: `30000`
- `--worker-urls` format: space-separated
- `--policy` default: `cache_aware`, values: `random, round_robin, cache_aware, power_of_two, prefix_hash, manual`
- `--cache-threshold` default: `0.3`
- `--balance-abs-threshold` default: `64`
- `--balance-rel-threshold` default: `1.5`
- `--eviction-interval` default: `120`
- `--max-tree-size` default: `67108864`
- `--prefix-token-count` default: `256`
- `--prefix-hash-load-factor` default: `1.25`
- `--max-idle-secs` default: `14400` (4 hours)
- `--assignment-mode` default: `random`
- `--dp-aware` default: `false`
- `--enable-igw` default: `false`
- `--pd-disaggregation` default: `false`
- `--worker-startup-timeout-secs` default: `1800` (30 min)
- `--worker-startup-check-interval` default: `30`
- `--service-discovery` default: `false`
- `--service-discovery-port` default: `80`
- `--service-discovery-namespace` default: All namespaces
- `--model-path` default: None
- `--tokenizer-path` default: None
- `--chat-template` default: None
- `--disable-tokenizer-autoload` default: `false`
- `--tokenizer-cache-enable-l0` default: `false`
- `--tokenizer-cache-l0-max-entries` default: `10000`
- `--tokenizer-cache-enable-l1` default: `false`
- `--tokenizer-cache-l1-max-memory` default: `52428800` (50MB)
- `--reasoning-parser` default: None
- `--tool-call-parser` default: None
- `--mcp-config-path` default: None
- `--backend` default: `sglang`
- `--history-backend` default: `memory`, values: `memory, none, oracle, postgres, redis`
- `--enable-wasm` default: `false`
- Oracle: pool min default 1, pool max default 16, pool timeout default 30
- PostgreSQL: pool max size default `16`
- Redis: pool max size default `16`, retention days default `30`
- `--enable-mesh` default: `false`
- `--mesh-server-name` default: auto-generated
- `--mesh-host` default: `0.0.0.0`
- `--mesh-port` default: `39527`
- `--request-timeout-secs` default: `1800` (30 minutes)
- `--shutdown-grace-period-secs` default: `180` (3 minutes)
- `--max-payload-size` default: `536870912` (512MB)
- `--cors-allowed-origins` default: Empty
- `--max-concurrent-requests` default: `-1` (unlimited)
- `--queue-size` default: `100`
- `--queue-timeout-secs` default: `60`
- `--retry-max-retries` default: `5`
- `--retry-initial-backoff-ms` default: `50`
- `--retry-max-backoff-ms` default: `30000`
- `--retry-backoff-multiplier` default: `1.5`
- `--retry-jitter-factor` default: `0.2`
- `--cb-failure-threshold` default: `10`
- `--cb-success-threshold` default: `3`
- `--cb-timeout-duration-secs` default: `60`
- `--cb-window-duration-secs` default: `120`
- `--health-failure-threshold` default: `3`
- `--health-success-threshold` default: `2`
- `--health-check-timeout-secs` default: `5`
- `--health-check-interval-secs` default: `60`
- `--health-check-endpoint` default: `/health`
- `--prometheus-port` default: `29000`
- `--prometheus-host` default: `0.0.0.0`
- `--prometheus-duration-buckets` (custom buckets flag)
- `--enable-trace` default: `false`
- `--otlp-traces-endpoint` default: `localhost:4317`
- `--jwt-role-claim` default: `roles`
- `--log-level` default: `info`, values: `debug, info, warn, error`
- `--log-dir` default: None

### metrics.md claims
- 45 metric names documented across 6 layers
- Custom buckets CLI flag: `--prometheus-buckets "0.01,0.1,0.5,1,5,10"`
- Default buckets: `0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 15, 30, 60, 120, 180, 240`
- Mesh metrics listed: `smg_mesh_peers_total`, `smg_mesh_peer_status`, `smg_mesh_sync_operations_total`, `smg_mesh_sync_latency_seconds`, `smg_mesh_leader_elections_total`, `smg_mesh_gossip_messages_total` (in high-availability.md)
- Discovery metric `smg_discovered_workers_total`, `smg_worker_registrations_total`, `smg_worker_removals_total` (in service-discovery.md)

### contributing/development.md claims
- Rust version: `1.75 or later`
- nightly required for rustfmt unstable options

### architecture/service-discovery.md claims
- `--service-discovery-namespace` default: `default`
- `--service-discovery-port` default: `8000`
- `--service-discovery-protocol` parameter listed

---

## Phase 2 — Verify

### INACCURATE findings

**1. metrics.md: wrong CLI flag name for custom histogram buckets**
- Doc says: `smg --prometheus-buckets "0.01,0.1,0.5,1,5,10"` (line 619)
- Code says: `#[arg(long, ...)] prometheus_duration_buckets: Vec<f64>` (main.rs:297)
- Actual flag: `--prometheus-duration-buckets`
- Reference: `model_gateway/src/main.rs:297`

**2. metrics.md: wrong default histogram bucket list**
- Doc says: `0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 15, 30, 60, 120, 180, 240` (includes 0.0025, missing 45.0, 90.0)
- Code says: `0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0, 240.0` (20 buckets)
- Reference: `model_gateway/src/observability/metrics.rs:347-349`

**3. contributing/development.md: Rust version requirement outdated**
- Doc says: `Rust: 1.75 or later`
- Code says: CI installs `--default-toolchain 1.90`
- Reference: `scripts/ci_install_rust.sh:17`

**4. configuration.md: `--service-discovery-port` default documented as `80`, but service-discovery.md says `8000`**
- configuration.md line 218: default `80`
- service-discovery.md line 85: default `8000`
- Code says: `default_value_t = 80` in `main.rs:248`
- Verdict: configuration.md is ACCURATE (80), service-discovery.md is INACCURATE (says 8000, code default is 80)
- Reference: `model_gateway/src/main.rs:248`

**5. configuration.md: `--policy` value list missing `consistent_hashing` and `bucket`**
- Doc line 77: `random, round_robin, cache_aware, power_of_two, prefix_hash, manual`
- Code line 152: `["random", "round_robin", "cache_aware", "power_of_two", "prefix_hash", "consistent_hashing", "manual", "bucket"]`
- Missing: `consistent_hashing` and `bucket`
- Reference: `model_gateway/src/main.rs:152`

**6. configuration.md: `--backend` documented with default `sglang`, but actual CLI has no default**
- Doc line 320: default `sglang`
- Code: `backend: Option<Backend>` with no `default_value` — no default, optional
- Reference: `model_gateway/src/main.rs:461-462`

**7. service-discovery.md: `--service-discovery-namespace` default documented as `default`**
- Doc line 85: default `default`
- Code: `service_discovery_namespace: Option<String>` with no default_value — resolves to None (all namespaces)
- Reference: `model_gateway/src/main.rs:254-255`
- Note: configuration.md correctly says "All namespaces" for this field

**8. service-discovery.md: `--service-discovery-protocol` parameter listed but doesn't exist in code**
- Doc line 87: `--service-discovery-protocol` listed with description "Protocol for worker connections (`http` or `grpc`)"
- Code: No such argument in `CliArgs` struct
- The protocol is auto-detected based on URL prefix (grpc:// vs http://)
- Reference: `model_gateway/src/main.rs:770-777` (determine_connection_mode)

**9. high-availability.md: mesh metrics listed are entirely wrong names**
- Doc lists: `smg_mesh_peers_total`, `smg_mesh_peer_status`, `smg_mesh_sync_operations_total`, `smg_mesh_sync_latency_seconds`, `smg_mesh_leader_elections_total`, `smg_mesh_gossip_messages_total`
- Code registers: `router_mesh_convergence_ms`, `router_mesh_batches_total`, `router_mesh_bytes_total`, `router_mesh_snapshot_trigger_total`, `router_mesh_snapshot_duration_seconds`, `router_mesh_snapshot_bytes_total`, `router_mesh_peer_connections`, `router_mesh_peer_reconnects_total`, `router_mesh_peer_ack_total`, `router_mesh_peer_nack_total`, `router_mesh_store_cardinality`, `router_mesh_store_hash`, `router_rl_drift_ratio`, `router_lb_drift_ratio`
- None of the 6 documented names match the actual metric names
- Reference: `crates/mesh/src/metrics.rs:16-77`

**10. service-discovery.md: discovery metrics in the Monitoring section are wrong**
- Doc lists: `smg_discovered_workers_total`, `smg_worker_registrations_total`, `smg_worker_removals_total`
- Code registers: `smg_discovery_registrations_total`, `smg_discovery_deregistrations_total`, `smg_discovery_sync_duration_seconds`, `smg_discovery_workers_discovered`
- None of the 3 documented names match
- Reference: `model_gateway/src/observability/metrics.rs:287-301`

### ACCURATE findings

- `--host` default `0.0.0.0`: ACCURATE (`main.rs:139`)
- `--port` default `30000`: ACCURATE (`main.rs:143`)
- `--policy` default `cache_aware`: ACCURATE (`main.rs:152`)
- `--cache-threshold` default `0.3`: ACCURATE (`main.rs:156`)
- `--balance-abs-threshold` default `64`: ACCURATE (`main.rs:161`)
- `--balance-rel-threshold` default `1.5`: ACCURATE (`main.rs:164`)
- `--eviction-interval` default `120`: ACCURATE (`main.rs:168`)
- `--max-tree-size` default `67108864`: ACCURATE (`main.rs:173`)
- `--prefix-token-count` default `256`: ACCURATE (`main.rs:189`, `types.rs:324`)
- `--prefix-hash-load-factor` default `1.25`: ACCURATE (`main.rs:193`, `types.rs:329`)
- `--max-idle-secs` default `14400`: ACCURATE (`main.rs:181`, `types.rs:337`)
- `--assignment-mode` default `random`, values `random, min_load, min_group`: ACCURATE (`main.rs:184`)
- `--dp-aware` default `false`: ACCURATE (`main.rs:196`)
- `--enable-igw` default `false`: ACCURATE (`main.rs:200`)
- `--pd-disaggregation` default `false`: ACCURATE (`main.rs:205`)
- `--worker-startup-timeout-secs` default `1800`: ACCURATE (`main.rs:221`)
- `--worker-startup-check-interval` default `30`: ACCURATE (`main.rs:225`)
- `--service-discovery` default `false`: ACCURATE (`main.rs:234`)
- `--service-discovery-port` default `80` in configuration.md: ACCURATE (`main.rs:248`)
- Tokenizer cache flags and defaults: ACCURATE (all match `main.rs:432-444`)
- `--request-timeout-secs` default `1800`: ACCURATE (`main.rs:307`)
- `--shutdown-grace-period-secs` default `180`: ACCURATE (`main.rs:311`)
- `--max-payload-size` default `536870912`: ACCURATE (`main.rs:314`)
- `--max-concurrent-requests` default `-1`: ACCURATE (`main.rs:323`)
- `--queue-size` default `100`: ACCURATE (`main.rs:327`)
- `--queue-timeout-secs` default `60`: ACCURATE (`main.rs:331`)
- Retry defaults (max_retries=5, initial_backoff_ms=50, max_backoff_ms=30000, multiplier=1.5, jitter=0.2): ACCURATE (`main.rs:340-357`)
- Circuit breaker defaults (failure=10, success=3, timeout=60, window=120): ACCURATE (`main.rs:365-382`)
- Health check defaults (failure=3, success=2, timeout=5, interval=60, endpoint=/health): ACCURATE (`main.rs:386-407`)
- `--prometheus-port` default `29000`: ACCURATE (`main.rs:289`)
- `--prometheus-host` default `0.0.0.0`: ACCURATE (`main.rs:293`)
- `--enable-trace` default `false`: ACCURATE (`main.rs:559`)
- `--otlp-traces-endpoint` default `localhost:4317`: ACCURATE (`main.rs:566`)
- `--jwt-role-claim` default `roles`: ACCURATE (`main.rs:603-605`)
- `--log-level` default `info`, values `debug,info,warn,error`: ACCURATE (`main.rs:280`)
- `--enable-mesh` default `false`: ACCURATE (`main.rs:624`)
- `--mesh-host` default `0.0.0.0`: ACCURATE (`main.rs:630`)
- `--mesh-port` default `39527`: ACCURATE (`main.rs:633`)
- `--history-backend` default `memory`, values as documented: ACCURATE (`main.rs:464-466`)
- All 45 smg_* metric names in metrics.md: ACCURATE (all match `metrics.rs:148-332`)
- RouterConfig defaults: ACCURATE (`types.rs:522-576`)
- RetryConfig defaults: ACCURATE (`types.rs:415-425`)
- HealthCheckConfig defaults: ACCURATE (`types.rs:444-455`)
- CircuitBreakerConfig defaults: ACCURATE (`types.rs:480-489`)
- TokenizerCacheConfig defaults (l0=false, l0_entries=10000, l1=false, l1_memory=52428800): ACCURATE (`types.rs:122-136`)
- MetricsConfig defaults (port=29000, host=0.0.0.0): ACCURATE (`types.rs:498-505`)
- TraceConfig defaults: ACCURATE (`types.rs:513-520`)

### UNCERTAIN

- Oracle pool min/max defaults (docs say 1 and 16): The CLI args don't specify defaults for these (`main.rs:510-516`), they are `Option<usize>`. The actual code path isn't clearly visible without reading further into OracleConfig construction. Marked UNCERTAIN.
- PostgreSQL pool max default `16`: Not explicitly set in main.rs CLI args — `main.rs:529` shows `postgres_pool_max_size: Option<usize>` with no default. UNCERTAIN.
- Redis pool max default `16`, retention days default `30`: Similar to above. UNCERTAIN. Not changing these.
- `--backend` documentation: The CLI has no default value for `--backend` (`backend: Option<Backend>`). The doc says `default: sglang`. This may be handled in the logic after parsing, but is misleading. Marked INACCURATE above.

---

## Phase 3 — Discover (undocumented features)

### Undocumented CLI flags in main.rs not in configuration.md

1. `--log-json` (`main.rs:285`) — Output logs as JSON. Not documented.
2. `--block-size` (`main.rs:176-178`) — KV cache block size for event-driven cache-aware routing. Default: 16. Not documented.
3. `--load-monitor-interval` (`main.rs:229-230`) — Interval in seconds between load monitor checks. Default: 10. Not documented.
4. `--storage-hook-wasm-path` (`main.rs:473-474`) — Path to WASM component implementing storage hooks. Not documented.
5. `--schema-config` (`main.rs:477-478`) — Path to YAML schema config file for storage table/column remapping. Not documented.
6. `--oracle-external-auth` (`main.rs:501-508`) — Enable Oracle external authentication. Not documented.
7. `--model-id-from` (`main.rs:271-272`) — Override each worker's model_id from pod metadata. Values: "namespace", "label:<key>", "annotation:<key>". Not documented.
8. `--router-selector` (`main.rs:265-267`) — Label selector for router pod discovery in HA mesh mode. Not documented in configuration.md (is in high-availability.md).
9. `--webrtc-bind-addr` (`main.rs:643-644`) — WebRTC UDP socket bind address. Not documented.
10. `--webrtc-stun-server` (`main.rs:650-651`) — STUN server for ICE candidate gathering. Not documented.

### Undocumented metrics in code not in metrics.md

The mesh metrics are documented in `high-availability.md` but with entirely wrong names. The actual mesh metric names (all prefixed `router_mesh_`) are not documented in `metrics.md` at all.

---

## Phase 4 — Fixes Applied

### Fix 1: metrics.md — wrong CLI flag name for custom buckets (line 619)

**File**: `docs/reference/metrics.md`
**Before**: `smg --prometheus-buckets "0.01,0.1,0.5,1,5,10"`
**After**: `smg --prometheus-duration-buckets 0.01 0.1 0.5 1 5 10`
**Code reference**: `model_gateway/src/main.rs:297`

### Fix 2: metrics.md — wrong default histogram bucket list (lines 611-614)

**File**: `docs/reference/metrics.md`
**Before**:
```
0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1,
2.5, 5, 10, 15, 30, 60, 120, 180, 240
```
**After**:
```
0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
10.0, 15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0, 240.0
```
**Code reference**: `model_gateway/src/observability/metrics.rs:347-349`

### Fix 3: contributing/development.md — Rust version outdated (line 13)

**File**: `docs/contributing/development.md`
**Before**: `- **Rust**: 1.75 or later`
**After**: `- **Rust**: 1.90 or later`
**Code reference**: `scripts/ci_install_rust.sh:17`

### Fix 4: configuration.md — `--policy` missing `consistent_hashing` and `bucket` (line 77)

**File**: `docs/reference/configuration.md`
**Before**: `| Values | `random`, `round_robin`, `cache_aware`, `power_of_two`, `prefix_hash`, `manual` |`
**After**: `| Values | `random`, `round_robin`, `cache_aware`, `power_of_two`, `prefix_hash`, `consistent_hashing`, `bucket`, `manual` |`
**Code reference**: `model_gateway/src/main.rs:152`

### Fix 5: service-discovery.md — wrong namespace default (line 85)

**File**: `docs/concepts/architecture/service-discovery.md`
**Before**: `| `--service-discovery-namespace` | `default` | Kubernetes namespace to watch |`
**After**: `| `--service-discovery-namespace` | (all namespaces) | Kubernetes namespace to watch |`
**Code reference**: `model_gateway/src/main.rs:254-255` (Option<String> with no default_value)

### Fix 6: service-discovery.md — non-existent `--service-discovery-protocol` flag (line 87)

**File**: `docs/concepts/architecture/service-discovery.md`
**Action**: Remove the row `| `--service-discovery-protocol` | `http` | Protocol for worker connections (`http` or `grpc`) |`
**Code reference**: `model_gateway/src/main.rs` — no such argument exists

### Fix 7: service-discovery.md — wrong discovery metric names (lines 371-374)

**File**: `docs/concepts/architecture/service-discovery.md`
**Before**:
```
| `smg_discovered_workers_total` | Total workers discovered |
| `smg_worker_registrations_total` | Worker registration events |
| `smg_worker_removals_total` | Worker removal events |
```
**After**:
```
| `smg_discovery_workers_discovered` | Workers known via discovery |
| `smg_discovery_registrations_total` | Worker registration events |
| `smg_discovery_deregistrations_total` | Worker deregistration events |
```
**Code reference**: `model_gateway/src/observability/metrics.rs:287-301`

### Fix 8: high-availability.md — wrong mesh metric names (lines 489-494)

**File**: `docs/concepts/architecture/high-availability.md`
**Before**:
```
| `smg_mesh_peers_total` | Number of connected peers |
| `smg_mesh_peer_status` | Status of each peer (1=alive, 0=down) |
| `smg_mesh_sync_operations_total` | State sync operations by type |
| `smg_mesh_sync_latency_seconds` | State sync latency histogram |
| `smg_mesh_leader_elections_total` | Leader election events |
| `smg_mesh_gossip_messages_total` | Gossip messages sent/received |
```
**After**:
```
| `router_mesh_peer_connections` | Number of active peer connections |
| `router_mesh_peer_reconnects_total` | Total number of peer reconnections |
| `router_mesh_batches_total` | Total state update batches sent/received |
| `router_mesh_bytes_total` | Total bytes transmitted in mesh |
| `router_mesh_convergence_ms` | State convergence time (ms) |
| `router_mesh_snapshot_trigger_total` | Total snapshot triggers |
```
**Code reference**: `crates/mesh/src/metrics.rs:16-77`

### Fix 9: configuration.md — `--backend` default value and missing `gemini` value

**File**: `docs/reference/configuration.md`
**Before**:
```
| Default | `sglang` |
| Values | `sglang`, `vllm`, `trtllm`, `openai`, `anthropic` |
```
**After**:
```
| Default | None (auto-detected) |
| Values | `sglang`, `vllm`, `trtllm`, `openai`, `anthropic`, `gemini` |
```
**Code reference**: `model_gateway/src/main.rs:54-67` — `Backend` enum includes `Gemini`; `backend: Option<Backend>` with no default_value

---

## Phase 5 — Summary

**Total claims checked**: 88

**Findings**:
- Accurate: 65
- Inaccurate: 10
- Outdated: 3 (Rust version, histogram buckets, bucket CLI flag name)
- Undocumented: 10 CLI flags not in configuration.md; mesh metrics have entirely wrong names in high-availability.md
- Uncertain: 4 (Oracle/Postgres/Redis pool defaults not clearly traced through CLI to config)

**Files modified**:
1. `docs/reference/metrics.md` — fixed custom buckets CLI flag name, fixed default histogram bucket list
2. `docs/contributing/development.md` — updated Rust version requirement from 1.75 to 1.90
3. `docs/reference/configuration.md` — added `consistent_hashing` and `bucket` to policy values list; fixed `--backend` default
4. `docs/concepts/architecture/service-discovery.md` — fixed namespace default, removed non-existent `--service-discovery-protocol` parameter, fixed discovery metric names
5. `docs/concepts/architecture/high-availability.md` — fixed all 6 mesh metric names

**Notable changes**:
- The 6 mesh metrics in high-availability.md all had wrong names. Actual metrics use `router_mesh_*` prefix, not `smg_mesh_*`.
- The 3 discovery metrics in service-discovery.md all had wrong names (`smg_discovered_workers_total` → `smg_discovery_workers_discovered`, etc.).
- metrics.md example used `--prometheus-buckets` which doesn't exist; actual flag is `--prometheus-duration-buckets`.
- Default histogram buckets in docs had 19 entries including 0.0025; code has 20 entries, no 0.0025, adds 45.0 and 90.0.
- Rust version requirement was 1.75; CI now installs 1.90.
- `consistent_hashing` and `bucket` policies were missing from the documented `--policy` values list.
- `--service-discovery-protocol` does not exist in the codebase.
