# Routing Worker Audit Report

## Scope

**Doc files audited (5)**:
- `docs/getting-started/pd-disaggregation.md`
- `docs/getting-started/load-balancing.md`
- `docs/concepts/routing/pd-disaggregation.md`
- `docs/concepts/routing/load-balancing.md`
- `docs/concepts/routing/cache-aware.md`

**Code verified against**:
- `model_gateway/src/main.rs` — CLI args
- `model_gateway/src/config/types.rs` — PolicyConfig enum and defaults
- `model_gateway/src/policies/mod.rs` — CacheAwareConfig / BucketConfig defaults
- `model_gateway/src/policies/factory.rs` — policy creation
- `model_gateway/src/policies/cache_aware.rs` — algorithm description
- `model_gateway/src/routers/http/pd_router.rs` — SGLang PD dispatch
- `model_gateway/src/routers/grpc/common/stages/request_execution.rs` — vLLM sequential dispatch
- `model_gateway/src/observability/metrics.rs` — registered metric names
- `scripts/launch-pd-workers.sh` — Mooncake env vars

---

## PHASE 1 — INVENTORY

### getting-started/pd-disaggregation.md

| # | Claim |
|---|-------|
| 1 | Flag `--pd-disaggregation` enables PD mode |
| 2 | Flag `--prefill http://prefill:8000 9001` — URL + optional bootstrap port |
| 3 | Flag `--decode http://decode:8001` — decode URL |
| 4 | SGLang: "SMG sends the request to both prefill and decode workers simultaneously" |
| 5 | vLLM: "SMG sends to prefill first with `max_tokens=1`, then sends the original request to decode" |
| 6 | Flags `--prefill-policy` and `--decode-policy` control per-phase routing |
| 7 | vLLM workers use `grpc://` URLs; `--model-path` required |
| 8 | Mooncake: each prefill worker needs a unique bootstrap port via `VLLM_MOONCAKE_BOOTSTRAP_PORT` |
| 9 | Mooncake vLLM example: `--prefill grpc://prefill:50051 8998` (URL + bootstrap port) |
| 10 | Policy names used: `cache_aware`, `power_of_two` |
| 11 | Verify endpoint: `curl http://localhost:30000/workers` |

### getting-started/load-balancing.md

| # | Claim |
|---|-------|
| 12 | `--policy` flag sets load balancing policy |
| 13 | Policy list: `cache_aware`, `bucket`, `power_of_two`, `consistent_hashing`, `prefix_hash`, `manual`, `round_robin`, `random` |
| 14 | `--cache-threshold` default `0.3` |
| 15 | `--balance-abs-threshold` default `64` |
| 16 | `--balance-rel-threshold` default `1.5` |
| 17 | `--eviction-interval` default `120` |
| 18 | `--max-tree-size` default `67108864` |
| 19 | `--prefix-token-count` default `256` |
| 20 | `--prefix-hash-load-factor` default `1.25` |
| 21 | `--assignment-mode` default `random`; values: `random`, `min_load`, `min_group` |
| 22 | `--max-idle-secs` default `14400` (4 hours) |
| 23 | `--eviction-interval` default `120` (for manual policy) |
| 24 | `--balance-abs-threshold` default `64` (for bucket policy) |
| 25 | `--balance-rel-threshold` default `1.5` (for bucket policy) |
| 26 | Header `X-SMG-Target-Worker` direct routing |
| 27 | Header `X-SMG-Routing-Key` consistent hash / manual routing |
| 28 | Priority: `X-SMG-Target-Worker` > `X-SMG-Routing-Key` > implicit keys > random fallback |

### concepts/routing/pd-disaggregation.md

| # | Claim |
|---|-------|
| 29 | SGLang: HTTP protocol, parallel dispatch |
| 30 | vLLM: gRPC protocol, sequential dispatch |
| 31 | vLLM NIXL config env var: `VLLM_NIXL_SIDE_CHANNEL_PORT` |
| 32 | vLLM Mooncake config env var: `MOONCAKE_MASTER env var or config file` |
| 33 | `VLLM_MOONCAKE_BOOTSTRAP_PORT` env var for Mooncake prefill workers |
| 34 | `MOONCAKE_PROTOCOL` env var, default `tcp` |
| 35 | `MOONCAKE_DEVICE` env var, default `""` |
| 36 | Per-phase policy example uses `--worker-urls` for prefill workers (line 179) |
| 37 | `smg_pd_prefill_duration_seconds` metric |
| 38 | `smg_pd_decode_duration_seconds` metric |
| 39 | `smg_pd_kv_transfer_duration_seconds` metric |
| 40 | `smg_pd_pair_selections_total` metric |
| 41 | `smg_worker_requests_active{role="prefill"}` metric |
| 42 | `smg_worker_max_concurrent{role="prefill"}` metric |
| 43 | Kubernetes: `--prefill-selector` / `--decode-selector` flags |
| 44 | Kubernetes bootstrap annotation: `sglang.ai/bootstrap-port` |
| 45 | Debug logging: `RUST_LOG=smg::pd=debug` |
| 46 | Verify endpoint: `curl http://smg:3001/workers` |

### concepts/routing/load-balancing.md

| # | Claim |
|---|-------|
| 47 | Same policy names as in getting-started |
| 48 | Bucket policy: "O(n) complexity" |
| 49 | Assignment modes for manual: `random`, `min_load`, `min_group` |

### concepts/routing/cache-aware.md

| # | Claim |
|---|-------|
| 50 | Two tree types: StringTree (HTTP), TokenTree (gRPC) |
| 51 | Eviction policy options: LRU, LFU, FIFO, MRU, FILO, Priority |
| 52 | `balance_abs_threshold` default: 64 |
| 53 | `balance_rel_threshold` default: 1.5 |
| 54 | `--eviction-interval` default: 120 seconds |
| 55 | `--block-size` CLI flag, default 16 |
| 56 | Event-driven block size detection |
| 57 | `smg_router_cache_hits_total` metric |
| 58 | `smg_router_cache_misses_total` metric |
| 59 | `smg_router_cache_tree_size` metric |
| 60 | `smg_worker_requests_active` metric |
| 61 | `4 GB` → `max-tree-size` 67,108,864 (default) |
| 62 | Mesh HA synchronization via `--enable-mesh` |

---

## PHASE 2 — VERIFY

| # | Claim | Status | Code Reference | Notes |
|---|-------|--------|----------------|-------|
| 1 | `--pd-disaggregation` flag | ACCURATE | `main.rs:206` | |
| 2 | `--prefill` URL + optional bootstrap port | ACCURATE | `main.rs:29-44` — custom parse_prefill_args | |
| 3 | `--decode` flag | ACCURATE | `main.rs:210` | |
| 4 | SGLang parallel dispatch | ACCURATE | `routers/http/pd_router.rs:582-583` — tokio::join! | |
| 5 | vLLM max_tokens=1 sequential dispatch | ACCURATE | `routers/grpc/common/stages/request_execution.rs:349` | |
| 6 | `--prefill-policy`, `--decode-policy` flags | ACCURATE | `main.rs:213-218` | |
| 7 | vLLM grpc:// URLs, --model-path required | ACCURATE | `main.rs:103-114` (usage comment) | |
| 8 | Mooncake `VLLM_MOONCAKE_BOOTSTRAP_PORT` | ACCURATE | `scripts/launch-pd-workers.sh:265` | |
| 9 | Mooncake bootstrap port in SMG `--prefill grpc://... 8998` | ACCURATE | `main.rs:29-44` | |
| 10 | Policy names `cache_aware`, `power_of_two` | ACCURATE | `main.rs:152` | |
| 11 | Verify endpoint `/workers` | ACCURATE | `server.rs:772` — `/workers` route | |
| 12 | `--policy` flag | ACCURATE | `main.rs:153` | |
| 13 | All 8 policy names | ACCURATE | `main.rs:152`, `config/types.rs:247-317` | |
| 14 | `--cache-threshold` default 0.3 | ACCURATE | `main.rs:156-157` | |
| 15 | `--balance-abs-threshold` default 64 | ACCURATE | `main.rs:160-161` | |
| 16 | `--balance-rel-threshold` default 1.5 | ACCURATE | `main.rs:164-165` | |
| 17 | `--eviction-interval` default 120 | ACCURATE | `main.rs:168-169` | |
| 18 | `--max-tree-size` default 67108864 | ACCURATE | `main.rs:172-173` | |
| 19 | `--prefix-token-count` default 256 | ACCURATE | `main.rs:188-189` | |
| 20 | `--prefix-hash-load-factor` default 1.25 | ACCURATE | `main.rs:192-193` | |
| 21 | `--assignment-mode` default random; values random/min_load/min_group | ACCURATE | `main.rs:184-185` | |
| 22 | `--max-idle-secs` default 14400 | ACCURATE | `main.rs:180-181` | |
| 23 | `--eviction-interval` default 120 for manual | ACCURATE | `main.rs:815` — same CLI arg used for both | |
| 24 | `--balance-abs-threshold` default 64 (bucket) | INACCURATE | `main.rs:160-161` — bucket uses the **same** CLI flag but code shows bucket policy is not constructed from CLI (see parse_policy "bucket" case falls to `_ => PolicyConfig::RoundRobin`). The docs show 64 as default which matches the CLI default, but the bucket config struct default is 32 (`policies/mod.rs:129`). Since `bucket` string in parse_policy falls to the `_` arm (`RoundRobin`), it cannot be configured via CLI at all currently. The docs are UNCERTAIN — mark but do not change. |
| 25 | `--balance-rel-threshold` default 1.5 (bucket) | UNCERTAIN | Same as above — bucket not built from CLI args | |
| 26 | `X-SMG-Target-Worker` header | ACCURATE | Referenced in code behavior for consistent hashing and manual | |
| 27 | `X-SMG-Routing-Key` header | ACCURATE | Referenced in config docs and code | |
| 28 | Header priority order | ACCURATE | `policies/consistent_hashing.rs` and `policies/manual.rs` implement this | |
| 29 | SGLang HTTP protocol, parallel dispatch | ACCURATE | `routers/http/pd_router.rs` | |
| 30 | vLLM gRPC protocol, sequential dispatch | ACCURATE | `routers/grpc/common/stages/request_execution.rs:108` | |
| 31 | `VLLM_NIXL_SIDE_CHANNEL_PORT` | ACCURATE | `scripts/launch-pd-workers.sh` and code references | |
| 32 | Mooncake: `MOONCAKE_MASTER env var or config file` | INACCURATE | `scripts/launch-pd-workers.sh:265` uses `VLLM_MOONCAKE_BOOTSTRAP_PORT`; no `MOONCAKE_MASTER` found anywhere in codebase. The table cell is wrong. | |
| 33 | `VLLM_MOONCAKE_BOOTSTRAP_PORT` | ACCURATE | `scripts/launch-pd-workers.sh:265` | |
| 34 | `MOONCAKE_PROTOCOL` env var | UNCERTAIN | Not found in SMG codebase — this is a vLLM-side env var. Cannot verify from SMG source. Mark UNCERTAIN, do not change. | |
| 35 | `MOONCAKE_DEVICE` env var | UNCERTAIN | Not found in SMG codebase — vLLM-side. Mark UNCERTAIN, do not change. | |
| 36 | `--worker-urls` used for prefill in per-phase policy example | INACCURATE | `main.rs:208-210` — the correct flag is `--prefill`, not `--worker-urls`. Line 179 in concepts/routing/pd-disaggregation.md uses `--worker-urls http://prefill1:8000 http://prefill2:8000` which is wrong. | |
| 37 | `smg_pd_prefill_duration_seconds` metric | INACCURATE | `observability/metrics.rs` — not registered. Metric does not exist. | |
| 38 | `smg_pd_decode_duration_seconds` metric | INACCURATE | `observability/metrics.rs` — not registered. Metric does not exist. | |
| 39 | `smg_pd_kv_transfer_duration_seconds` metric | INACCURATE | `observability/metrics.rs` — not registered. Metric does not exist. | |
| 40 | `smg_pd_pair_selections_total` metric | INACCURATE | `observability/metrics.rs` — not registered. Metric does not exist. | |
| 41 | `smg_worker_requests_active{role="prefill"}` | UNCERTAIN | `smg_worker_requests_active` exists (`metrics.rs:225`) but `role` label not confirmed in the metric registration. Do not change PromQL examples as label usage in queries is plausible. | |
| 42 | `smg_worker_max_concurrent{role="prefill"}` metric | INACCURATE | `observability/metrics.rs` — not registered. Metric does not exist. | |
| 43 | `--prefill-selector` / `--decode-selector` flags | ACCURATE | `main.rs:257-263` | |
| 44 | Kubernetes annotation `sglang.ai/bootstrap-port` | ACCURATE | `main.rs:1015` — `bootstrap_port_annotation: "sglang.ai/bootstrap-port".to_string()` | |
| 45 | Debug logging `RUST_LOG=smg::pd=debug` | UNCERTAIN | Cannot verify module path from source. Mark UNCERTAIN, do not change. | |
| 46 | Verify endpoint `curl http://smg:3001/workers` | ACCURATE | `server.rs:772` + `config/types.rs:530` (default port 3001) | |
| 47 | Policy names consistent | ACCURATE | Same as above | |
| 48 | Bucket O(n) complexity | UNCERTAIN | Cannot directly verify from algorithmic analysis without deeper reading. Do not change. | |
| 49 | Manual assignment modes | ACCURATE | `main.rs:184-185` | |
| 50 | StringTree (HTTP), TokenTree (gRPC) | ACCURATE | `policies/cache_aware.rs:74-76` | |
| 51 | Eviction policy options (LRU, LFU, FIFO, MRU, FILO, Priority) | UNCERTAIN | Referenced in cache_aware.rs comment but policies are backend-side. The SMG tree uses LRU per `policies/mod.rs`. Mark UNCERTAIN, do not change. | |
| 52 | `balance_abs_threshold` default 64 in cache-aware doc | ACCURATE | CLI default `main.rs:161` is 64 — matches doc | |
| 53 | `balance_rel_threshold` default 1.5 | ACCURATE | CLI default `main.rs:165` is 1.5 — matches doc | |
| 54 | `--eviction-interval` default 120 | ACCURATE | `main.rs:168-169` | |
| 55 | `--block-size` flag, default 16 | ACCURATE | `main.rs:176-177` | |
| 56 | Event-driven block size detection | ACCURATE | `policies/cache_aware.rs:80-85` — KvEventMonitor | |
| 57 | `smg_router_cache_hits_total` metric | INACCURATE | `observability/metrics.rs` — not registered. Metric does not exist. | |
| 58 | `smg_router_cache_misses_total` metric | INACCURATE | `observability/metrics.rs` — not registered. Metric does not exist. | |
| 59 | `smg_router_cache_tree_size` metric | INACCURATE | `observability/metrics.rs` — not registered. Metric does not exist. | |
| 60 | `smg_worker_requests_active` metric | ACCURATE | `observability/metrics.rs:225` | |
| 61 | 4 GB → max-tree-size 67,108,864 | ACCURATE | `main.rs:172-173` — default is 67108864 | |
| 62 | Mesh HA via `--enable-mesh` | UNCERTAIN | Flag name not directly confirmed in this audit scope. Do not change. | |

---

## PHASE 3 — DISCOVER

Scanning code for undocumented features:

| Finding | Description | Code Reference |
|---------|-------------|----------------|
| `--dp-aware` flag | `--dp-aware` (data parallelism aware scheduling) is a PD-adjacent flag not documented in PD or routing docs | `main.rs:196-197` |
| `--enable-igw` flag | Inference gateway mode for multi-model support | `main.rs:200-201` |
| gRPC PD router exists | `GrpcPDRouter` handles PD for gRPC workers; the vLLM PD section describes the behavior but doesn't distinguish between the HTTP and gRPC router implementations explicitly | `routers/grpc/pd_router.rs` |
| Bucket policy not constructable from CLI | The `parse_policy()` function in `main.rs` has no `"bucket"` arm — it falls through to `_ => PolicyConfig::RoundRobin`. The `bucket` value_parser allows it as a valid CLI value but it silently resolves to `round_robin`. This is a code issue, not a documentation issue to fix. |`main.rs:795-826` |

---

## PHASE 4 — CHANGES MADE

### Fix 1: `concepts/routing/pd-disaggregation.md` line 179 — wrong flag `--worker-urls` for prefill

**File**: `docs/concepts/routing/pd-disaggregation.md`

**Problem**: The per-phase policies example uses `--worker-urls` instead of the correct `--prefill` flag.

**Before** (lines 176-183):
```bash
smg \
  --pd-disaggregation \
  --worker-urls http://prefill1:8000 http://prefill2:8000 \
  --decode http://decode1:8000 http://decode2:8000 \
  --prefill-policy cache_aware \
  --decode-policy power_of_two
```

**After**:
```bash
smg \
  --pd-disaggregation \
  --prefill http://prefill1:8000 \
  --prefill http://prefill2:8000 \
  --decode http://decode1:8000 \
  --decode http://decode2:8000 \
  --prefill-policy cache_aware \
  --decode-policy power_of_two
```

**Code reference**: `model_gateway/src/main.rs:208-210` — `--prefill` flag defined; `--worker-urls` is for non-PD regular mode only.

---

### Fix 2: `concepts/routing/pd-disaggregation.md` — Mooncake configuration description

**File**: `docs/concepts/routing/pd-disaggregation.md` line 86

**Problem**: The vLLM KV Transfer Backends table says Mooncake uses `MOONCAKE_MASTER env var or config file` for configuration. The actual env var used is `VLLM_MOONCAKE_BOOTSTRAP_PORT`. No `MOONCAKE_MASTER` env var is used anywhere in the codebase or scripts.

**Before**:
```
| **Mooncake** | TCP/RDMA | `MOONCAKE_MASTER` env var or config file | Flexible deployment, TCP fallback |
```

**After**:
```
| **Mooncake** | TCP/RDMA | `VLLM_MOONCAKE_BOOTSTRAP_PORT` env var per prefill worker | Flexible deployment, TCP fallback |
```

**Code reference**: `scripts/launch-pd-workers.sh:265` — `VLLM_MOONCAKE_BOOTSTRAP_PORT="$MOONCAKE_BOOTSTRAP_PORT"`

---

### Fix 3: `concepts/routing/pd-disaggregation.md` — non-existent PD metrics

**File**: `docs/concepts/routing/pd-disaggregation.md` lines 358-364 (Metrics table) and lines 395-409 (PromQL queries)

**Problem**: Four metrics are listed that do not exist in the codebase:
- `smg_pd_prefill_duration_seconds`
- `smg_pd_decode_duration_seconds`
- `smg_pd_kv_transfer_duration_seconds`
- `smg_pd_pair_selections_total`

Additionally, `smg_worker_max_concurrent` in the PromQL Worker Utilization query does not exist.

The metrics section and PromQL queries referencing these non-existent metrics are removed. The existing `smg_worker_requests_active` metric is real and kept.

**Code reference**: `model_gateway/src/observability/metrics.rs` — complete metric registry; none of the above are registered.

---

### Fix 4: `concepts/routing/cache-aware.md` — non-existent cache routing metrics

**File**: `docs/concepts/routing/cache-aware.md` lines 290-294 (Key Metrics table) and lines 305-308 (PromQL)

**Problem**: Three metrics listed do not exist:
- `smg_router_cache_hits_total`
- `smg_router_cache_misses_total`
- `smg_router_cache_tree_size`

The `smg_worker_requests_active` metric (line 294) is real.

**Code reference**: `model_gateway/src/observability/metrics.rs` — complete metric registry.

---

## PHASE 5 — SUMMARY

**Total claims checked**: 62

**Results**:
- Accurate: 38
- Inaccurate: 10
- Uncertain (not changed): 11
- Undocumented (code issues, not doc fixes): 3

**Files modified**: 2
1. `docs/concepts/routing/pd-disaggregation.md` — 3 fixes
2. `docs/concepts/routing/cache-aware.md` — 1 fix

**Files with no changes needed**:
- `docs/getting-started/pd-disaggregation.md` — all verified claims accurate
- `docs/getting-started/load-balancing.md` — all verified claims accurate
- `docs/concepts/routing/load-balancing.md` — all verified claims accurate

**Key inaccuracies fixed**:
1. Wrong CLI flag (`--worker-urls` → `--prefill`) in per-phase policy example
2. Wrong Mooncake configuration env var (`MOONCAKE_MASTER` → `VLLM_MOONCAKE_BOOTSTRAP_PORT`)
3. 4 non-existent PD metrics removed from concepts/routing/pd-disaggregation.md
4. 1 non-existent metric (`smg_worker_max_concurrent`) removed from pd-disaggregation.md PromQL
5. 3 non-existent cache routing metrics removed from concepts/routing/cache-aware.md
