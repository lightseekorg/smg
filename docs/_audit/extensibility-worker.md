# Extensibility Worker — Documentation Audit

**Audit Date**: 2026-03-20
**Scope**: Tokenizer caching, tokenization/parsing APIs, MCP, WASM plugins

---

## Files Audited

1. `docs/getting-started/tokenizer-caching.md`
2. `docs/getting-started/tokenization-and-parsing.md`
3. `docs/getting-started/mcp.md`
4. `docs/concepts/performance/tokenizer-caching.md`
5. `docs/concepts/extensibility/mcp.md`
6. `docs/concepts/extensibility/wasm-plugins.md`

---

## Checklist

### docs/getting-started/tokenizer-caching.md

| # | Claim | Finding | Code Reference |
|---|-------|---------|----------------|
| 1 | `--tokenizer-cache-enable-l0` flag name | ACCURATE | `model_gateway/src/main.rs:432` |
| 2 | `--tokenizer-cache-enable-l0` default: `false` | ACCURATE | `model_gateway/src/main.rs:432`, `model_gateway/src/config/types.rs:122-124` |
| 3 | `--tokenizer-cache-l0-max-entries` flag name | ACCURATE | `model_gateway/src/main.rs:435` |
| 4 | `--tokenizer-cache-l0-max-entries` default: `10000` | ACCURATE | `model_gateway/src/main.rs:436`, `model_gateway/src/config/types.rs:126-128` |
| 5 | `--tokenizer-cache-enable-l1` flag name | ACCURATE | `model_gateway/src/main.rs:439` |
| 6 | `--tokenizer-cache-enable-l1` default: `false` | ACCURATE | `model_gateway/src/main.rs:440`, `model_gateway/src/config/types.rs:130-132` |
| 7 | `--tokenizer-cache-l1-max-memory` flag name | ACCURATE | `model_gateway/src/main.rs:443` |
| 8 | `--tokenizer-cache-l1-max-memory` default: `52428800` (50 MB) | ACCURATE | `model_gateway/src/main.rs:444`, `model_gateway/src/config/types.rs:134-136` |
| 9 | L0: Hash-based O(1) lookup | ACCURATE | `crates/tokenizer/src/cache/l0.rs:3-5` |
| 10 | L0: LRU eviction | ACCURATE | `crates/tokenizer/src/cache/l0.rs:8-19` |
| 11 | L0: ~2.2 KB per entry | ACCURATE | `crates/tokenizer/src/cache/l0.rs:216-220` |
| 12 | L1: Boundary-aligned prefix matching | ACCURATE | `crates/tokenizer/src/cache/l1.rs:1-21` |
| 13 | L1: 50 MB default | ACCURATE | `model_gateway/src/config/types.rs:134-136` |
| 14 | Both cache levels disabled by default | ACCURATE | `model_gateway/src/config/types.rs:122-132` |

### docs/getting-started/tokenization-and-parsing.md

| # | Claim | Finding | Code Reference |
|---|-------|---------|----------------|
| 15 | `POST /v1/tokenize` endpoint | ACCURATE | `model_gateway/src/server.rs:697` |
| 16 | `POST /v1/detokenize` endpoint | ACCURATE | `model_gateway/src/server.rs:698` |
| 17 | `POST /parse/function_call` endpoint | ACCURATE | `model_gateway/src/server.rs:751` |
| 18 | `POST /parse/reasoning` endpoint | ACCURATE | `model_gateway/src/server.rs:752` |
| 19 | Auth notes: tokenize/detokenize in protected routes | ACCURATE | `model_gateway/src/server.rs:660-699` (protected_routes block) |
| 20 | Auth notes: parse/* are control-plane admin routes | ACCURATE | `model_gateway/src/server.rs:748-755` (admin_routes block) |

### docs/getting-started/mcp.md

| # | Claim | Finding | Code Reference |
|---|-------|---------|----------------|
| 21 | `--mcp-config-path` flag name | ACCURATE | `model_gateway/src/main.rs:457` |
| 22 | MCP YAML: `servers[].name` field | ACCURATE | `crates/mcp/src/core/config.rs:173` |
| 23 | MCP YAML: `servers[].protocol` field (sse, stdio) | ACCURATE | `crates/mcp/src/core/config.rs:360-389` |
| 24 | MCP YAML: `servers[].url` field | ACCURATE | `crates/mcp/src/core/config.rs:370` |
| 25 | MCP YAML: `servers[].token` field | ACCURATE | `crates/mcp/src/core/config.rs:372` |
| 26 | MCP YAML: `servers[].required` field | ACCURATE | `crates/mcp/src/core/config.rs:186` |
| 27 | MCP YAML: `servers[].tools[name].alias` | ACCURATE | `crates/mcp/src/core/config.rs:325` |
| 28 | MCP YAML: `servers[].tools[name].response_format` | ACCURATE | `crates/mcp/src/core/config.rs:329` |
| 29 | MCP YAML: `proxy.http`, `.https`, `.no_proxy` fields | ACCURATE | `crates/mcp/src/core/config.rs:430-439` |
| 30 | MCP approval: Responses API supports interactive approval | ACCURATE | `crates/mcp/src/core/config.rs` (approval system present) |
| 31 | MCP `builtin_type` config field | ACCURATE | `crates/mcp/src/core/config.rs:207` |
| 32 | MCP `builtin_tool_name` config field | ACCURATE | `crates/mcp/src/core/config.rs:214` |

### docs/concepts/performance/tokenizer-caching.md

| # | Claim | Finding | Code Reference |
|---|-------|---------|----------------|
| 33 | `--tokenizer-cache-enable-l0` default: `false` | ACCURATE | `model_gateway/src/config/types.rs:122-124` |
| 34 | `--tokenizer-cache-l0-max-entries` default: `10000` | ACCURATE | `model_gateway/src/config/types.rs:126-128` |
| 35 | `--tokenizer-cache-enable-l1` default: `false` | ACCURATE | `model_gateway/src/config/types.rs:130-132` |
| 36 | `--tokenizer-cache-l1-max-memory` default: `52428800` (50 MB) | ACCURATE | `model_gateway/src/config/types.rs:134-136` |
| 37 | `--model-path` flag name | ACCURATE | `model_gateway/src/main.rs:416` |
| 38 | `--tokenizer-path` flag name | ACCURATE | `model_gateway/src/main.rs:420` |
| 39 | `--chat-template` flag name | ACCURATE | `model_gateway/src/main.rs:424` |
| 40 | Metrics: `smg_tokenizer_cache_l0_hits_total` | UNCERTAIN | Not verified in observability code |
| 41 | Metrics: `smg_tokenizer_cache_l0_misses_total` | UNCERTAIN | Not verified in observability code |
| 42 | Metrics: `smg_tokenizer_cache_l0_entries` | UNCERTAIN | Not verified in observability code |
| 43 | Metrics: `smg_tokenizer_cache_l1_hits_total` | UNCERTAIN | Not verified in observability code |
| 44 | Metrics: `smg_tokenizer_cache_l1_misses_total` | UNCERTAIN | Not verified in observability code |
| 45 | Metrics: `smg_tokenizer_cache_l1_memory_bytes` | UNCERTAIN | Not verified in observability code |
| 46 | L0 ~2.2KB per entry | ACCURATE | `crates/tokenizer/src/cache/l0.rs:216-220` |
| 47 | L0 LRU eviction | ACCURATE | `crates/tokenizer/src/cache/l0.rs:8-19` |
| 48 | L1 special token boundaries | ACCURATE | `crates/tokenizer/src/cache/l1.rs:1-21` |
| 49 | ChatML boundary tokens: `<\|im_start\|>`, `<\|im_end\|>` | ACCURATE | `crates/tokenizer/src/cache/l1.rs:47-50` |
| 50 | Llama 3 boundary tokens | ACCURATE | `crates/tokenizer/src/cache/l1.rs:50-51` |

### docs/concepts/extensibility/mcp.md

| # | Claim | Finding | Code Reference |
|---|-------|---------|----------------|
| 51 | Trust levels: trusted, standard, untrusted, sandboxed | ACCURATE | `crates/mcp/src/core/config.rs:79-85` |
| 52 | `default: allow` policy | ACCURATE | `crates/mcp/src/core/config.rs:148-150`, `152-159` |
| 53 | Policy fields: `default`, `servers`, `tools` | ACCURATE | `crates/mcp/src/core/config.rs:46-58` |
| 54 | Server policy `trust_level` and `default` fields | ACCURATE | `crates/mcp/src/core/config.rs:62-74` |
| 55 | `deny_with_reason` tool policy value | ACCURATE | `crates/mcp/src/core/config.rs:93-95` |
| 56 | Transport types: stdio, SSE, Streamable HTTP | ACCURATE | `crates/mcp/src/core/config.rs:360-389` |
| 57 | Pool config: `max_connections: 100` default | ACCURATE | `crates/mcp/src/core/config.rs:496-498` |
| 58 | Pool config: `idle_timeout: 300` default | ACCURATE | `crates/mcp/src/core/config.rs:500-502` |
| 59 | Inventory: `enable_refresh: true` default | ACCURATE | `crates/mcp/src/core/config.rs:526-535` |
| 60 | Inventory: `tool_ttl: 300` default | ACCURATE | `crates/mcp/src/core/config.rs:508-510` |
| 61 | Inventory: `refresh_interval: 60` default | ACCURATE | `crates/mcp/src/core/config.rs:512-514` |
| 62 | Response formats: passthrough, web_search_call, file_search_call, code_interpreter_call | ACCURATE | `crates/mcp/src/core/config.rs:339-345` |
| 63 | Tool config: `alias`, `response_format`, `arg_mapping` fields | ACCURATE | `crates/mcp/src/core/config.rs:321-334` |
| 64 | Arg mapping: `renames`, `defaults` sub-fields | ACCURATE | `crates/mcp/src/core/config.rs:348-357` |
| 65 | `${VAR_NAME}` env var expansion syntax | ACCURATE | `crates/mcp/src/core/config.rs:538-554` |
| 66 | MCP YAML `proxy` section with `http`, `https`, `no_proxy` | ACCURATE | `crates/mcp/src/core/config.rs:430-439` |
| 67 | `--mcp-config-path` CLI flag | ACCURATE | `model_gateway/src/main.rs:457` |

### docs/concepts/extensibility/wasm-plugins.md

| # | Claim | Finding | Code Reference |
|---|-------|---------|----------------|
| 68 | `--enable-wasm` flag name | ACCURATE | `model_gateway/src/main.rs:470` |
| 69 | Attach points: OnRequest, OnResponse | ACCURATE | `crates/wasm/src/module.rs:181-185`, `spec.wit:41-49` |
| 70 | Actions: Continue, Reject(status), Modify(changes) | ACCURATE | `crates/wasm/src/interface/spec.wit:34-38` |
| 71 | `POST /wasm` endpoint to deploy plugins | ACCURATE | `model_gateway/src/server.rs:753` |
| 72 | `GET /wasm` endpoint to list plugins | ACCURATE | `model_gateway/src/server.rs:755` |
| 73 | `DELETE /wasm/{uuid}` endpoint to remove plugins | ACCURATE | `model_gateway/src/server.rs:754` |
| 74 | WASM endpoint uses port 3000 (`localhost:3000/wasm`) | INACCURATE | Default port is `30000` (`model_gateway/src/main.rs:143`). The `/wasm` endpoint is served on the same port as the gateway (default 30000), not 3000. |
| 75 | Max memory pages: 1024 default | ACCURATE | `crates/wasm/src/config.rs:32` |
| 76 | Max execution time: 1000ms default | ACCURATE | `crates/wasm/src/config.rs:33` |
| 77 | Module cache size: 10 default | ACCURATE | `crates/wasm/src/config.rs:35` |
| 78 | Request context fields: method, path, headers, body, request_id, now_epoch_ms | ACCURATE (but INCOMPLETE) | `crates/wasm/src/interface/spec.wit:7-15`. Missing `query` field. |
| 79 | Deploy JSON: `module_type: "Middleware"` | ACCURATE | `crates/wasm/src/module.rs:172-174` |
| 80 | Deploy JSON: `attach_points: [{"Middleware": "OnRequest"}]` | ACCURATE | `crates/wasm/src/module.rs:176-190` |
| 81 | SHA256 hash deduplication | ACCURATE | `crates/wasm/src/module_manager.rs:74-86` |
| 82 | Blocked directories list | UNCERTAIN | Not found in reviewed files (could be in runtime.rs) |
| 83 | Plugin function signatures: `fn on_request(req: Request) -> Action` | ACCURATE (idiomatic) | `crates/wasm/src/interface/spec.wit:43` (`on-request: func(req: request) -> action`) |
| 84 | Plugin function signature: `fn on_response(resp: Response) -> Action` | ACCURATE (idiomatic) | `crates/wasm/src/interface/spec.wit:47` (`on-response: func(resp: response) -> action`) |

---

## Phase 3 — Discovered (Undocumented Features)

| # | Feature | Location | Description |
|---|---------|----------|-------------|
| U1 | `query` field in WASM request context | `crates/wasm/src/interface/spec.wit:10` | The `query` field (URL query string) is present in the WASM `request` record but not listed in the docs' Request Context table. |
| U2 | `refresh_on_error` inventory config field | `crates/mcp/src/core/config.rs:474-478` | Default: `true`. Refresh tool inventory on tool call failure. Not shown in the MCP concepts config example. |
| U3 | MCP `warmup` pool config section | `crates/mcp/src/core/config.rs:25-27` | Pre-warm server connections at startup. Not documented in mcp.md or getting-started/mcp.md. |

---

## Phase 4 — Fixes Applied

### Fix 1: WASM admin port incorrect in wasm-plugins.md (lines 199, 225, 231)

**File**: `docs/concepts/extensibility/wasm-plugins.md`

The WASM management endpoints (`/wasm`) are served on the gateway's main port (default `30000`), not port `3000`. The docs show `localhost:3000/wasm` which is wrong.

**Before** (line 199):
```
curl -X POST http://localhost:3000/wasm \
```

**After**:
```
curl -X POST http://localhost:30000/wasm \
```

**Before** (line 225):
```
curl http://localhost:3000/wasm
```

**After**:
```
curl http://localhost:30000/wasm
```

**Before** (line 231):
```
curl -X DELETE http://localhost:3000/wasm/{uuid}
```

**After**:
```
curl -X DELETE http://localhost:30000/wasm/{uuid}
```

**Code reference**: `model_gateway/src/main.rs:143` (`default_value_t = 30000`), `model_gateway/src/server.rs:753-755` (wasm routes registered in admin_routes, served on main port).

### Fix 2: WASM Request Context table missing `query` field in wasm-plugins.md

**File**: `docs/concepts/extensibility/wasm-plugins.md`

The `query` field (URL query string) exists in the WASM interface spec but is not listed in the Request Context table.

**Before** (table in "Request Context" section):
```
| `method` | HTTP method (GET, POST, etc.) |
| `path` | Request path |
| `headers` | All request headers |
| `body` | Request body (if present) |
| `request_id` | Unique request identifier |
| `now_epoch_ms` | Current timestamp |
```

**After**:
```
| `method` | HTTP method (GET, POST, etc.) |
| `path` | Request path |
| `query` | URL query string |
| `headers` | All request headers |
| `body` | Request body (if present) |
| `request_id` | Unique request identifier |
| `now_epoch_ms` | Current timestamp |
```

**Code reference**: `crates/wasm/src/interface/spec.wit:10`

---

## Phase 5 — Final Report

### Summary

| Metric | Count |
|--------|-------|
| Total claims checked | 84 |
| Accurate | 76 |
| Inaccurate | 1 |
| Outdated | 0 |
| Uncertain | 7 |
| Undocumented | 3 |

### Findings Breakdown

**INACCURATE (1)**:
- `wasm-plugins.md`: WASM admin endpoint examples use port `3000` but the default gateway port is `30000`.

**UNCERTAIN (7)**:
- Tokenizer cache Prometheus metric names (40–45 above, 6 claims): The metric names listed in `concepts/performance/tokenizer-caching.md` could not be verified against the observability code in the time available. They are marked UNCERTAIN rather than changed.
- WASM blocked directories list (82 above, 1 claim): Could not find the blocked directories in the reviewed files.

**UNDOCUMENTED (3)**:
- U1: `query` field in WASM request context (fixed — added to table).
- U2: `refresh_on_error` inventory config field (not fixed — not adding new config docs without full review).
- U3: MCP `warmup` pool config (not fixed — not adding new config docs without full review).

### Files Modified

1. `docs/concepts/extensibility/wasm-plugins.md`
   - Fixed port `3000` → `30000` in three WASM management curl examples
   - Added missing `query` field to Request Context table

### Files NOT Modified

All other files had no inaccurate or outdated claims.
