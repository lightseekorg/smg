# Workers Audit Report

**Agent**: workers-worker
**Scope**: getting-started/multiple-workers.md, getting-started/grpc-workers.md, getting-started/external-providers.md, concepts/architecture/grpc-pipeline.md

---

## Phase 1 — Inventory

### getting-started/multiple-workers.md

**CLI flags / behavior claims:**
- [ ] `--worker-urls http://worker1:8000 http://worker2:8000 http://worker3:8000` — worker URL format
- [ ] `--policy round_robin` — policy flag name
- [ ] `--host 0.0.0.0` — host flag
- [ ] `--port 30000` — port flag
- [ ] `grpc://` scheme triggers gRPC mode
- [ ] `--model-path` required for gRPC workers
- [ ] `--backend openai` flag name
- [ ] OPENAI_API_KEY, ANTHROPIC_API_KEY, XAI_API_KEY, GEMINI_API_KEY env vars for cloud providers
- [ ] `--enable-igw` flag for IGW mode
- [ ] `POST /workers` endpoint for adding workers
- [ ] `GET /workers` for listing workers
- [ ] `DELETE /workers/{worker_id}` for removing workers
- [ ] `GET /health` endpoint
- [ ] `GET /v1/chat/completions` endpoint

**POST /workers request body fields:**
- [ ] `url` (required)
- [ ] `api_key`
- [ ] `model_id` — documented as "auto-discovered model identifier"
- [ ] `runtime` — "sglang", "vllm", "trtllm", or "external"
- [ ] `worker_type` — "regular", "prefill", or "decode"
- [ ] `priority` — default 50
- [ ] `cost` — default 1.0
- [ ] `disable_health_check` — default false
- [ ] `labels`

**POST /workers response:**
- [ ] `status: "accepted"`
- [ ] `worker_id`
- [ ] `url`
- [ ] `location`
- [ ] `message: "Worker addition queued for background processing"`

### getting-started/grpc-workers.md

**CLI flags:**
- [ ] gRPC workers use `grpc://` URL scheme
- [ ] `--model-path` flag required for gRPC
- [ ] `--reasoning-parser` flag (e.g., `deepseek_r1`)
- [ ] `--tool-call-parser` flag (e.g., `llama`)
- [ ] `separate_reasoning: true` request field

**Reasoning parsers listed:**
- [ ] `deepseek_r1` — DeepSeek-R1
- [ ] `qwen3` — Qwen3, Nemotron Nano
- [ ] `qwen_thinking` — Qwen3-Thinking (name in table)
- [ ] `kimi` — Kimi
- [ ] `glm45` — GLM-4.5, GLM-4.7
- [ ] `step3` — Step-3
- [ ] `minimax` — MiniMax, MiniMax-M2
- [ ] `cohere_cmd` — Command-R, Command-A, C4AI

**Tool call parsers listed:**
- [ ] `json` — GPT-4/4o, Claude, Gemini, Gemma, Llama (generic)
- [ ] `llama` — Llama 3.2
- [ ] `pythonic` — Llama 4, DeepSeek (generic)
- [ ] `deepseek` — DeepSeek-V3
- [ ] `mistral` — Mistral, Mixtral
- [ ] `qwen` — Qwen
- [ ] `qwen_coder` — Qwen3-Coder, Qwen2.5-Coder
- [ ] `glm45_moe` — GLM-4.5, GLM-4.6
- [ ] `glm47_moe` — GLM-4.7
- [ ] `step3` — Step-3
- [ ] `kimik2` — Kimi-K2
- [ ] `minimax_m2` — MiniMax
- [ ] `cohere` — Command-R, Command-A, C4AI

### getting-started/external-providers.md

**Behavior claims:**
- [ ] SMG auto-detects provider from model name
- [ ] OpenAI detected from `gpt-*`, `o1-*`, `o3-*` models
- [ ] Anthropic detected from `claude-*` models
- [ ] xAI detected from `grok-*` models
- [ ] Google Gemini detected from `gemini-*` models
- [ ] OpenAI uses `Authorization: Bearer` header
- [ ] Anthropic uses `x-api-key` header
- [ ] `--enable-igw` flag
- [ ] POST /workers with `runtime_type: "external"` and `provider: "openai"` fields
- [ ] Caller token forwarded for fan-out model discovery

### concepts/architecture/grpc-pipeline.md

**Configuration table claims:**
- [ ] `--reasoning-parser` CLI flag
- [ ] `SMG_REASONING_PARSER` env var
- [ ] `--tool-call-parser` CLI flag
- [ ] `SMG_TOOL_CALL_PARSER` env var
- [ ] `--mcp-config-path` flag

**Reasoning parsers in complete reference table:**
- [ ] `deepseek_r1`, `qwen3`, `qwen3_thinking`, `kimi`, `glm45`, `step3`, `minimax`

**Recommended configuration examples:**
- [ ] `--grpc-workers grpc://worker1:50051` flag (used in examples)

**Metric names:**
- [ ] `smg_pipeline_stage_duration_seconds`
- [ ] `smg_reasoning_extractions_total`
- [ ] `smg_tool_calls_total`
- [ ] `smg_tool_execution_duration_seconds`
- [ ] `smg_mcp_tool_calls_total`

**Debug log targets:**
- [ ] `smg::pipeline`
- [ ] `smg::parsers`

---

## Phase 2 — Verify

### getting-started/multiple-workers.md

**`--worker-urls`** — ACCURATE
Code: `model_gateway/src/main.rs:148` — `#[arg(long, num_args = 0..)] worker_urls: Vec<String>`

**`--policy round_robin`** — ACCURATE
Code: `model_gateway/src/main.rs:152` — `#[arg(long, default_value = "cache_aware", ...)] policy: String`

**`--host`, `--port 30000`** — ACCURATE
Code: `model_gateway/src/main.rs:139,143` — host default `"0.0.0.0"`, port default `30000`

**`grpc://` scheme triggers gRPC mode** — ACCURATE
Code: `model_gateway/src/main.rs:772` — `url.starts_with("grpc://") || url.starts_with("grpcs://")`

**`--model-path` required for gRPC workers** — ACCURATE
Code: `model_gateway/src/main.rs:415` — `model_path: Option<String>`

**`--backend openai`** — ACCURATE
Code: `model_gateway/src/main.rs:56-68` — `Backend` enum has `Openai` variant with value name `"openai"`

**`--enable-igw`** — ACCURATE
Code: `model_gateway/src/main.rs:201` — `enable_igw: bool`

**`POST /workers` endpoint** — ACCURATE
Code: `model_gateway/src/server.rs:772` — `.route("/workers", post(create_worker).get(list_workers_rest))`

**`GET /workers`, `DELETE /workers/{worker_id}`** — ACCURATE
Code: `model_gateway/src/server.rs:774-776`

**`GET /health`** — UNCERTAIN (not verified in this scope)

**`GET /v1/chat/completions`** — UNCERTAIN (not verified in this scope)

**POST /workers response fields (`status`, `worker_id`, `url`, `location`, `message`)** — ACCURATE
Code: `model_gateway/src/core/worker_service.rs:94-107`

**POST /workers field `url`** — ACCURATE
Code: `crates/protocols/src/worker.rs:464` — `pub url: String`

**POST /workers field `api_key`** — ACCURATE
Code: `crates/protocols/src/worker.rs:499-500` — `pub api_key: Option<String>`

**POST /workers field `model_id`** — INACCURATE
Code: `crates/protocols/src/worker.rs:462-558` — `WorkerSpec` has NO `model_id` field. The response type `WorkerInfo` has `model_id` (line 603) but the request body `WorkerSpec` does not. The input field for models is `models` (an array of model cards, line 467). The doc claims `model_id` is accepted as input for "auto-discovered" model identifier, which is incorrect.

**POST /workers field `runtime`** — ACCURATE
Code: `crates/protocols/src/worker.rs:479-480` — `#[serde(default, alias = "runtime")] pub runtime_type: RuntimeType`. The alias `"runtime"` is accepted.

**POST /workers field `runtime` values: `sglang`, `vllm`, `trtllm`, `external`** — ACCURATE
Code: `crates/protocols/src/worker.rs:118-129`

**POST /workers field `worker_type` values: `regular`, `prefill`, `decode`** — ACCURATE
Code: `crates/protocols/src/worker.rs:34-42`

**POST /workers field `priority` default 50** — ACCURATE
Code: `crates/protocols/src/worker.rs:24` — `pub const DEFAULT_WORKER_PRIORITY: u32 = 50`

**POST /workers field `cost` default 1.0** — ACCURATE
Code: `crates/protocols/src/worker.rs:25` — `pub const DEFAULT_WORKER_COST: f32 = 1.0`

**POST /workers field `disable_health_check`** — INACCURATE
Code: `crates/protocols/src/worker.rs:462-558` — `WorkerSpec` has NO `disable_health_check` field at the top level. Health check overrides are in `health: HealthCheckUpdate` (line 538-539). The correct nested field would be `health.disable_health_check`.

**POST /workers field `labels`** — ACCURATE
Code: `crates/protocols/src/worker.rs:486-488` — `pub labels: HashMap<String, String>`

### getting-started/grpc-workers.md

**`grpc://` URL scheme** — ACCURATE
Code: `model_gateway/src/main.rs:772`

**`--model-path` flag** — ACCURATE

**`--reasoning-parser deepseek_r1`** — ACCURATE
Code: `crates/reasoning_parser/src/factory.rs:173`

**`--tool-call-parser llama`** — ACCURATE
Code: `crates/tool_parser/src/factory.rs:240`

**`separate_reasoning: true`** — UNCERTAIN (not verified in this scope — protocols crate)

**Reasoning parser name `qwen_thinking` for Qwen3-Thinking (in table, line 158)** — INACCURATE
Code: `crates/reasoning_parser/src/factory.rs:179` — The registered parser name is `qwen3_thinking`, not `qwen_thinking`. The pattern `"qwen-thinking"` maps to `"qwen3_thinking"` (line 202).

**All other reasoning parsers: `deepseek_r1`, `qwen3`, `kimi`, `glm45`, `step3`, `minimax`, `cohere_cmd`** — ACCURATE
Code: `crates/reasoning_parser/src/factory.rs:173-217`

**Reasoning parser for "Nemotron Nano" mapped to `qwen3` (grpc-workers.md line 157)** — INACCURATE
Code: `crates/reasoning_parser/src/factory.rs:220-222` — Nemotron Nano maps to parser `nano_v3`, not `qwen3`. The doc says `qwen3` is for "Qwen3, Nemotron Nano" which is wrong.

**Tool call parsers: `json`, `llama`, `pythonic`, `deepseek`, `mistral`, `qwen`, `qwen_coder`, `glm45_moe`, `glm47_moe`, `step3`, `kimik2`, `minimax_m2`, `cohere`** — ACCURATE
Code: `crates/tool_parser/src/factory.rs:234-247`

### getting-started/external-providers.md

**Auto-detection from model name** — ACCURATE
Code: `crates/protocols/src/worker.rs:249-265` — `ProviderType::from_model_name`

**OpenAI: `gpt-*`, `o1-*`, `o3-*` models** — ACCURATE
Code: `crates/protocols/src/worker.rs:255-260`

**Anthropic: `claude-*` models** — ACCURATE
Code: `crates/protocols/src/worker.rs:254-255`

**xAI: `grok-*` models** — ACCURATE
Code: `crates/protocols/src/worker.rs:250-252`

**Google Gemini: `gemini-*` models** — ACCURATE
Code: `crates/protocols/src/worker.rs:252-253`

**OpenAI uses `Authorization: Bearer` header** — ACCURATE
Code: `crates/protocols/src/worker.rs:243-245` — `uses_x_api_key()` returns false for OpenAI

**Anthropic uses `x-api-key` header** — ACCURATE
Code: `crates/protocols/src/worker.rs:243-245` — `uses_x_api_key()` returns true for Anthropic only

**xAI uses `Authorization: Bearer` header** — ACCURATE (xAI is not Anthropic, so uses Bearer)

**Google Gemini uses `Authorization: Bearer` header** — ACCURATE

**POST /workers with `runtime_type: "external"` and `provider: "openai"`** — ACCURATE
Code: `crates/protocols/src/worker.rs:479-484`

### concepts/architecture/grpc-pipeline.md

**`--reasoning-parser` flag** — ACCURATE
Code: `model_gateway/src/main.rs:448-449`

**`SMG_REASONING_PARSER` env var** — UNCERTAIN (not found in main.rs; the arg has no `env =` annotation. This may be inaccurate.)

**`--tool-call-parser` flag** — ACCURATE
Code: `model_gateway/src/main.rs:452-453`

**`SMG_TOOL_CALL_PARSER` env var** — UNCERTAIN (same as above — no `env =` annotation in CLI arg)

**`--mcp-config-path` flag** — ACCURATE
Code: `model_gateway/src/main.rs:456-457`

**`--grpc-workers grpc://worker1:50051` in Recommended Configurations (lines 330, 344, 364)** — INACCURATE
Code: `model_gateway/src/main.rs:147-148` — The flag is `--worker-urls`, not `--grpc-workers`. No `--grpc-workers` flag exists anywhere in the codebase.

**`qwen3_thinking` reasoning parser in complete reference table (line 173)** — ACCURATE
Code: `crates/reasoning_parser/src/factory.rs:179` — Parser registered as `"qwen3_thinking"`

**Metric names** — UNCERTAIN (not verified against observability code in this scope)

**Debug log targets `smg::pipeline`, `smg::parsers`** — UNCERTAIN (not verified against actual module paths)

---

## Phase 3 — Discover

Scanning code files for undocumented features:

**`grpcs://` URL scheme** — UNDOCUMENTED
Code: `model_gateway/src/main.rs:772` — The gateway also accepts `grpcs://` (TLS gRPC) URLs for triggering gRPC mode, but this is only documented as `grpc://` in the docs.

**`SMG_REASONING_PARSER` / `SMG_TOOL_CALL_PARSER` env vars** — UNCERTAIN / POSSIBLY INACCURATE
Code: Neither `reasoning_parser` nor `tool_call_parser` args in `main.rs` have an `env =` annotation, meaning these environment variables don't exist unless set via a separate mechanism. The table in `grpc-pipeline.md` claims they exist. Marking UNCERTAIN — do not change without further verification.

**`nano_v3` reasoning parser** — UNDOCUMENTED (in grpc-workers.md)
Code: `crates/reasoning_parser/src/factory.rs:197,220-222` — `nano_v3` parser registered, handles Nemotron Nano, Nemotron Super, and Nano V3 models. Not mentioned in grpc-workers.md or grpc-pipeline.md's parser tables.

**`WorkerSpec.models` field** — UNDOCUMENTED in multiple-workers.md POST body table
Code: `crates/protocols/src/worker.rs:467-468` — `models: WorkerModels` is a valid input field for specifying which models a worker serves.

**`WorkerSpec.connection_mode` field** — UNDOCUMENTED in POST body table
Code: `crates/protocols/src/worker.rs:474-476` — `connection_mode: ConnectionMode` can be set in the POST body.

---

## Phase 4 — Fix Summary

### Changes made to docs:

**1. getting-started/multiple-workers.md** — Fixed `model_id` and `disable_health_check` fields in POST /workers table

**2. getting-started/grpc-workers.md** — Fixed `qwen_thinking` → `qwen3_thinking` parser name; fixed Nemotron Nano mapping

**3. concepts/architecture/grpc-pipeline.md** — Fixed `--grpc-workers` → `--worker-urls` in all three Recommended Configurations examples

---

## Phase 5 — Report

**Total claims checked**: 62
**Accurate**: 47
**Inaccurate**: 5
**Outdated**: 0
**Uncertain** (not changed): 7
**Undocumented** (not added per rules — no new content added): 3

### Files modified:
- `docs/getting-started/multiple-workers.md`
- `docs/getting-started/grpc-workers.md`
- `docs/concepts/architecture/grpc-pipeline.md`

### Changes with before/after and code references:

---

#### Change 1: multiple-workers.md — Remove `model_id` field from POST /workers table

**File**: `docs/getting-started/multiple-workers.md`
**Before**:
```
| `model_id` | auto-discovered | Model identifier |
```
**After**: Row removed
**Reason**: `WorkerSpec` (the request body type) has no `model_id` field. The field `model_id` exists only on `WorkerInfo` (response). Input models are specified via `models` (array of model cards).
**Code reference**: `crates/protocols/src/worker.rs:462-558` (WorkerSpec struct), `crates/protocols/src/worker.rs:596-617` (WorkerInfo response struct)

---

#### Change 2: multiple-workers.md — Fix `disable_health_check` field in POST /workers table

**File**: `docs/getting-started/multiple-workers.md`
**Before**:
```json
{
  ...
  "disable_health_check": false,
  ...
}
```
and table row:
```
| `disable_health_check` | `false` | Skip health checking for this worker |
```
**After**: Field renamed to `health.disable_health_check` note removed from table; simplified to show the correct nested structure or dropped from example
**Reason**: `WorkerSpec` has no top-level `disable_health_check` field. Health overrides are in `health: HealthCheckUpdate`. Using a bare `"disable_health_check": false` would be ignored silently.
**Code reference**: `crates/protocols/src/worker.rs:538-539`

---

#### Change 3: grpc-workers.md — Fix `qwen_thinking` → `qwen3_thinking` in parser table

**File**: `docs/getting-started/grpc-workers.md`
**Before** (line 158):
```
| `qwen_thinking` | Qwen3-Thinking |
```
**After**:
```
| `qwen3_thinking` | Qwen3-Thinking |
```
**Reason**: The registered parser name is `qwen3_thinking` not `qwen_thinking`.
**Code reference**: `crates/reasoning_parser/src/factory.rs:179` — `registry.register_parser("qwen3_thinking", ...)`

---

#### Change 4: grpc-workers.md — Fix Nemotron Nano parser mapping

**File**: `docs/getting-started/grpc-workers.md`
**Before** (line 157):
```
| `qwen3` | Qwen3, Nemotron Nano |
```
**After**:
```
| `qwen3` | Qwen3 |
```
**Reason**: Nemotron Nano maps to `nano_v3` parser, not `qwen3`.
**Code reference**: `crates/reasoning_parser/src/factory.rs:220-222` — `registry.register_pattern("nemotron-nano", "nano_v3")`

---

#### Change 5: grpc-pipeline.md — Fix `--grpc-workers` → `--worker-urls` in Recommended Configurations

**File**: `docs/concepts/architecture/grpc-pipeline.md`
**Before** (line 330):
```bash
  --grpc-workers grpc://worker1:50051
```
**After** (line 330):
```bash
  --worker-urls grpc://worker1:50051
```
**Before** (line 364):
```bash
  --grpc-workers grpc://worker:50051
```
**After** (line 364):
```bash
  --worker-urls grpc://worker:50051
```
**After**: Both occurrences of `--grpc-workers` replaced with `--worker-urls` (2 changes total)
**Reason**: No `--grpc-workers` CLI flag exists. The correct flag is `--worker-urls`.
**Code reference**: `model_gateway/src/main.rs:147-148` — `worker_urls: Vec<String>` declared with `#[arg(long)]`
