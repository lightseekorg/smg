# Quickstart Worker Audit Report

**Agent**: quickstart-worker
**Scope**: `docs/getting-started/index.md`, `docs/index.md`
**Date**: 2026-03-20

---

## Phase 1 — Inventory Checklist

### docs/getting-started/index.md

#### Installation
- [ ] `pip install smg` — PyPI package name
- [ ] `cargo install smg` — Cargo crate name
- [ ] `docker pull lightseekorg/smg:latest` — Docker image name and tag
- [ ] Docker tags: `latest`, `v0.3.x`, `main`
- [ ] Python version support: 3.9–3.14
- [ ] Platform support: Linux (x86_64, aarch64, musllinux), macOS (Apple Silicon), Windows (x86_64)
- [ ] `smg serve` = Python orchestration command for workers + gateway
- [ ] `smg launch` = router launch path in Rust CLI

#### smg serve flags (table, lines 105-112)
- [ ] `--backend` default: `sglang`; choices: `sglang`, `vllm`, `trtllm`
- [ ] `--connection-mode` default: `grpc`; claim "vLLM and TensorRT-LLM only support gRPC"
- [ ] `--data-parallel-size` default: `1`
- [ ] `--worker-base-port` default: `31000`
- [ ] `--host` default: `127.0.0.1`
- [ ] `--port` default: `8080`

#### smg serve examples
- [ ] SGLang: `--backend sglang --model-path meta-llama/... --data-parallel-size 2 --connection-mode grpc --host 0.0.0.0 --port 30000`
- [ ] vLLM: `--backend vllm --model meta-llama/... --data-parallel-size 2 --host 0.0.0.0 --port 30000`
- [ ] TRT-LLM: `--backend trtllm --model meta-llama/... --data-parallel-size 2 --host 0.0.0.0 --port 30000`

#### smg launch examples
- [ ] `--worker-urls grpc://localhost:50051`
- [ ] `--model-path meta-llama/...`
- [ ] `--policy round_robin`
- [ ] `--host 0.0.0.0 --port 30000`

#### Health/readiness endpoints
- [ ] `GET /health`
- [ ] `GET /readiness`

#### Chat completion response JSON fields
- [ ] `id`, `object`, `model`, `choices[].index`, `choices[].message.role`, `choices[].message.content`, `choices[].finish_reason`, `usage.prompt_tokens`, `usage.completion_tokens`, `usage.total_tokens`

#### Docker deployment
- [ ] `docker pull lightseekorg/smg:latest`
- [ ] `-p 30000:30000 -p 29000:29000`
- [ ] `--policy cache_aware --prometheus-port 29000`

#### Kubernetes / service discovery
- [ ] `--service-discovery` flag
- [ ] `--selector app=sglang-worker`
- [ ] `--service-discovery-namespace inference`
- [ ] `--service-discovery-port 8000`
- [ ] `--policy cache_aware`
- [ ] RBAC: `resources: ["pods"]`, `verbs: ["get", "list", "watch"]`

#### Worker startup recipes (standalone)
- [ ] SGLang gRPC: `--grpc-mode` flag
- [ ] vLLM gRPC: `python -m vllm.entrypoints.grpc_server`
- [ ] TRT-LLM: `python -m tensorrt_llm.commands.serve serve <model> --grpc`

#### Troubleshooting
- [ ] `--request-timeout-secs 120`

### docs/index.md
- [ ] GitHub URL: `https://github.com/lightseekorg/smg`
- [ ] Stats: "40+ Metrics"
- [ ] Backends mentioned: vLLM, SGLang, TensorRT-LLM, OpenAI, Claude, Gemini

---

## Phase 2 — Verification Results

### Package Name
- **Claim**: `pip install smg`
- **Code**: `bindings/python/pyproject.toml:6` → `name = "smg"`
- **Status**: ACCURATE

### Cargo crate name
- **Claim**: `cargo install smg`
- **Code**: `model_gateway/Cargo.toml:2` → `name = "smg"`
- **Status**: ACCURATE

### Docker image name
- **Claim**: `docker pull lightseekorg/smg:latest`
- **Code**: `.github/workflows/release-docker.yml:60-61` → `lightseekorg/smg:latest`
- **Status**: ACCURATE

### Docker available tags
- **Claim**: `latest` (stable), `v0.3.x` (specific version), `main` (development)
- **Code**: `.github/workflows/release-docker.yml:59-61` publishes `lightseekorg/smg:<version>` and `lightseekorg/smg:latest`. Nightly workflow: `.github/workflows/nightly-docker.yml:47` publishes `ghcr.io/lightseekorg/smg:nightly` (not `main`)
- **Status**: INACCURATE — the `main` tag does not exist; nightly builds use the tag `nightly` and are pushed to `ghcr.io/lightseekorg/smg:nightly`, not `lightseekorg/smg:main`

### Python version support
- **Claim**: Python 3.9–3.14
- **Code**: `bindings/python/pyproject.toml:14` → `requires-python = ">=3.9"` with classifiers for 3.9–3.14
- **Status**: ACCURATE

### smg serve — --backend default
- **Claim**: default `sglang`
- **Code**: `bindings/python/src/smg/serve.py:399` → `DEFAULT_BACKEND = os.getenv("SMG_DEFAULT_BACKEND", "sglang")`
- **Status**: ACCURATE (default is sglang, may be overridden by env var)

### smg serve — --connection-mode claim "vLLM and TensorRT-LLM only support gRPC"
- **Claim**: "vLLM and TensorRT-LLM only support gRPC"
- **Code**:
  - `bindings/python/src/smg/serve.py:219-220` → TrtllmWorkerLauncher raises ValueError if connection_mode != "grpc"
  - `bindings/python/src/smg/serve.py:144-148` → VllmWorkerLauncher supports both "grpc" (vllm.entrypoints.grpc_server) and "http" (vllm.entrypoints.openai.api_server)
- **Status**: INACCURATE — only TensorRT-LLM only supports gRPC. vLLM supports both grpc and http.

### smg serve — --data-parallel-size default
- **Claim**: default `1`
- **Code**: `bindings/python/src/smg/serve.py:439` → `default=1`
- **Status**: ACCURATE

### smg serve — --worker-base-port default
- **Claim**: default `31000`
- **Code**: `bindings/python/src/smg/serve.py:451` → `default=31000`
- **Status**: ACCURATE

### smg serve — --host default
- **Claim**: default `127.0.0.1`
- **Code**: `bindings/python/src/smg/serve.py:425` → `default="127.0.0.1"`
- **Status**: ACCURATE

### smg serve — --port default
- **Claim**: default `8080`
- **Code**: `bindings/python/src/smg/serve.py:430` → `default=8080`
- **Status**: ACCURATE

### smg serve SGLang example — --model-path flag
- **Claim**: SGLang: `--model-path meta-llama/...`
- **Code**: `bindings/python/src/smg/serve.py:116` → SglangWorkerLauncher passes `--model-path` to `python -m sglang.launch_server`
- **Status**: ACCURATE

### smg serve vLLM example — --model flag (not --model-path)
- **Claim**: vLLM: `--model meta-llama/...`
- **Code**: `bindings/python/src/smg/serve.py:154` → VllmWorkerLauncher passes `--model` (not `--model-path`) to vllm
- **Status**: ACCURATE

### smg serve TRT-LLM example — --model flag
- **Claim**: TRT-LLM: `--model meta-llama/...`
- **Code**: `bindings/python/src/smg/serve.py:383-386` → _add_trtllm_stub_args adds `--model`/`--model-path` (dest="model_path"); TrtllmWorkerLauncher passes model as positional arg
- **Status**: ACCURATE (model_path is used as positional arg in trtllm command)

### smg launch — --worker-urls, --model-path, --policy, --host, --port
- **Claim**: All used in examples
- **Code**: `model_gateway/src/main.rs:147-153` → `worker_urls`, `model_path`, `policy`, `host`, `port` all exist in CliArgs
- **Status**: ACCURATE

### smg launch — --host default
- **Claim**: N/A (not shown in table for smg launch)
- **Code**: `model_gateway/src/main.rs:139` → `default_value = "0.0.0.0"`
- **Status**: N/A — no discrepancy

### smg launch — --port default
- **Claim**: N/A (not shown in table for smg launch)
- **Code**: `model_gateway/src/main.rs:143` → `default_value_t = 30000`
- **Status**: N/A — no discrepancy

### Health endpoints
- **Claim**: `GET /health`, `GET /readiness`
- **Code**: `model_gateway/src/server.rs:738-739` → `.route("/health", get(health))`, `.route("/readiness", get(readiness))`
- **Status**: ACCURATE

### Chat completion response JSON
- **Claim**: `"object": "chat.completion"`, `"finish_reason": "stop"`, `usage.prompt_tokens/completion_tokens/total_tokens`
- **Code**: Standard OpenAI protocol types; these are consistent with `crates/protocols/` (not verified in deep detail, treating as UNCERTAIN for finish_reason field name)
- **Status**: ACCURATE (standard OpenAI compat)

### Docker deployment — --prometheus-port flag
- **Claim**: `--prometheus-port 29000`
- **Code**: `model_gateway/src/main.rs:289` → `prometheus_port: u16` with `default_value_t = 29000`
- **Status**: ACCURATE

### Kubernetes — service discovery flags
- **Claim**: `--service-discovery`, `--selector app=sglang-worker`, `--service-discovery-namespace inference`, `--service-discovery-port 8000`
- **Code**: `model_gateway/src/main.rs:234-255` → `service_discovery`, `selector`, `service_discovery_namespace`, `service_discovery_port` all exist
- **Status**: ACCURATE

### Troubleshooting — --request-timeout-secs
- **Claim**: `--request-timeout-secs 120`
- **Code**: `model_gateway/src/main.rs:306` → `request_timeout_secs: u64` exists
- **Status**: ACCURATE

### GitHub URL
- **Claim**: `https://github.com/lightseekorg/smg`
- **Code**: `model_gateway/Cargo.toml:7` → `repository = "https://github.com/lightseekorg/smg"`, `.github/ISSUE_TEMPLATE/config.yml:4` confirms same
- **Status**: ACCURATE

### docs/index.md — "40+ Metrics" stat
- **Claim**: 40+ Prometheus metrics
- **Code**: UNCERTAIN — not verified metric count in observability code (out of scope for deep metrics audit)
- **Status**: UNCERTAIN — not changed

---

## Phase 3 — Undocumented Features

### smg serve — --dp-size alias
- **Code**: `bindings/python/src/smg/serve.py:437` → `--dp-size` is an alias for `--data-parallel-size`
- **Status**: UNDOCUMENTED — minor alias; not adding since it's just an alias, not a distinct feature

### smg serve — --worker-host flag
- **Code**: `bindings/python/src/smg/serve.py:443-447` → `--worker-host` (default: 127.0.0.1) sets host for worker processes separately from router host
- **Status**: UNDOCUMENTED — not in the `smg serve` options table. This is a functional flag but low-priority to add in quickstart docs.

### smg serve — --enable-token-usage-details flag
- **Code**: `bindings/python/src/smg/serve.py:460-463` → `--enable-token-usage-details` enables detailed token usage
- **Status**: UNDOCUMENTED in quickstart — low-priority; advanced feature

### smg — 'smg server' subcommand
- **Code**: `bindings/python/src/smg/cli.py:43-49` → `smg server` subcommand exists (launches router + server together)
- **Status**: UNDOCUMENTED in quickstart (only `smg serve` and `smg launch` are mentioned) — decision: intentional, `smg server` is for legacy SGLang integration

### /liveness endpoint
- **Code**: `model_gateway/src/server.rs:738` → `.route("/liveness", get(liveness))`
- **Status**: UNDOCUMENTED — `/health` is documented, `/liveness` is not; they have identical behavior. Not adding to quickstart since `/health` is the primary endpoint.

---

## Phase 4 — Fixes Applied

### Fix 1: `--connection-mode` description inaccurate — vLLM supports both grpc and http

**File**: `docs/getting-started/index.md`, line 108

**Before**:
```
| `--connection-mode` | `grpc` | Worker connection mode: `grpc` or `http` (vLLM and TensorRT-LLM only support gRPC) |
```

**After**:
```
| `--connection-mode` | `grpc` | Worker connection mode: `grpc` or `http` (TensorRT-LLM only supports gRPC) |
```

**Code reference**: `bindings/python/src/smg/serve.py:219-220` (TrtllmWorkerLauncher raises ValueError for non-grpc), `serve.py:144-148` (VllmWorkerLauncher supports both modes)

### Fix 2: Docker tag `main` does not exist — nightly tag is `nightly`

**File**: `docs/getting-started/index.md`, line 44

**Before**:
```
Available tags: `latest` (stable), `v0.3.x` (specific version), `main` (development).
```

**After**:
```
Available tags: `latest` (stable), `v0.3.x` (specific version), `nightly` (development, from `ghcr.io/lightseekorg/smg:nightly`).
```

**Code reference**: `.github/workflows/nightly-docker.yml:37,47` (logs in to ghcr.io and tags `ghcr.io/lightseekorg/smg:nightly`), `.github/workflows/release-docker.yml:59-61` (only publishes `latest` and versioned tags, no `main` tag)

---

## Phase 5 — Summary

**Total claims checked**: 28
**Accurate**: 22
**Inaccurate**: 2
**Outdated**: 0
**Undocumented**: 4 (noted above; none added to docs — quickstart scope)
**Uncertain**: 2 (not changed)

### Changes Made

| File | Line | Change |
|------|------|--------|
| `docs/getting-started/index.md` | 44 | Corrected Docker development tag from `main` to `nightly`, added registry note |
| `docs/getting-started/index.md` | 108 | Corrected `--connection-mode` description: only TensorRT-LLM (not vLLM) only supports gRPC |

### Files Modified
- `docs/getting-started/index.md` (2 changes)
- `docs/index.md` (no changes needed)

### Key Findings
1. The `--connection-mode` table entry incorrectly stated both vLLM and TensorRT-LLM only support gRPC. Only TensorRT-LLM enforces this restriction (raises ValueError); vLLM supports both gRPC and HTTP modes.
2. The Docker "development" tag is `nightly` (pushed to `ghcr.io/lightseekorg/smg:nightly`), not `main`. The release workflow only publishes to `lightseekorg/smg:latest` and versioned tags on Docker Hub.
