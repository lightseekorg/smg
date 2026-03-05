# Workspace Reorganization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move 13 Rust crates into `core/` (8) and `infra/` (5) group directories to reduce root clutter and reflect architectural hierarchy.

**Architecture:** Flat directory move with path updates. No package renames, no Rust source changes. Single atomic commit.

**Tech Stack:** git mv, Cargo workspace paths, GitHub Actions YAML

---

### Task 1: Move crate directories with git mv

**Files:**
- Create: `core/`, `infra/`
- Move: 13 crate directories

**Step 1: Create group directories**

```bash
mkdir core infra
```

**Step 2: Move core crates**

```bash
git mv protocols core/
git mv reasoning_parser core/
git mv tool_parser core/
git mv tokenizer core/
git mv multimodal core/
git mv kv_index core/
git mv mcp core/
git mv grpc_client core/
```

**Step 3: Move infra crates**

```bash
git mv auth infra/
git mv mesh infra/
git mv data_connector infra/
git mv wasm infra/
git mv workflow infra/
```

**Step 4: Verify moves**

Run: `ls core/ infra/`
Expected: 8 dirs in core, 5 dirs in infra

---

### Task 2: Update root Cargo.toml

**Files:**
- Modify: `Cargo.toml` (lines 2, 7-19)

**Step 1: Update workspace members (line 2)**

Change:
```toml
members = ["model_gateway", "protocols", "reasoning_parser", "tool_parser", "workflow", "tokenizer", "auth", "mcp", "kv_index", "data_connector", "multimodal", "wasm", "mesh", "grpc_client", "bindings/python", "bindings/golang", "clients/rust", "clients/openapi-gen"]
```

To:
```toml
members = ["model_gateway", "core/protocols", "core/reasoning_parser", "core/tool_parser", "infra/workflow", "core/tokenizer", "infra/auth", "core/mcp", "core/kv_index", "infra/data_connector", "core/multimodal", "infra/wasm", "infra/mesh", "core/grpc_client", "bindings/python", "bindings/golang", "clients/rust", "clients/openapi-gen"]
```

**Step 2: Update workspace dependency paths (lines 7-19)**

Change each path:
```toml
openai-protocol = { version = "1.2.0", path = "core/protocols" }
reasoning-parser = { version = "1.2.0", path = "core/reasoning_parser" }
tool-parser = { version = "1.1.1", path = "core/tool_parser" }
wfaas = { version = "1.0.2", path = "infra/workflow" }
llm-tokenizer = { version = "1.2.0", path = "core/tokenizer" }
smg-auth = { version = "1.1.1", path = "infra/auth" }
smg-mcp = { version = "2.1.0", path = "core/mcp" }
kv-index = { version = "1.0.1", path = "core/kv_index" }
smg-data-connector = { version = "2.0.0", path = "infra/data_connector", package = "data-connector" }
llm-multimodal = { version = "1.2.0", path = "core/multimodal" }
smg-wasm = { version = "1.0.1", path = "infra/wasm", package = "smg-wasm" }
smg-mesh = { version = "1.1.1", path = "infra/mesh", package = "smg-mesh" }
smg-grpc-client = { version = "1.2.0", path = "core/grpc_client" }
```

**Step 3: Verify Cargo.toml parses**

Run: `cargo metadata --no-deps --format-version 1 > /dev/null`
Expected: No errors

---

### Task 3: Update inter-crate Cargo.toml relative paths

**Files:**
- Modify: `bindings/python/Cargo.toml` (lines 21, 24)
- Modify: `bindings/golang/Cargo.toml` (lines 31, 34)
- Modify: `clients/openapi-gen/Cargo.toml` (line 10)

**Step 1: Update bindings/python/Cargo.toml**

Line 21: `path = "../../auth"` → `path = "../../infra/auth"`
Line 24: `path = "../../tool_parser"` → `path = "../../core/tool_parser"`

**Step 2: Update bindings/golang/Cargo.toml**

Line 31: `path = "../../tokenizer"` → `path = "../../core/tokenizer"`
Line 34: `path = "../../tool_parser"` → `path = "../../core/tool_parser"`

**Step 3: Update clients/openapi-gen/Cargo.toml**

Line 10: `path = "../../protocols"` → `path = "../../core/protocols"`

**Step 4: Verify workspace builds**

Run: `cargo check --workspace`
Expected: Compiles with no errors

---

### Task 4: Update CI workflows — benchmark files

**Files:**
- Modify: `.github/workflows/benchmark-tool-parser.yml` (lines 8-11, 16-19)
- Modify: `.github/workflows/benchmark-tokenizer.yml` (lines 8-11, 16-19)
- Modify: `.github/workflows/benchmark-radix-tree.yml` (lines 8-11, 16-19)

**Step 1: benchmark-tool-parser.yml**

Change all `tool_parser/` references to `core/tool_parser/`:
```yaml
# Lines 7-11 (push paths) and 15-19 (pull_request paths)
- 'core/tool_parser/**'
- '!core/tool_parser/**/tests/**'
- '!core/tool_parser/**/*_test.rs'
- '!core/tool_parser/**/test_*.rs'
```

**Step 2: benchmark-tokenizer.yml**

Change all `tokenizer/` references to `core/tokenizer/`:
```yaml
- 'core/tokenizer/**'
- '!core/tokenizer/**/tests/**'
- '!core/tokenizer/**/*_test.rs'
- '!core/tokenizer/**/test_*.rs'
```

**Step 3: benchmark-radix-tree.yml**

Change all `kv_index/` references to `core/kv_index/`:
```yaml
- 'core/kv_index/**'
- '!core/kv_index/**/tests/**'
- '!core/kv_index/**/*_test.rs'
- '!core/kv_index/**/test_*.rs'
```

---

### Task 5: Update CI workflows — nightly-benchmark.yml

**Files:**
- Modify: `.github/workflows/nightly-benchmark.yml` (line 49)

**Step 1: Update hashFiles cache key**

Update the hashFiles glob on line 49. Replace each crate path with its new location:
- `auth/src/**` → `infra/auth/src/**`
- `data_connector/src/**` → `infra/data_connector/src/**`
- `grpc_client/src/**` → `core/grpc_client/src/**`
- `kv_index/src/**` → `core/kv_index/src/**`
- `mcp/src/**` → `core/mcp/src/**`
- `mesh/src/**` → `infra/mesh/src/**`
- `multimodal/src/**` → `core/multimodal/src/**`
- `protocols/src/**` → `core/protocols/src/**`
- `reasoning_parser/src/**` → `core/reasoning_parser/src/**`
- `tokenizer/src/**` → `core/tokenizer/src/**`
- `tool_parser/src/**` → `core/tool_parser/src/**`
- `wasm/src/**` → `infra/wasm/src/**`
- `workflow/src/**` → `infra/workflow/src/**`

Keep `model_gateway/src/**` unchanged.

---

### Task 6: Update CI workflows — pr-test-rust.yml

**Files:**
- Modify: `.github/workflows/pr-test-rust.yml` (lines 93-102, 119, 121, 160, 187, 253, 273, 303-326, 390, 514, 590)

**Step 1: Update grpc_client/python paths (lines 93-102)**

Change all `grpc_client/` to `core/grpc_client/`:
```yaml
rm -f core/grpc_client/python/smg_grpc_proto/proto
mkdir -p core/grpc_client/python/smg_grpc_proto/proto
cp core/grpc_client/proto/*.proto core/grpc_client/python/smg_grpc_proto/proto/
rm -rf core/grpc_client/python/dist/
cd core/grpc_client/python && python -m build
pip install core/grpc_client/python/dist/*.whl
```

**Step 2: Update wasm fixture paths (lines 119, 160, 187, 253, 390, 514, 590)**

Change all `wasm/tests/fixtures/` to `infra/wasm/tests/fixtures/`:
```yaml
# Cache path (line 119)
infra/wasm/tests/fixtures/*.wasm

# Build script (lines 160, 253)
bash infra/wasm/tests/fixtures/build_fixtures.sh

# Artifact paths (lines 187, 390, 514, 590)
path: infra/wasm/tests/fixtures/*.wasm
# or
path: infra/wasm/tests/fixtures/
```

**Step 3: Update hashFiles cache key (line 121)**

Same pattern as Task 5 — update all 13 crate src paths. Keep `model_gateway/src/**`, `bindings/golang/src/**`, and `examples/wasm/**` unchanged.

**Step 4: Update multimodal script path (line 273)**

Change: `python multimodal/scripts/generate_vision_golden.py`
To: `python core/multimodal/scripts/generate_vision_golden.py`

**Step 5: Update path filters (lines 303-326)**

Change each crate path in the dorny/paths-filter config:
```yaml
- 'core/protocols/**'
- 'core/tokenizer/**'
- 'core/tool_parser/**'
- 'core/reasoning_parser/**'
- 'core/multimodal/**'
- 'core/grpc_client/**'
- 'core/mcp/**'
- 'infra/data_connector/**'
```

---

### Task 7: Update CI workflows — release-crates.yml and release-grpc.yml

**Files:**
- Modify: `.github/workflows/release-crates.yml` (lines 25-65)
- Modify: `.github/workflows/release-grpc.yml` (lines 8, 38, 63-68, 71, 76)

**Step 1: release-crates.yml — update matrix paths**

Update each `path:` entry in the matrix:
```yaml
- crate: openai-protocol
  path: core/protocols
- crate: reasoning-parser
  path: core/reasoning_parser
- crate: kv-index
  path: core/kv_index
- crate: data-connector
  path: infra/data_connector
- crate: wfaas
  path: infra/workflow
- crate: smg-wasm
  path: infra/wasm
- crate: smg-auth
  path: infra/auth
- crate: llm-tokenizer
  path: core/tokenizer
- crate: tool-parser
  path: core/tool_parser
- crate: smg-grpc-client
  path: core/grpc_client
- crate: llm-multimodal
  path: core/multimodal
- crate: smg-mesh
  path: infra/mesh
```

**Step 2: release-grpc.yml — update grpc_client paths**

Change all `grpc_client/` to `core/grpc_client/`:
```yaml
# Line 8 (trigger path)
- core/grpc_client/python/pyproject.toml

# Line 38 (changed files check)
grep -q '^core/grpc_client/python/pyproject.toml$'

# Lines 63-68 (build steps)
rm -f core/grpc_client/python/smg_grpc_proto/proto
mkdir -p core/grpc_client/python/smg_grpc_proto/proto
cp core/grpc_client/proto/*.proto core/grpc_client/python/smg_grpc_proto/proto/
cd core/grpc_client/python && python -m build
twine check --strict core/grpc_client/python/dist/*

# Line 76 (artifact)
path: core/grpc_client/python/dist/
```

---

### Task 8: Update shell scripts

**Files:**
- Modify: `scripts/ci_install_vllm.sh` (line 31)
- Modify: `scripts/check_release_versions.sh` (lines 61-83)

**Step 1: ci_install_vllm.sh**

Line 31: `uv pip install -e grpc_client/python/` → `uv pip install -e core/grpc_client/python/`

**Step 2: check_release_versions.sh**

Update the middle (path) field for each entry:
```bash
"openai-protocol|core/protocols|openai-protocol"
"reasoning-parser|core/reasoning_parser|reasoning-parser"
"tool-parser|core/tool_parser|tool-parser"
"wfaas|infra/workflow|wfaas"
"llm-tokenizer|core/tokenizer|llm-tokenizer"
"smg-auth|infra/auth|smg-auth"
"smg-mcp|core/mcp|smg-mcp"
"kv-index|core/kv_index|kv-index"
"data-connector|infra/data_connector|smg-data-connector"
"llm-multimodal|core/multimodal|llm-multimodal"
"smg-wasm|infra/wasm|smg-wasm"
"smg-mesh|infra/mesh|smg-mesh"
"smg-grpc-client|core/grpc_client|smg-grpc-client"
"smg-grpc-proto|core/grpc_client/python|core/grpc_client/python/smg_grpc_proto/__init__.py"
```

---

### Task 9: Final verification and commit

**Step 1: Full workspace check**

Run: `cargo check --workspace`
Expected: Compiles with no errors

**Step 2: Run unit tests**

Run: `cargo test --workspace --lib`
Expected: All tests pass

**Step 3: Verify no stale references**

Run: `grep -rn '"protocols"\|"reasoning_parser"\|"tool_parser"\|"tokenizer"\|"multimodal"\|"kv_index"\|"grpc_client"' Cargo.toml .github/ scripts/ bindings/ clients/ | grep -v 'crate:\|name =\|package ='`
Expected: No matches (all old paths updated)

**Step 4: Commit**

```bash
git add -A
git commit -s -m "refactor: reorganize workspace into core/ and infra/ directories

Move 13 Rust crates into architectural group directories:

Core (8): protocols, reasoning_parser, tool_parser, tokenizer,
multimodal, kv_index, mcp, grpc_client

Infra (5): auth, mesh, data_connector, wasm, workflow

model_gateway stays at root as the main application.

Updates all workspace Cargo.toml paths, inter-crate relative paths,
CI workflow references, release configurations, and shell scripts.

No package name changes. No Rust source code changes.
Zero functional impact — only directory layout and path references.

Refs: docs/plans/2026-03-05-workspace-reorganization-design.md"
```
