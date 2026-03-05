# Workspace Reorganization Design

**Date**: 2026-03-05
**Status**: Approved

## Problem

The repository has 22 top-level directories, 13 of which are Rust crates. This creates visual clutter and obscures the architectural hierarchy between core libraries, infrastructure services, and the main application.

## Goals

1. Reduce top-level directory count (22 → 14)
2. Make the folder structure reflect dependency layers (core libs vs infra vs app)
3. Preserve git history via `git mv`
4. Zero package name changes — only path updates

## Target Layout

```
smg/
├── core/                          # Core library crates (8)
│   ├── protocols/                 # openai-protocol
│   ├── reasoning_parser/          # reasoning-parser
│   ├── tool_parser/               # tool-parser
│   ├── tokenizer/                 # llm-tokenizer
│   ├── multimodal/                # llm-multimodal
│   ├── kv_index/                  # kv-index
│   ├── mcp/                       # smg-mcp
│   └── grpc_client/               # smg-grpc-client
├── infra/                         # Infrastructure/service crates (5)
│   ├── auth/                      # smg-auth
│   ├── mesh/                      # smg-mesh
│   ├── data_connector/            # data-connector
│   ├── wasm/                      # smg-wasm
│   └── workflow/                  # wfaas
├── model_gateway/                 # Main application (stays at root)
├── bindings/                      # Language bindings (unchanged)
├── clients/                       # Client libraries (unchanged)
├── e2e_test/                      # E2E tests (unchanged)
├── grpc_servicer/                 # gRPC servicer (unchanged)
├── docker/                        # Docker configs (unchanged)
├── docs/                          # Documentation (unchanged)
├── examples/                      # Examples (unchanged)
├── scripts/                       # Scripts (unchanged)
└── Cargo.toml                     # Workspace root
```

## Grouping Rationale

**Core** — foundational libraries that define protocols, parsing, tokenization, indexing, MCP client, and gRPC client functionality. These are depended on by many other crates.

**Infrastructure** — service-layer crates for authentication, service mesh, data connectors, WASM runtime, and workflow orchestration. These build on core crates to provide infrastructure capabilities.

**Application** — `model_gateway` is the main binary/application that composes core and infra crates. Stays at root for top-level visibility.

## Change Scope

### What changes

1. **Root `Cargo.toml`** — workspace `members` and `[workspace.dependencies]` path values
2. **Inter-crate `Cargo.toml`** — any `path = "../<crate>"` relative references between crates
3. **CI workflows** (`.github/workflows/`) — any hardcoded crate paths
4. **Dockerfiles** — any `COPY` or path references to moved crates
5. **Scripts** — any path references in `scripts/`

### What does NOT change

- Package names (e.g., `smg-mcp`, `openai-protocol`) — unchanged
- Rust source code — zero `.rs` file modifications
- `bindings/`, `clients/` — already nested, unchanged
- Non-Rust directories (`docker/`, `docs/`, `examples/`, `scripts/`, `e2e_test/`, `grpc_servicer/`) — unchanged

## Migration Strategy

Single atomic PR with all moves and path updates in one commit:

1. Create `core/` and `infra/` directories
2. `git mv` each crate into its group directory
3. Update root `Cargo.toml` (workspace members + dependency paths)
4. Update inter-crate `Cargo.toml` relative paths
5. Update CI workflows, Dockerfiles, and scripts
6. Verify with `cargo check --workspace`
7. Run tests to confirm nothing broke

**Risk mitigation**: `git mv` preserves history. Single commit enables easy revert. No package name changes means zero impact on external dependents.
