# WASM Guest Storage Hook Example

A WebAssembly Component Model storage hook for Shepherd Model Gateway that demonstrates:

- **Multi-tenancy**: Extracts `tenant_id` from request context and writes it as an extra column
- **Audit trail**: Adds `CREATED_BY` / `STORED_BY` extra columns from `user_id` context
- **Validation**: Rejects `StoreResponse` operations when `tenant_id` is missing from context
- **Pass-through**: Leaves read/delete operations unchanged

## Before Hook Behavior

| Operation | Behavior |
|-----------|----------|
| `StoreResponse` | Requires `tenant_id` in context (rejects otherwise); adds `TENANT_ID` and optionally `STORED_BY` extra columns |
| `CreateConversation` | Adds `TENANT_ID` and `CREATED_BY` extra columns if available in context |
| `CreateItem` / `LinkItem` | Adds `TENANT_ID` extra column if available |
| All others | Continues without extra columns |

## After Hook Behavior

Passes through extra columns unchanged. A production hook might log operations, update metrics, or enrich columns with post-operation data.

## Prerequisites

```bash
rustup target add wasm32-wasip2
cargo install wasm-tools
```

## Build

```bash
cd examples/wasm/wasm-guest-storage-hook
./build.sh
```

The compiled component will be at:
```
target/wasm32-wasip2/release/wasm_guest_storage_hook.component.wasm
```

## Usage

```rust
use smg_wasm::WasmStorageHook;  // requires `storage-hooks` feature
use smg_data_connector::StorageFactoryConfig;

// Load the compiled WASM component
let wasm_bytes = std::fs::read("path/to/wasm_guest_storage_hook.component.wasm")?;
let hook = WasmStorageHook::new(&wasm_bytes)?;

// Pass the hook to the storage factory
let config = StorageFactoryConfig {
    backend: HistoryBackend::Postgres,
    postgres: Some(pg_config),
    hook: Some(Arc::new(hook)),
    ..Default::default()
};
let (conversations, items, responses) = create_storage(config)?;
```

## Schema Configuration

For the extra columns to be persisted, configure them in your schema:

```yaml
schema:
  responses:
    extra_columns:
      TENANT_ID:
        sql_type: "VARCHAR(128)"
      STORED_BY:
        sql_type: "VARCHAR(128)"
  conversations:
    extra_columns:
      TENANT_ID:
        sql_type: "VARCHAR(128)"
      CREATED_BY:
        sql_type: "VARCHAR(128)"
  conversation_items:
    extra_columns:
      TENANT_ID:
        sql_type: "VARCHAR(128)"
```

## WIT Interface

The storage hook world is defined in `crates/wasm/src/interface/storage/storage-hooks.wit` and exports:

- `storage-hook-before`: Called before storage operations to provide extra columns or reject
- `storage-hook-after`: Called after successful operations with the result and extra columns
