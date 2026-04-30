---
title: Interceptors
---

# Interceptors

Interceptors are vendor-neutral extension points that fire at well-defined moments in the Responses request lifecycle. They allow custom logic — memory, audit logging, rate limiting, PII redaction — to ship as separate crates that SMG core does not need to know about.

---

## Overview

<div class="grid" markdown>

<div class="card" markdown>

### :material-puzzle: Two-Phase Hooks

Interceptors observe and optionally mutate requests at `before_model` and `after_persist`. Each phase has a typed context with the data needed for that lifecycle moment.

</div>

<div class="card" markdown>

### :material-shield-check: Non-Fatal by Design

Trait methods return `()`, never `Result`. The registry catches panics and logs them. A buggy interceptor cannot fail an SMG request.

</div>

<div class="card" markdown>

### :material-package-variant: Vendor-Neutral

Interceptors live in their own crates. SMG core is unaware of any specific extension. Operators opt in via build features and YAML configuration.

</div>

<div class="card" markdown>

### :material-lightning-bolt: Zero Cost When Unused

An empty registry skips all hook compute (turn counting, metadata construction). Operators with no extensions configured pay only one boolean check per phase.

</div>

</div>

---

## Phases

Interceptors fire at exactly two moments in the Responses request lifecycle:

| Phase | When | Use cases |
|-------|------|-----------|
| **`before_model`** | After history is loaded; before the request reaches the model | Inject memory context, enforce policy, mutate the request input |
| **`after_persist`** | After conversation items + response are persisted successfully | Enqueue async work (memory consolidation, summarization), write audit rows, emit metrics |

!!! info "Firing cadence"
    `before_model` fires **once per request**, not per inner MCP-tool-loop
    iteration. `after_persist` fires once for each successful persistence —
    that is, once per Responses turn **and** once per
    `POST /v1/conversations/{id}/items` batch.

---

## The Trait

Implementors register against a single trait:

```rust
use async_trait::async_trait;
use smg_extensions::{AfterPersistCtx, BeforeModelCtx, ResponsesInterceptor};

#[async_trait]
pub trait ResponsesInterceptor: Send + Sync + 'static {
    /// Stable identifier for diagnostics and panic logging.
    fn name(&self) -> &'static str;

    async fn before_model(&self, ctx: &mut BeforeModelCtx<'_>);

    async fn after_persist(&self, ctx: &AfterPersistCtx<'_>);
}
```

Both `before_model` and `after_persist` have default no-op implementations, so an interceptor only needs to override the phase it cares about.

### Recommended destructure pattern

`BeforeModelCtx` and `AfterPersistCtx` are both `#[non_exhaustive]`. New fields may be added in future versions without a major version bump. **Always destructure with `..`** so future additions don't break compilation:

```rust
async fn before_model(&self, ctx: &mut BeforeModelCtx<'_>) {
    let BeforeModelCtx { headers, request, conversation_id, .. } = ctx;
    // ...
}
```

---

## Configuration

Operators enable interceptors via the `extensions:` block in `server.yaml`:

```yaml
# server.yaml
extensions:
  - kind: my-extension
    # Arbitrary YAML consumed by the extension itself.
    setting: value
    nested:
      key: value
```

The `extensions:` field is parsed by SMG core into a list of `kind`/`config` pairs. Each extension crate parses its own `config` payload from a `serde_yml::Value`.

An empty `extensions: []` list (or omitting the field entirely) means zero interceptors are registered. SMG core stays unaware of any specific extension.

!!! tip "Two gates: build-time and runtime"
    Default-off feature flags in `model_gateway/Cargo.toml` are the
    **build-time** gate — extensions you never enable are not linked into
    the binary. The `extensions:` YAML block is the **runtime** gate —
    even a built-in extension stays inert until an operator opts in.

---

## Writing an Interceptor

1. **Create a new crate** that depends on `smg-extensions`.
2. **Implement `ResponsesInterceptor`** for your type. Override only the phases you need; the rest default to no-ops.
3. **Register a `match` arm** in `model_gateway/src/app_context.rs` (inside `AppContextBuilder::from_config`) under a feature flag — for example:

    ```rust
    #[cfg(feature = "my-extension")]
    "my-extension" => {
        registry_builder.register(my_extension::build(&spec.config)?);
    }
    ```

4. **Document the YAML schema** consumed by your `build()` function in your crate's README.

Default-off feature flags in `model_gateway/Cargo.toml` keep extension dependencies optional at compile time. Operators who don't enable the feature build a smaller binary with no extension code linked in.

---

## Error Model

Interceptor methods return `()` — there is no `Result`. Internal failures inside an implementation should be logged with the implementor's own `tracing` calls.

Panics propagating out of either phase are caught by the registry via `catch_unwind` and logged at warn level. Execution continues to the next interceptor and to the next request lifecycle step.

| Failure mode | Effect on SMG request |
|--------------|----------------------|
| Implementor logs an error and returns | Request continues normally |
| Implementor panics | Registry catches, logs at warn, request continues |
| Implementor blocks indefinitely | Request blocks (interceptors are awaited inline) |

This is a deliberate trade-off: operators get a strict guarantee that no extension can **fail** a request, in exchange for explicit error handling inside each interceptor.

---

## What's Next?

<div class="grid" markdown>

<div class="card" markdown>

### :material-book-open-variant: smg-extensions API

Trait, context types, and registry reference.

[View on docs.rs →](https://docs.rs/smg-extensions)

</div>

<div class="card" markdown>

### :material-puzzle-outline: WASM Plugins

Sandboxed extension model for request/response transformations.

[WASM Plugins →](wasm-plugins.md)

</div>

<div class="card" markdown>

### :material-tools: MCP Integration

Connect models to external tools.

[Model Context Protocol →](mcp.md)

</div>

</div>
