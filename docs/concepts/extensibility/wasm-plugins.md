---
title: WASM Plugins
---

# WASM Plugins

WebAssembly (WASM) plugins enable custom middleware logic in SMG's request pipeline without recompiling or restarting the gateway.

---

## Overview

<div class="grid" markdown>

<div class="card" markdown>

### :material-puzzle: Dynamic Extension

Deploy custom logic at runtime via REST API. Add, update, or remove plugins with immediate effect—no gateway restart required.

</div>

<div class="card" markdown>

### :material-shield-lock: Sandboxed Execution

Plugins run in isolated WASM environments with strict resource limits. No access to host memory, filesystem, or network.

</div>

<div class="card" markdown>

### :material-translate: Language Agnostic

Write plugins in Rust, Go, C, or any language that compiles to WebAssembly Component Model.

</div>

<div class="card" markdown>

### :material-lightning-bolt: High Performance

Pre-compiled components with LRU caching. Typical overhead: <1ms per invocation.

</div>

</div>

---

## How It Works

WASM plugins execute in SMG's **middleware layer**, intercepting requests before they reach workers and responses before they return to clients.

<div class="architecture-diagram">
  <img src="../../../assets/images/wasm-plugins.svg" alt="WASM Plugin Pipeline Architecture">
</div>

### Plugin Chain Execution

Plugins execute in deployment order. Each plugin receives the request/response and returns an action:

1. **Continue**: Pass through to the next plugin (or worker)
2. **Reject(status)**: Stop processing and return an error response immediately
3. **Modify(changes)**: Apply transformations and continue

If any plugin returns `Reject`, subsequent plugins are skipped and the error response is returned to the client.

---

## Attach Points

Plugins register at specific points in the request lifecycle:

| Attach Point | When | Use Cases |
|--------------|------|-----------|
| **OnRequest** | Before forwarding to worker | Authentication, rate limiting, validation, header injection |
| **OnResponse** | After receiving worker response | Response transformation, error normalization, logging |

---

## Example Plugins

SMG includes ready-to-use example plugins demonstrating common middleware patterns:

<div class="grid" markdown>

<div class="card" markdown>

### :material-key: auth-middleware

API key authentication for `/api` and `/v1` routes.

- Validates `Authorization` or `x-api-key` header
- Returns **401 Unauthorized** on failure
- Attach point: **OnRequest**

</div>

<div class="card" markdown>

### :material-speedometer: ratelimit-middleware

Per-identifier rate limiting with configurable thresholds.

- 60 requests/minute default
- Tracks by API key, IP, or request ID
- Returns **429 Too Many Requests** when exceeded
- Attach point: **OnRequest**

</div>

<div class="card" markdown>

### :material-file-document: logging-middleware

Request tracking and response transformation.

- Adds `x-request-id`, `x-wasm-processed`, `x-processed-at`
- Converts 500 → 503 for better client handling
- Attach points: **OnRequest** and **OnResponse**

</div>

</div>

Find complete source code and build instructions in [`examples/wasm/`](https://github.com/lightseekorg/smg/tree/main/examples/wasm).

---

## Plugin Development

### Interface

Plugins implement the SMG middleware interface using the WebAssembly Component Model:

```rust
// OnRequest: Called before forwarding to worker
fn on_request(req: Request) -> Action {
    // Validate, modify, or reject the request
    Action::Continue
}

// OnResponse: Called after receiving worker response
fn on_response(resp: Response) -> Action {
    // Transform or log the response
    Action::Continue
}
```

### Actions

| Action | Effect | Example |
|--------|--------|---------|
| `Action::Continue` | Pass through unmodified | Logging, metrics |
| `Action::Reject(401)` | Return error immediately | Auth failure |
| `Action::Modify(changes)` | Apply transformations | Add headers, rewrite body |

### Request Context

Plugins receive rich context for decision-making:

| Field | Description |
|-------|-------------|
| `method` | HTTP method (GET, POST, etc.) |
| `path` | Request path |
| `headers` | All request headers |
| `body` | Request body (if present) |
| `request_id` | Unique request identifier |
| `now_epoch_ms` | Current timestamp |

---

## Configuration

### Enabling WASM Support

```bash
smg --enable-wasm --worker-urls http://worker:8000
```

### Runtime Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `max_memory_pages` | 1024 | Maximum memory (64KB per page = 64MB) |
| `max_execution_time_ms` | 1000 | Execution timeout per invocation |
| `module_cache_size` | 10 | Cached compiled modules per worker |

---

## Module Management

Plugins are managed via the Admin API at runtime:

### Deploy Plugins

```bash
curl -X POST http://localhost:3000/wasm \
  -H "Content-Type: application/json" \
  -d '{
    "modules": [
      {
        "name": "auth-middleware",
        "file_path": "/plugins/auth.component.wasm",
        "module_type": "Middleware",
        "attach_points": [{"Middleware": "OnRequest"}]
      },
      {
        "name": "logging-middleware",
        "file_path": "/plugins/logging.component.wasm",
        "module_type": "Middleware",
        "attach_points": [
          {"Middleware": "OnRequest"},
          {"Middleware": "OnResponse"}
        ]
      }
    ]
  }'
```

### List Plugins

```bash
curl http://localhost:3000/wasm
```

### Remove Plugin

```bash
curl -X DELETE http://localhost:3000/wasm/{uuid}
```

---

## Security Model

WASM plugins execute in a sandboxed environment with multiple protection layers:

| Layer | Protection |
|-------|------------|
| **Path Validation** | Only absolute paths, `.wasm` extension required, system directories blocked |
| **Runtime Sandboxing** | No access to host memory, filesystem, network, or system calls |
| **Resource Limits** | Memory caps, execution timeouts, stack size limits |
| **Deduplication** | SHA256 hash prevents duplicate module loading |

**Blocked directories**: `/etc/`, `/proc/`, `/sys/`, `/dev/`, `/boot/`, `/root/`, `/var/log/`, `/var/run/`

---

## Performance

| Operation | Typical Time |
|-----------|-------------|
| Cache hit + execution | <1ms |
| Cache miss (first load) | 10-50ms |
| Simple plugin (auth check) | 0.1-0.3ms |
| Complex plugin (body transform) | 0.5-2ms |

---

## What's Next?

<div class="grid" markdown>

<div class="card" markdown>

### :material-github: Example Plugins

Complete source code with build instructions.

[View Examples →](https://github.com/lightseekorg/smg/tree/main/examples/wasm)

</div>

<div class="card" markdown>

### :material-tools: MCP Integration

Connect models to external tools.

[Model Context Protocol →](mcp.md)

</div>

</div>
