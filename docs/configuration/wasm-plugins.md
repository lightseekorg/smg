# WASM Plugins

WebAssembly (WASM) plugins enable dynamic extensibility for Shepherd Model Gateway (SMG), allowing you to deploy custom middleware logic without recompiling or restarting the router.

## Overview

WASM plugins provide a secure, portable way to extend SMG functionality:

- **Request/Response Transformation**: Modify headers, body content, or status codes
- **Custom Routing Logic**: Implement path-based routing decisions
- **Authentication/Authorization**: Add API key validation, token verification
- **Rate Limiting**: Implement custom rate limiting strategies
- **Request Tracking**: Add tracing headers, logging, and metrics
- **Content Filtering**: Validate or transform request/response bodies

### Key Benefits

- **Hot Deployment**: Add, remove, or update plugins without restart
- **Language Agnostic**: Write plugins in Rust, Go, C, or any language that compiles to WASM
- **Sandboxed Execution**: Plugins run in isolated environments with strict resource limits
- **High Performance**: Pre-compiled components with LRU caching minimize overhead

## Enabling WASM

Enable WASM support when starting the router:

```bash
smg --enable-wasm --worker-urls=http://0.0.0.0:30000 --port=3000
```

The `--enable-wasm` flag initializes the WASM runtime with default configuration:

| Setting | Default | Description |
|---------|---------|-------------|
| `max_memory_pages` | 1024 | Maximum memory (64KB per page = 64MB) |
| `max_execution_time_ms` | 1000 | Execution timeout per invocation |
| `max_stack_size` | 1MB | Maximum stack size |
| `thread_pool_size` | CPU cores (capped at 4) | Worker threads for WASM execution |
| `module_cache_size` | 10 | Cached modules per worker |
| `max_body_size` | 10MB | Maximum HTTP body size for processing |

> **Note:** These are the default runtime parameters. Currently, these values cannot be overridden via CLI flags.

## WASM Module API

### Add Modules

**POST /wasm**

Register one or more WASM modules.

**Request:**
```json
{
  "modules": [
    {
      "name": "auth-middleware",
      "file_path": "/absolute/path/to/module.component.wasm",
      "module_type": "Middleware",
      "attach_points": [{"Middleware": "OnRequest"}]
    }
  ]
}
```

**Response (200 OK):**
```json
{
  "modules": [
    {
      "name": "auth-middleware",
      "file_path": "/absolute/path/to/module.component.wasm",
      "module_type": "Middleware",
      "attach_points": [{"Middleware": "OnRequest"}],
      "add_result": {"Success": "550e8400-e29b-41d4-a716-446655440000"}
    }
  ]
}
```

**Response (400 Bad Request):**
```json
{
  "modules": [
    {
      "name": "invalid-module",
      "file_path": "/path/to/invalid.wasm",
      "module_type": "Middleware",
      "attach_points": [{"Middleware": "OnRequest"}],
      "add_result": {"Error": "Module file must have .wasm extension"}
    }
  ]
}
```

**Module Configuration Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Unique identifier for the module |
| `file_path` | string | Yes | Absolute path to the WASM component file |
| `module_type` | string | Yes | Currently only `"Middleware"` is supported |
| `attach_points` | array | Yes | Lifecycle hooks where the module executes |

**Supported Attach Points:**

| Attach Point | Description |
|--------------|-------------|
| `{"Middleware": "OnRequest"}` | Execute before forwarding to upstream |
| `{"Middleware": "OnResponse"}` | Execute after receiving upstream response |
| `{"Middleware": "OnError"}` | Execute on error (not yet implemented) |

### Remove Module

**DELETE /wasm/{module_uuid}**

Remove a registered module by its UUID.

```bash
curl -X DELETE http://localhost:3000/wasm/550e8400-e29b-41d4-a716-446655440000
```

**Response (200 OK):**
```
Module removed successfully
```

### List Modules

**GET /wasm**

List all registered modules with execution metrics.

**Response:**
```json
{
  "modules": [
    {
      "module_uuid": "550e8400-e29b-41d4-a716-446655440000",
      "module_meta": {
        "name": "auth-middleware",
        "file_path": "/path/to/module.component.wasm",
        "sha256_hash": "a1b2c3d4...",
        "size_bytes": 102400,
        "created_at": "2025-01-15T10:30:00.000000000Z",
        "last_accessed_at": "2025-01-15T12:45:30.123456789Z",
        "access_count": 1500,
        "attach_points": [{"Middleware": "OnRequest"}]
      }
    }
  ],
  "metrics": {
    "total_executions": 1500,
    "successful_executions": 1495,
    "failed_executions": 5,
    "total_execution_time_ms": 75000,
    "max_execution_time_ms": 150,
    "average_execution_time_ms": 50.0
  }
}
```

## Security Model

SMG implements multiple layers of security for WASM plugin execution.

### Path Validation

**Absolute Path Requirement:**
- All module paths must be absolute (start with `/`)
- Relative paths are rejected

**Path Traversal Protection:**
- `..` (parent directory) components are blocked
- `.` (current directory) components are blocked
- Prevents escaping intended directories

**File Extension Enforcement:**
- Only `.wasm` files are accepted (case-insensitive)
- Prevents loading arbitrary file types

### System Directory Blocking

The following directories are blocked to prevent information disclosure:

| Directory | Reason |
|-----------|--------|
| `/etc/` | System configuration |
| `/proc/` | Process information |
| `/sys/` | Kernel/device information |
| `/dev/` | Device files |
| `/boot/` | Boot configuration |
| `/root/` | Root user home |
| `/var/log/` | System logs |
| `/var/run/` | Runtime data |

### Symlink Validation

Symlinks are resolved and validated:

1. The symlink target is canonicalized
2. Blocked directory checks are re-applied to the resolved path
3. The resolved path must still have `.wasm` extension
4. Prevents symlink-based directory escape attacks

Example of blocked symlink:
```bash
# This will be rejected even if /safe/dir/module.wasm is a symlink
# that points to /etc/shadow
ln -s /etc/shadow /safe/dir/module.wasm
```

### Runtime Sandboxing

WASM modules execute in isolated wasmtime environments:

- **Memory Isolation**: Modules cannot access host memory
- **No Direct System Access**: File system, network, and other system resources are blocked by default
- **Resource Limits**: Memory, execution time, and stack size are enforced
- **WASI Sandboxing**: Standard WASI capabilities are restricted

**Resource Limits:**

| Resource | Limit | Description |
|----------|-------|-------------|
| Memory | 64MB (default) | Maximum 4GB configurable |
| Execution Time | 1s (default) | Maximum 5 minutes |
| Stack Size | 1MB (default) | Maximum 16MB |

### Duplicate Detection

Modules are deduplicated by SHA256 hash:
- Prevents registering the same module multiple times
- Each module file must be unique

## Plugin Development

### Prerequisites

- Rust toolchain (latest stable)
- WASM target: `rustup target add wasm32-wasip2`
- WASM tools: `cargo install wasm-tools`

### Supported Languages

Any language that compiles to WebAssembly Component Model:

| Language | Toolchain | Binding Generator |
|----------|-----------|-------------------|
| Rust | `wasm32-wasip2` target | `wit-bindgen` |
| Go | TinyGo | `wit-bindgen-go` |
| C/C++ | wasi-sdk | `wit-bindgen-c` |
| JavaScript | jco | `jco componentize` |
| Python | componentize-py | Built-in |

### WebAssembly Interface Types (WIT)

SMG uses the WebAssembly Component Model with a defined interface. The interface specification (`spec.wit`):

```wit
package smg:gateway;

interface middleware-types {
  record header { name: string, value: string }

  record request {
    method: string,
    path: string,
    query: string,
    headers: list<header>,
    body: list<u8>,
    request-id: string,
    now-epoch-ms: u64,
  }

  record response {
    status: u16,
    headers: list<header>,
    body: list<u8>,
  }

  record modify-action {
    status: option<u16>,
    headers-set: list<header>,
    headers-add: list<header>,
    headers-remove: list<string>,
    body-replace: option<list<u8>>,
  }

  variant action {
    continue,
    reject(u16),
    modify(modify-action),
  }
}

interface middleware-on-request {
  use middleware-types.{request, action};
  on-request: func(req: request) -> action;
}

interface middleware-on-response {
  use middleware-types.{response, action};
  on-response: func(resp: response) -> action;
}

world smg {
  export middleware-on-request;
  export middleware-on-response;
}
```

### Action Types

Plugins return one of three actions:

| Action | Description |
|--------|-------------|
| `Continue` | Pass through without modification |
| `Reject(status_code)` | Stop processing, return error response |
| `Modify(modify_action)` | Apply modifications to request/response |

### Building a Rust Plugin

**1. Create Project Structure:**

```bash
mkdir my-plugin && cd my-plugin
cargo init --lib
```

**2. Configure Cargo.toml:**

```toml
[package]
name = "my-plugin"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
wit-bindgen = { version = "0.21", features = ["macros"] }
```

**3. Implement the Plugin (src/lib.rs):**

```rust
wit_bindgen::generate!({
    path: "path/to/smg/wasm/src/interface",
    world: "smg",
});

use exports::smg::gateway::{
    middleware_on_request::Guest as OnRequestGuest,
    middleware_on_response::Guest as OnResponseGuest,
};
use smg::gateway::middleware_types::{Action, Request, Response};

struct MyPlugin;

impl OnRequestGuest for MyPlugin {
    fn on_request(req: Request) -> Action {
        // Your request processing logic
        Action::Continue
    }
}

impl OnResponseGuest for MyPlugin {
    fn on_response(resp: Response) -> Action {
        // Your response processing logic
        Action::Continue
    }
}

export!(MyPlugin);
```

**4. Build the Component:**

```bash
# Build WASM module
cargo build --target wasm32-wasip2 --release

# Wrap into component format
wasm-tools component new \
    target/wasm32-wasip2/release/my_plugin.wasm \
    -o my_plugin.component.wasm
```

**5. Deploy:**

```bash
curl -X POST http://localhost:3000/wasm \
  -H "Content-Type: application/json" \
  -d '{
    "modules": [{
      "name": "my-plugin",
      "file_path": "/absolute/path/to/my_plugin.component.wasm",
      "module_type": "Middleware",
      "attach_points": [{"Middleware": "OnRequest"}]
    }]
  }'
```

## Examples

### Request Transformation Plugin

Add tracking headers to all requests:

```rust
impl OnRequestGuest for LoggingPlugin {
    fn on_request(req: Request) -> Action {
        let modify_action = ModifyAction {
            status: None,
            headers_set: vec![],
            headers_add: vec![
                Header {
                    name: "x-request-id".to_string(),
                    value: req.request_id.clone(),
                },
                Header {
                    name: "x-processed-at".to_string(),
                    value: req.now_epoch_ms.to_string(),
                },
            ],
            headers_remove: vec![],
            body_replace: None,
        };
        Action::Modify(modify_action)
    }
}
```

### Authentication Plugin

Validate API keys for protected routes:

```rust
const VALID_API_KEY: &str = "secret-key-12345";

fn find_header(headers: &[Header], name: &str) -> Option<String> {
    headers
        .iter()
        .find(|h| h.name.eq_ignore_ascii_case(name))
        .map(|h| h.value.clone())
}

impl OnRequestGuest for AuthPlugin {
    fn on_request(req: Request) -> Action {
        // Only check /api and /v1 routes
        if !req.path.starts_with("/api") && !req.path.starts_with("/v1") {
            return Action::Continue;
        }

        // Extract API key from Authorization header or x-api-key
        let api_key = find_header(&req.headers, "authorization")
            .and_then(|h| h.strip_prefix("Bearer ").map(String::from))
            .or_else(|| find_header(&req.headers, "x-api-key"));

        match api_key {
            Some(key) if key == VALID_API_KEY => Action::Continue,
            _ => Action::Reject(401), // Unauthorized
        }
    }
}
```

### Response Filtering Plugin

Convert error status codes:

```rust
impl OnResponseGuest for ErrorConverterPlugin {
    fn on_response(resp: Response) -> Action {
        // Convert 500 Internal Server Error to 503 Service Unavailable
        if resp.status == 500 {
            let modify_action = ModifyAction {
                status: Some(503),
                headers_set: vec![],
                headers_add: vec![
                    Header {
                        name: "x-original-status".to_string(),
                        value: "500".to_string(),
                    },
                ],
                headers_remove: vec![],
                body_replace: None,
            };
            return Action::Modify(modify_action);
        }
        Action::Continue
    }
}
```

### Custom Validation Plugin

Validate request body content:

```rust
impl OnRequestGuest for ValidationPlugin {
    fn on_request(req: Request) -> Action {
        // Only validate POST/PUT requests
        if req.method != "POST" && req.method != "PUT" {
            return Action::Continue;
        }

        // Check for required content-type header
        let content_type = find_header(&req.headers, "content-type");
        if content_type.as_deref() != Some("application/json") {
            return Action::Reject(415); // Unsupported Media Type
        }

        // Validate body is not empty
        if req.body.is_empty() {
            return Action::Reject(400); // Bad Request
        }

        Action::Continue
    }
}
```

## Troubleshooting

### Common Errors

**"Module file path must be absolute"**
- Use absolute paths starting with `/`
- Incorrect: `./module.wasm` or `module.wasm`
- Correct: `/home/user/modules/module.wasm`

**"Path traversal (..) not allowed"**
- Remove `..` components from the path
- Use canonical absolute paths

**"Access to /etc/ directory is not allowed"**
- Move WASM modules to a non-system directory
- Blocked directories: `/etc/`, `/proc/`, `/sys/`, `/dev/`, `/boot/`, `/root/`, `/var/log/`, `/var/run/`

**"Module file must have .wasm extension"**
- Rename the file to end with `.wasm`
- Ensure the file is actually a WASM binary

**"Invalid WASM component"**
- The file must be in Component Model format, not plain WASM module
- Wrap with: `wasm-tools component new input.wasm -o output.component.wasm`
- Error hint: "use 'wasm-tools component new' to wrap the WASM module into a component"

**"Duplicate SHA256 hash detected"**
- A module with identical content is already registered
- Remove the existing module first, or use a different module file

**"WASM module manager not initialized"**
- Start SMG with `--enable-wasm` flag
- Check logs for WASM runtime initialization errors

### Debugging Plugins

**Enable Debug Logging:**
```bash
RUST_LOG=smg::wasm=debug smg --enable-wasm ...
```

**Check Module Status:**
```bash
curl http://localhost:3000/wasm | jq
```

**Test Plugin Locally:**
```bash
# Validate component format
wasm-tools validate my_plugin.component.wasm

# Print component structure
wasm-tools print my_plugin.component.wasm | head -20
```

**Common Plugin Issues:**

1. **Plugin not executing**: Verify attach points match the request lifecycle
2. **Headers not appearing**: Check header names are lowercase
3. **Timeout errors**: Reduce computation or increase `max_execution_time_ms`
4. **Memory errors**: Reduce allocations or increase `max_memory_pages`

### Performance Tuning

**Module Caching:**
- Modules are cached per worker thread
- Increase `module_cache_size` for many modules
- Default: 10 modules per worker

**Thread Pool Sizing:**
- Default scales with CPU cores (capped at 4)
- Increase for high-throughput scenarios
- Monitor execution queue depth

**Memory Limits:**
- Default 64MB should suffice for most plugins
- Increase only if plugins require large allocations
- Maximum configurable: 4GB

## See Also

- [Architecture Overview](/docs/concepts/architecture/overview.md) - System architecture
- [Examples](/examples/wasm/) - Complete example plugins
- WIT Specification - The interface definition is embedded in the Rust source code; see the [WebAssembly Interface Types (WIT)](#webassembly-interface-types-wit) section above for the full specification
