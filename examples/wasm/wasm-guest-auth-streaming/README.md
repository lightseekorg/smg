# WASM Guest Auth — Streaming (Headers-Only)

API key authentication middleware that uses **headers-only interfaces** to avoid body buffering, enabling streaming pass-through for request and response bodies.

## Why Headers-Only?

Standard WASM middleware buffers the entire request/response body before executing. For auth middleware that only inspects headers, this is unnecessary overhead — it blocks Time-To-First-Byte (TTFB) especially for large or streaming responses (SSE).

By implementing `on-request-headers` / `on-response-headers` instead of `on-request` / `on-response`, and deploying with `body_policy: "HeadersOnly"`, the gateway **skips body buffering entirely**.

**Result:** TTFB drops from hundreds of milliseconds to ~90µs.

## Building

```bash
cd examples/wasm/wasm-guest-auth-streaming
./build.sh
```

## Deploying

```bash
curl -X POST http://localhost:3000/wasm \
  -H "Content-Type: application/json" \
  -d '{
    "modules": [
      {
        "name": "auth-streaming",
        "file_path": "/path/to/wasm_guest_auth_streaming.component.wasm",
        "module_type": "Middleware",
        "attach_points": [{"Middleware": "OnRequest"}],
        "body_policy": "HeadersOnly"
      }
    ]
  }'
```

> **Important:** The `body_policy: "HeadersOnly"` field is what tells the gateway to skip buffering. Without it, the default `Required` policy is used and the body is still buffered — even though the module only uses headers.

## Testing

```bash
# Should be rejected (no API key)
curl -v http://localhost:3000/v1/chat/completions

# Should pass (valid API key)
curl -v -H "Authorization: Bearer <YOUR_API_KEY>" \
  http://localhost:3000/v1/chat/completions
```

## Customization

- Edit `EXPECTED_API_KEY` in `src/lib.rs` for your own key
- Modify `check_api_key()` to add JWT validation, multi-tenant keys, etc.
