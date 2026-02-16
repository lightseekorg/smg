//! WASM Guest Auth (Streaming) Example for Shepherd Model Gateway
//!
//! This example demonstrates **headers-only** middleware — a streaming
//! optimisation that avoids buffering request/response bodies entirely.
//!
//! When a module is deployed with `body_policy: "HeadersOnly"`, the gateway
//! skips `to_bytes` and passes only headers to the WASM component. The body
//! streams through untouched, dramatically reducing latency (TTFB) for large
//! payloads and SSE/streaming responses.
//!
//! # When to use headers-only middleware
//! Any middleware that only inspects or modifies **headers** (not the body)
//! should use this pattern:
//! - API key / JWT validation
//! - Request tracing / correlation-ID injection
//! - Rate limiting by header-based identifiers
//! - CORS header injection
//!
//! # Key difference from `wasm-guest-auth`
//! | Aspect       | `wasm-guest-auth`              | `wasm-guest-auth-streaming`        |
//! |--------------|--------------------------------|------------------------------------|
//! | Interfaces   | `on-request` / `on-response`   | `on-request-headers` / `on-response-headers` |
//! | Body access  | Full body available            | No body — streams through          |
//! | `body_policy` | `Required` (default)          | `HeadersOnly`                      |
//! | TTFB impact  | Blocked until body buffered    | Near-zero overhead                 |

wit_bindgen::generate!({
    path: "../../../wasm/src/interface",
    world: "smg",
});

use exports::smg::gateway::{
    // Headers-only interfaces (our primary implementation)
    middleware_on_request_headers::Guest as OnRequestHeadersGuest,
    middleware_on_response_headers::Guest as OnResponseHeadersGuest,
    // Full-body interfaces (stub — required by the world but never called
    // when the module is deployed with body_policy: HeadersOnly)
    middleware_on_request::Guest as OnRequestGuest,
    middleware_on_response::Guest as OnResponseGuest,
};
use smg::gateway::middleware_types::{Action, Request, RequestHeaders, Response, ResponseHeaders};

/// Expected API Key (in production, pass via configuration or host import)
const EXPECTED_API_KEY: &str = "secret-api-key-12345";

struct Middleware;


// Helper

fn find_header_value(
    headers: &[smg::gateway::middleware_types::Header],
    name: &str,
) -> Option<String> {
    headers
        .iter()
        .find(|h| h.name.eq_ignore_ascii_case(name))
        .map(|h| h.value.clone())
}

fn check_api_key(headers: &[smg::gateway::middleware_types::Header], path: &str) -> Action {
    if !path.starts_with("/api") && !path.starts_with("/v1") {
        return Action::Continue;
    }

    let api_key = find_header_value(headers, "authorization")
        .and_then(|h| {
            h.strip_prefix("Bearer ")
                .or_else(|| h.strip_prefix("ApiKey "))
                .map(|s| s.to_string())
        })
        .or_else(|| find_header_value(headers, "x-api-key"));

    if api_key.as_deref() != Some(EXPECTED_API_KEY) {
        return Action::Reject(401);
    }

    Action::Continue
}

// Headers-only interfaces  (the actual logic runs here)

impl OnRequestHeadersGuest for Middleware {
    fn on_request_headers(req: RequestHeaders) -> Action {
        check_api_key(&req.headers, &req.path)
    }
}

impl OnResponseHeadersGuest for Middleware {
    fn on_response_headers(_resp: ResponseHeaders) -> Action {
        Action::Continue
    }
}


// Full-body stubs  (required by the WIT world, never invoked at runtime
//                   when the module is deployed with body_policy: HeadersOnly)


impl OnRequestGuest for Middleware {
    fn on_request(req: Request) -> Action {
        // Fallback — reuse the same logic
        check_api_key(&req.headers, &req.path)
    }
}

impl OnResponseGuest for Middleware {
    fn on_response(_resp: Response) -> Action {
        Action::Continue
    }
}

// Export the component
export!(Middleware);
