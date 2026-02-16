//! WASM Guest Auth Example for Shepherd Model Gateway
//!
//! This example demonstrates API key authentication middleware
//! for Shepherd Model Gateway using the WebAssembly Component Model.
//!
//! Features:
//! - API Key authentication

wit_bindgen::generate!({
    path: "../../../wasm/src/interface",
    world: "smg",
});

use exports::smg::gateway::{
    middleware_on_request::Guest as OnRequestGuest,
    middleware_on_request_headers::Guest as OnRequestHeadersGuest,
    middleware_on_response::Guest as OnResponseGuest,
    middleware_on_response_headers::Guest as OnResponseHeadersGuest,
};
use smg::gateway::middleware_types::{Action, Request, RequestHeaders, Response, ResponseHeaders};

/// Expected API Key (in production, this should be passed as configuration)
const EXPECTED_API_KEY: &str = "secret-api-key-12345";

/// Main middleware implementation
struct Middleware;

// Helper function to find header value
fn find_header_value(
    headers: &[smg::gateway::middleware_types::Header],
    name: &str,
) -> Option<String> {
    headers
        .iter()
        .find(|h| h.name.eq_ignore_ascii_case(name))
        .map(|h| h.value.clone())
}

// Implement on-request interface
impl OnRequestGuest for Middleware {
    fn on_request(req: Request) -> Action {
        // API Key Authentication
        // Check for API key in Authorization header for /api routes
        if req.path.starts_with("/api") || req.path.starts_with("/v1") {
            let api_key = find_header_value(&req.headers, "authorization")
                .and_then(|h| {
                    h.strip_prefix("Bearer ")
                        .or_else(|| h.strip_prefix("ApiKey "))
                        .map(|s| s.to_string())
                })
                .or_else(|| find_header_value(&req.headers, "x-api-key"));

            // Reject if API key is missing or invalid
            if api_key.as_deref() != Some(EXPECTED_API_KEY) {
                return Action::Reject(401);
            }
        }

        // Authentication passed, continue processing
        Action::Continue
    }
}

// Implement on-response interface (empty - not used for auth)
impl OnResponseGuest for Middleware {
    fn on_response(_resp: Response) -> Action {
        Action::Continue
    }
}

// Headers-only stubs (required by the WIT world, unused with default body_policy)
impl OnRequestHeadersGuest for Middleware {
    fn on_request_headers(_req: RequestHeaders) -> Action {
        Action::Continue
    }
}

impl OnResponseHeadersGuest for Middleware {
    fn on_response_headers(_resp: ResponseHeaders) -> Action {
        Action::Continue
    }
}

// Export the component
export!(Middleware);