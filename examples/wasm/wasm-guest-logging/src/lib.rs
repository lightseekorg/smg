//! WASM Guest Logging Example for Shepherd Model Gateway
//!
//! This example demonstrates logging and tracing middleware
//! for Shepherd Model Gateway using the WebAssembly Component Model.
//!
//! Features:
//! - Request tracking and tracing headers
//! - Response status code conversion

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
use smg::gateway::middleware_types::{
    Action, Header, ModifyAction, Request, RequestHeaders, Response, ResponseHeaders,
};

/// Main middleware implementation
struct Middleware;

// Helper function to create header
fn create_header(name: &str, value: &str) -> Header {
    Header {
        name: name.to_string(),
        value: value.to_string(),
    }
}

// Implement on-request interface
impl OnRequestGuest for Middleware {
    fn on_request(req: Request) -> Action {
        let mut modify_action = ModifyAction {
            status: None,
            headers_set: vec![],
            headers_add: vec![],
            headers_remove: vec![],
            body_replace: None,
        };

        // Request Logging and Tracing
        // Add tracing headers with request ID
        modify_action
            .headers_add
            .push(create_header("x-request-id", &req.request_id));
        modify_action
            .headers_add
            .push(create_header("x-wasm-processed", "true"));
        modify_action.headers_add.push(create_header(
            "x-processed-at",
            &req.now_epoch_ms.to_string(),
        ));

        // Add custom header for API requests
        if req.path.starts_with("/api") || req.path.starts_with("/v1") {
            modify_action
                .headers_add
                .push(create_header("x-api-route", "true"));
        }

        Action::Modify(modify_action)
    }
}

// Implement on-response interface
impl OnResponseGuest for Middleware {
    fn on_response(resp: Response) -> Action {
        // Status code conversion: Convert 500 to 503 for better client handling
        if resp.status == 500 {
            let modify_action = ModifyAction {
                status: Some(503),
                headers_set: vec![],
                headers_add: vec![],
                headers_remove: vec![],
                body_replace: None,
            };
            Action::Modify(modify_action)
        } else {
            // No modification needed
            Action::Continue
        }
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
