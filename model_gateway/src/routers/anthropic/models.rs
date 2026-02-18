//! Anthropic Models API implementation

use axum::{
    body::Body,
    extract::Request,
    response::{IntoResponse, Response},
};
use tracing::{debug, warn};

use super::{
    utils::{read_response_body_limited, should_propagate_header, ReadBodyResult},
    worker::get_healthy_anthropic_workers,
    AnthropicRouter,
};
use crate::routers::error;

const MAX_RESPONSE_SIZE: usize = 10 * 1024 * 1024;

pub async fn handle_list_models(router: &AnthropicRouter, req: Request<Body>) -> Response {
    debug!("Handling list models request");

    let headers = req.headers();

    // SECURITY: Filter by Anthropic provider in multi-provider setups
    let healthy_workers = get_healthy_anthropic_workers(&router.context().worker_registry);

    if healthy_workers.is_empty() {
        warn!("No healthy Anthropic workers available for /v1/models request");
        return error::service_unavailable("no_workers", "No healthy Anthropic workers available");
    }

    let worker = &healthy_workers[0];

    let url = format!("{}/v1/models", worker.url());

    debug!("Forwarding list models request to: {}", url);

    let mut req_builder = router.http_client().get(&url);

    for (key, value) in headers {
        if should_propagate_header(key.as_str()) {
            req_builder = req_builder.header(key, value);
        }
    }

    match req_builder.send().await {
        Ok(response) => {
            let status = response.status();

            // SECURITY: Check content-length header first to reject obviously oversized responses
            if let Some(content_length) = response.content_length() {
                if content_length > MAX_RESPONSE_SIZE as u64 {
                    warn!(
                        "Response content-length too large: {} bytes (max {})",
                        content_length, MAX_RESPONSE_SIZE
                    );
                    return error::internal_error(
                        "response_too_large",
                        "Response body exceeds maximum size",
                    );
                }
            }

            // SECURITY: Read body incrementally with size limit
            match read_response_body_limited(response, MAX_RESPONSE_SIZE).await {
                ReadBodyResult::Ok(body) => (status, body).into_response(),
                ReadBodyResult::TooLarge => {
                    warn!("Response body too large (max {} bytes)", MAX_RESPONSE_SIZE);
                    error::internal_error(
                        "response_too_large",
                        "Response body exceeds maximum size",
                    )
                }
                ReadBodyResult::Error(e) => {
                    warn!("Failed to read response body: {}", e);
                    error::internal_error("read_error", format!("Failed to read response: {}", e))
                }
            }
        }
        Err(e) => {
            warn!("Failed to forward list models request: {}", e);
            error::bad_gateway(
                "forward_failed",
                format!("Failed to forward request: {}", e),
            )
        }
    }
}
