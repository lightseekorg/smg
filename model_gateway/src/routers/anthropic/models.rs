//! Anthropic Models API implementation

use axum::{
    body::Body,
    extract::Request,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use tracing::{debug, warn};

use super::{utils::should_propagate_header, AnthropicRouter};

/// Handle /v1/models request
pub async fn handle_list_models(
    router: &AnthropicRouter,
    req: Request<Body>,
) -> Response {
    debug!("Handling list models request");

    // Extract headers from the request
    let headers = req.headers();

    // Get a healthy worker to avoid credential leakage and ensure reliability
    let healthy_workers = router.context().worker_registry.get_workers_filtered(
        None, // model_id
        None, // provider
        None, // connection_mode
        None, // runtime_type
        true, // healthy_only
    );

    if healthy_workers.is_empty() {
        warn!("No healthy workers available for /v1/models request");
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "No healthy workers available",
        )
            .into_response();
    }

    let worker = &healthy_workers[0];

    // Forward request to backend
    let url = format!("{}/v1/models", worker.url());

    debug!("Forwarding list models request to: {}", url);

    let mut req_builder = router
        .http_client()
        .get(&url);

    // Propagate relevant headers
    for (key, value) in headers {
        if should_propagate_header(key.as_str()) {
            req_builder = req_builder.header(key, value);
        }
    }

    // Send request
    match req_builder.send().await {
        Ok(response) => {
            let status = response.status();
            let body = match response.text().await {
                Ok(text) => text,
                Err(e) => {
                    warn!("Failed to read response body: {}", e);
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Failed to read response: {}", e),
                    )
                        .into_response();
                }
            };

            // Return response with same status and body
            (status, body).into_response()
        }
        Err(e) => {
            warn!("Failed to forward list models request: {}", e);
            (
                StatusCode::BAD_GATEWAY,
                format!("Failed to forward request: {}", e),
            )
                .into_response()
        }
    }
}
