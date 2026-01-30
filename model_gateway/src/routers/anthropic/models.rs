//! Anthropic Models API implementation

use axum::{
    body::Body,
    extract::Request,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use tracing::{debug, warn};

use super::AnthropicRouter;

/// Handle /v1/models request
pub async fn handle_list_models(
    router: &AnthropicRouter,
    req: Request<Body>,
) -> Response {
    debug!("Handling list models request");

    // Extract headers from the request
    let headers = req.headers();

    // Get a worker - prefer any healthy worker
    let workers = router.context().worker_registry.get_all();

    if workers.is_empty() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "No workers available",
        )
            .into_response();
    }

    let worker = &workers[0];

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

/// Check if header should be propagated to backend
fn should_propagate_header(key: &str) -> bool {
    matches!(
        key.to_lowercase().as_str(),
        "authorization" | "x-api-key" | "anthropic-version" | "anthropic-beta"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_propagate_header() {
        assert!(should_propagate_header("authorization"));
        assert!(should_propagate_header("Authorization"));
        assert!(should_propagate_header("x-api-key"));
        assert!(should_propagate_header("X-Api-Key"));
        assert!(should_propagate_header("X-API-KEY"));
        assert!(should_propagate_header("anthropic-version"));
        assert!(should_propagate_header("anthropic-beta"));

        assert!(!should_propagate_header("cookie"));
        assert!(!should_propagate_header("user-agent"));
        assert!(!should_propagate_header("host"));
    }
}
