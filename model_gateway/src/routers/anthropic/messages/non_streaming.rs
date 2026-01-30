//! Non-streaming Messages API handler
//!
//! This module handles all aspects of non-streaming Messages API requests:
//! - Worker selection via routing policies
//! - Request forwarding to Anthropic-compatible workers
//! - Response parsing and error handling
//! - Metrics and observability

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::{
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use tracing::{debug, error, info, warn};

use crate::{
    core::Worker,
    protocols::messages::{CreateMessageRequest, Message},
    routers::anthropic::AnthropicRouter,
};

#[cfg(test)]
use crate::protocols::messages::{InputMessage, Role};

// ============================================================================
// Worker Selection
// ============================================================================

/// Select worker for Messages API request using configured policy
async fn select_worker(
    router: &AnthropicRouter,
    model_id: Option<&str>,
    request: &CreateMessageRequest,
) -> Result<Arc<dyn Worker>, String> {
    let model = model_id.unwrap_or(&request.model);

    // Delegate to router's worker selection method (matches OpenAI pattern)
    router.select_worker_for_model(model)
}

// ============================================================================
// Request Forwarding
// ============================================================================

/// Forward Messages API request to worker
async fn forward_to_worker(
    http_client: &reqwest::Client,
    worker: &Arc<dyn Worker>,
    headers: Option<&HeaderMap>,
    request: &CreateMessageRequest,
) -> Result<reqwest::Response, String> {
    let url = format!("{}/v1/messages", worker.url());

    debug!("Forwarding request to worker: {}", url);

    let mut req = http_client
        .post(&url)
        .json(request)
        .timeout(Duration::from_secs(120));

    // Propagate relevant headers
    if let Some(headers) = headers {
        for (key, value) in headers {
            if should_propagate_header(key.as_str()) {
                req = req.header(key, value);
            }
        }
    }

    // Send request
    req.send().await.map_err(|e| {
        warn!("Request to worker {} failed: {}", worker.url(), e);
        format!("Request failed: {}", e)
    })
}

/// Check if header should be propagated to worker
fn should_propagate_header(key: &str) -> bool {
    matches!(
        key.to_lowercase().as_str(),
        "authorization" | "x-api-key" | "anthropic-version" | "anthropic-beta"
    )
}

// ============================================================================
// Response Processing
// ============================================================================

/// Parse worker response into Messages API format
async fn parse_response(response: reqwest::Response) -> Response {
    let status = response.status();

    debug!("Received response with status: {}", status);

    if !status.is_success() {
        return handle_error_response(response).await;
    }

    // Parse JSON response
    match response.json::<Message>().await {
        Ok(msg_response) => {
            debug!("Successfully parsed response for message: {}", msg_response.id);

            // Return successful response
            (StatusCode::OK, Json(msg_response)).into_response()
        }
        Err(e) => {
            error!("Failed to parse response: {}", e);
            (StatusCode::BAD_GATEWAY, "Invalid response from backend").into_response()
        }
    }
}

/// Handle error response from worker
async fn handle_error_response(response: reqwest::Response) -> Response {
    let status = response.status();

    // Try to get response body for error details
    let body = response.text().await.unwrap_or_else(|e| {
        warn!("Failed to read error response body: {}", e);
        format!("Backend returned error: {}", status)
    });

    warn!("Backend error: {} - {}", status, body);

    // Pass through error to client with original status code
    (status, body).into_response()
}

// ============================================================================
// Metrics and Observability
// ============================================================================

/// Record incoming Messages API request
fn record_request(model: &str) {
    debug!(
        "Recording request for model: {} (metrics integration pending)",
        model
    );
    // TODO: Integrate with observability::metrics once available
    // Metrics::REQUESTS_TOTAL
    //     .with_label_values(&["anthropic", "messages", model])
    //     .inc();
}

/// Record Messages API response with latency and status
fn record_response(model: &str, status: u16, duration: Duration) {
    debug!(
        "Recording response for model: {}, status: {}, duration: {:?}",
        model, status, duration
    );
    // TODO: Integrate with observability::metrics once available
    // Metrics::RESPONSE_TIME
    //     .with_label_values(&["anthropic", "messages", model, &status.to_string()])
    //     .observe(duration.as_secs_f64());
}

/// Record token usage from response
#[allow(dead_code)]
fn record_usage(response: &Message) {
    let model = &response.model;
    let usage = &response.usage;

    debug!(
        "Recording usage for model: {}, input_tokens: {}, output_tokens: {}",
        model, usage.input_tokens, usage.output_tokens
    );
    // TODO: Integrate with observability::metrics once available
    // Metrics::TOKENS_PROCESSED
    //     .with_label_values(&["input", model])
    //     .inc_by(usage.input_tokens as f64);
    //
    // Metrics::TOKENS_PROCESSED
    //     .with_label_values(&["output", model])
    //     .inc_by(usage.output_tokens as f64);
}

/// Record error for Messages API request
fn record_error(model: &str, error_type: &str) {
    debug!(
        "Recording error for model: {}, error_type: {}",
        model, error_type
    );
    // TODO: Integrate with observability::metrics once available
    // Metrics::ERRORS_TOTAL
    //     .with_label_values(&["anthropic", "messages", model, error_type])
    //     .inc();
}

// ============================================================================
// Main Handler
// ============================================================================

/// Handle non-streaming Messages API request
pub async fn handle_non_streaming(
    router: &AnthropicRouter,
    headers: Option<&HeaderMap>,
    request: &CreateMessageRequest,
    model_id: Option<&str>,
) -> Response {
    let start_time = Instant::now();
    let model = model_id.unwrap_or(&request.model);

    info!(
        "Handling non-streaming Messages API request for model: {}",
        model
    );

    // Record incoming request
    record_request(model);

    // 1. Select worker via policy
    let worker = match select_worker(router, model_id, request).await {
        Ok(w) => {
            debug!("Selected worker: {}", w.url());
            w
        }
        Err(e) => {
            error!("Worker selection failed: {}", e);
            record_error(model, "worker_selection_error");
            return (StatusCode::SERVICE_UNAVAILABLE, e).into_response();
        }
    };

    // 2. Forward request to worker
    let worker_response = match forward_to_worker(router.http_client(), &worker, headers, request)
        .await
    {
        Ok(r) => r,
        Err(e) => {
            error!("Request forwarding failed: {}", e);
            record_error(model, "forward_error");
            return (StatusCode::BAD_GATEWAY, e).into_response();
        }
    };

    // 3. Parse and return response
    let response = parse_response(worker_response).await;

    // Record response metrics
    let duration = start_time.elapsed();
    record_response(model, response.status().as_u16(), duration);

    info!(
        "Completed non-streaming request for model: {} in {:?}",
        model, duration
    );

    response
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{app_context::AppContext, config::RouterConfig, protocols::messages::InputContent};

    fn create_test_message(role: Role, content: &str) -> InputMessage {
        InputMessage {
            role,
            content: InputContent::String(content.to_string()),
        }
    }

    // ========================================================================
    // Header Propagation Tests
    // ========================================================================

    #[test]
    fn test_should_propagate_header() {
        assert!(should_propagate_header("authorization"));
        assert!(should_propagate_header("Authorization"));
        assert!(should_propagate_header("x-api-key"));
        assert!(should_propagate_header("X-API-Key"));
        assert!(should_propagate_header("anthropic-version"));
        assert!(should_propagate_header("anthropic-beta"));

        assert!(!should_propagate_header("cookie"));
        assert!(!should_propagate_header("user-agent"));
        assert!(!should_propagate_header("host"));
        assert!(!should_propagate_header("content-length"));
    }

    // ========================================================================
    // Handler Tests
    // ========================================================================

    #[tokio::test]
    async fn test_worker_selection_error() {
        let config = RouterConfig::default();
        let context = Arc::new(AppContext::from_config(config, 120).await.unwrap());
        let router = AnthropicRouter::new(context);

        // Create valid request but no workers available
        let request = CreateMessageRequest {
            model: "test-model".to_string(),
            messages: vec![create_test_message(Role::User, "Hello")],
            max_tokens: 100,
            metadata: None,
            service_tier: None,
            stop_sequences: None,
            stream: None,
            system: None,
            temperature: None,
            thinking: None,
            tool_choice: None,
            tools: None,
            top_k: None,
            top_p: None,
            container: None,
            mcp_servers: None,
        };

        let response = handle_non_streaming(&router, None, &request, None).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }
}
