//! Non-streaming Messages API handler
//!
//! This module handles all aspects of non-streaming Messages API requests:
//! - Request validation
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
    protocols::messages::{
        CreateMessageRequest, CustomTool, InputMessage, InputSchema, Message, Role, Tool,
    },
    routers::anthropic::AnthropicRouter,
};

// ============================================================================
// Request Validation
// ============================================================================

/// Validate Messages API request
fn validate_request(request: &CreateMessageRequest) -> Result<(), String> {
    // Check required fields
    if request.messages.is_empty() {
        return Err("messages array cannot be empty".to_string());
    }

    if request.max_tokens == 0 {
        return Err("max_tokens must be greater than 0".to_string());
    }

    if request.model.is_empty() {
        return Err("model cannot be empty".to_string());
    }

    // Validate message sequence
    validate_message_sequence(&request.messages)?;

    // Validate tools if present
    if let Some(tools) = &request.tools {
        validate_tools(tools)?;
    }

    Ok(())
}

/// Validate message role sequence
///
/// Rules:
/// 1. Must start with user message
/// 2. Messages must alternate between user and assistant
/// 3. Must end with user message
fn validate_message_sequence(messages: &[InputMessage]) -> Result<(), String> {
    if messages.is_empty() {
        return Err("messages array cannot be empty".to_string());
    }

    // Must start with user message
    if messages[0].role != Role::User {
        return Err("First message must have role 'user'".to_string());
    }

    // Must end with user message
    if messages.last().unwrap().role != Role::User {
        return Err("Last message must have role 'user'".to_string());
    }

    // Validate alternating roles
    for window in messages.windows(2) {
        if window[0].role == window[1].role {
            return Err(format!(
                "Messages must alternate between 'user' and 'assistant' roles. Found consecutive {:?} messages.",
                window[0].role
            ));
        }
    }

    Ok(())
}

/// Validate tool definitions
fn validate_tools(tools: &[Tool]) -> Result<(), String> {
    if tools.is_empty() {
        return Err("tools array cannot be empty if provided".to_string());
    }

    for (idx, tool) in tools.iter().enumerate() {
        // Extract name based on tool variant
        let name = match tool {
            Tool::Custom(t) => &t.name,
            Tool::Bash(t) => &t.name,
            Tool::TextEditor(t) => &t.name,
            Tool::WebSearch(t) => &t.name,
        };

        if name.is_empty() {
            return Err(format!("Tool at index {} has empty name", idx));
        }

        // Tool name should be valid (alphanumeric, underscore, hyphen)
        if !name
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
        {
            return Err(format!(
                "Tool '{}' has invalid name. Only alphanumeric, underscore, and hyphen allowed.",
                name
            ));
        }

        // Validate input_schema for Custom tools
        if let Tool::Custom(custom_tool) = tool {
            // InputSchema must have a valid schema_type
            if custom_tool.input_schema.schema_type.is_empty() {
                return Err(format!("Tool '{}' has invalid input_schema", name));
            }
        }
    }

    Ok(())
}

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

    // 1. Validate request
    if let Err(e) = validate_request(request) {
        error!("Request validation failed: {}", e);
        record_error(model, "validation_error");
        return (StatusCode::BAD_REQUEST, e).into_response();
    }

    // 2. Select worker via policy
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

    // 3. Forward request to worker
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

    // 4. Parse and return response
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
    use std::collections::HashMap;

    // ========================================================================
    // Validation Tests
    // ========================================================================

    fn create_test_message(role: Role, content: &str) -> InputMessage {
        InputMessage {
            role,
            content: InputContent::String(content.to_string()),
        }
    }

    #[test]
    fn test_valid_request() {
        let request = CreateMessageRequest {
            model: "claude-sonnet-4-5-20250929".to_string(),
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

        assert!(validate_request(&request).is_ok());
    }

    #[test]
    fn test_empty_messages() {
        let request = CreateMessageRequest {
            model: "test".to_string(),
            messages: vec![],
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

        assert!(validate_request(&request).is_err());
    }

    #[test]
    fn test_zero_max_tokens() {
        let request = CreateMessageRequest {
            model: "test".to_string(),
            messages: vec![create_test_message(Role::User, "Hello")],
            max_tokens: 0,
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

        assert!(validate_request(&request).is_err());
    }

    #[test]
    fn test_must_start_with_user() {
        let request = CreateMessageRequest {
            model: "test".to_string(),
            messages: vec![create_test_message(Role::Assistant, "Hello")],
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

        let result = validate_request(&request);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("First message must have role 'user'"));
    }

    #[test]
    fn test_must_end_with_user() {
        let request = CreateMessageRequest {
            model: "test".to_string(),
            messages: vec![
                create_test_message(Role::User, "Hello"),
                create_test_message(Role::Assistant, "Hi"),
            ],
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

        let result = validate_request(&request);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("Last message must have role 'user'"));
    }

    #[test]
    fn test_alternating_roles() {
        let request = CreateMessageRequest {
            model: "test".to_string(),
            messages: vec![
                create_test_message(Role::User, "Hello"),
                create_test_message(Role::User, "Again"),
            ],
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

        let result = validate_request(&request);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must alternate"));
    }

    #[test]
    fn test_valid_multi_turn() {
        let request = CreateMessageRequest {
            model: "test".to_string(),
            messages: vec![
                create_test_message(Role::User, "Hello"),
                create_test_message(Role::Assistant, "Hi"),
                create_test_message(Role::User, "How are you?"),
            ],
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

        assert!(validate_request(&request).is_ok());
    }

    #[test]
    fn test_valid_tools() {
        let tools = vec![Tool::Custom(CustomTool {
            name: "get_weather".to_string(),
            tool_type: None,
            description: Some("Get weather".to_string()),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: Some(HashMap::new()),
                required: None,
                additional: HashMap::new(),
            },
            cache_control: None,
        })];

        assert!(validate_tools(&tools).is_ok());
    }

    #[test]
    fn test_empty_tools_array() {
        let tools = vec![];
        assert!(validate_tools(&tools).is_err());
    }

    #[test]
    fn test_tool_with_empty_name() {
        let tools = vec![Tool::Custom(CustomTool {
            name: "".to_string(),
            tool_type: None,
            description: None,
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: None,
                required: None,
                additional: HashMap::new(),
            },
            cache_control: None,
        })];

        assert!(validate_tools(&tools).is_err());
    }

    #[test]
    fn test_tool_with_invalid_name() {
        let tools = vec![Tool::Custom(CustomTool {
            name: "get@weather".to_string(), // @ not allowed
            tool_type: None,
            description: None,
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: None,
                required: None,
                additional: HashMap::new(),
            },
            cache_control: None,
        })];

        assert!(validate_tools(&tools).is_err());
    }

    #[test]
    fn test_tool_with_empty_schema_type() {
        let tools = vec![Tool::Custom(CustomTool {
            name: "test_tool".to_string(),
            tool_type: None,
            description: None,
            input_schema: InputSchema {
                schema_type: "".to_string(), // Empty schema type
                properties: None,
                required: None,
                additional: HashMap::new(),
            },
            cache_control: None,
        })];

        assert!(validate_tools(&tools).is_err());
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
    async fn test_validation_error() {
        let config = RouterConfig::default();
        let context = Arc::new(AppContext::from_config(config, 120).await.unwrap());
        let router = AnthropicRouter::new(context);

        // Create invalid request (empty messages)
        let request = CreateMessageRequest {
            model: "test-model".to_string(),
            messages: vec![],
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
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

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
