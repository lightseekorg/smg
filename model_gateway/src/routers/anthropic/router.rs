//! Anthropic API router implementation
//!
//! This router handles Anthropic-specific APIs including:
//! - Messages API (/v1/messages) with SSE streaming
//! - Tool use and MCP integration
//! - Extended thinking and prompt caching

use std::{any::Any, sync::Arc};

use async_trait::async_trait;
use axum::{
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};

use super::messages;
use crate::{
    app_context::AppContext,
    protocols::{chat::ChatCompletionRequest, messages::CreateMessageRequest},
    routers::RouterTrait,
};

/// Router for Anthropic-specific APIs
///
/// Handles Anthropic's Messages API with support for:
/// - Streaming and non-streaming responses
/// - Tool use via MCP
/// - Extended thinking
/// - Prompt caching
/// - Citations
#[derive(Debug)]
pub struct AnthropicRouter {
    #[allow(dead_code)]
    context: Arc<AppContext>,
}

impl AnthropicRouter {
    /// Create a new Anthropic router
    pub fn new(context: Arc<AppContext>) -> Self {
        Self { context }
    }
}

#[async_trait]
impl RouterTrait for AnthropicRouter {
    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Route chat completion requests (not supported by Anthropic router)
    async fn route_chat(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &ChatCompletionRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (
            StatusCode::NOT_FOUND,
            "Chat completions not supported on Anthropic router. Use /v1/messages instead.",
        )
            .into_response()
    }

    /// Route Anthropic Messages API requests (/v1/messages)
    async fn route_messages(
        &self,
        headers: Option<&HeaderMap>,
        body: &CreateMessageRequest,
        _model_id: Option<&str>,
    ) -> Response {
        if body.stream.unwrap_or(false) {
            messages::handle_streaming(headers, body).await
        } else {
            messages::handle_non_streaming(headers, body).await
        }
    }

    fn router_type(&self) -> &'static str {
        "anthropic"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RouterConfig;

    #[tokio::test]
    async fn test_router_creation() {
        let config = RouterConfig::default();
        let context = Arc::new(AppContext::from_config(config, 120).await.unwrap());
        let router = AnthropicRouter::new(context);
        assert_eq!(router.router_type(), "anthropic");
    }

    #[tokio::test]
    async fn test_router_type() {
        let config = RouterConfig::default();
        let context = Arc::new(AppContext::from_config(config, 120).await.unwrap());
        let router = AnthropicRouter::new(context);
        assert_eq!(router.router_type(), "anthropic");
    }
}
