//! Anthropic API router implementation
//!
//! This router handles Anthropic-specific APIs including:
//! - Messages API (/v1/messages) with SSE streaming
//! - Tool use and MCP integration
//! - Extended thinking and prompt caching

use std::{any::Any, sync::Arc, time::Duration};

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use tracing::debug;

use super::{messages, models};
use crate::{
    app_context::AppContext,
    core::Worker,
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
    context: Arc<AppContext>,
    http_client: reqwest::Client,
}

impl AnthropicRouter {
    /// Create a new Anthropic router
    pub fn new(context: Arc<AppContext>) -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            context,
            http_client,
        }
    }

    /// Get reference to app context
    pub fn context(&self) -> &Arc<AppContext> {
        &self.context
    }

    /// Get reference to HTTP client
    pub fn http_client(&self) -> &reqwest::Client {
        &self.http_client
    }

    /// Find workers that can handle the given model and select the least loaded one
    ///
    /// This method follows the same pattern as OpenAI router to correctly handle
    /// wildcard workers (workers with empty model lists that accept any model).
    pub fn find_best_worker_for_model(&self, model_id: &str) -> Option<Arc<dyn Worker>> {
        self.context
            .worker_registry
            .get_workers_filtered(
                None, // Don't filter by model in get_workers_filtered
                None, // provider
                None, // connection_mode
                None, // runtime_type
                true, // healthy_only
            )
            .into_iter()
            .filter(|w| w.supports_model(model_id))
            .min_by_key(|w| w.load())
    }

    /// Check if any worker supports the model (regardless of health/load)
    pub fn any_worker_supports_model(&self, model_id: &str) -> bool {
        self.context
            .worker_registry
            .get_workers_filtered(None, None, None, None, false)
            .into_iter()
            .any(|w| w.supports_model(model_id))
    }

    /// Select a worker for the given model
    ///
    /// Returns an error string if no suitable worker is found.
    pub fn select_worker_for_model(&self, model_id: &str) -> Result<Arc<dyn Worker>, String> {
        debug!("Selecting worker for model: {}", model_id);

        self.find_best_worker_for_model(model_id)
            .ok_or_else(|| format!("No healthy workers available for model '{}'", model_id))
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
        model_id: Option<&str>,
    ) -> Response {
        if body.stream.unwrap_or(false) {
            messages::handle_streaming(self, headers, body, model_id).await
        } else {
            messages::handle_non_streaming(self, headers, body, model_id).await
        }
    }

    /// Get available models from Anthropic API
    async fn get_models(&self, req: Request<Body>) -> Response {
        models::handle_list_models(self, req).await
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
