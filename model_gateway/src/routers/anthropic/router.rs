//! Anthropic API router implementation
//!
//! This router handles Anthropic-specific APIs including:
//! - Messages API (/v1/messages) with SSE streaming
//! - Tool use and MCP integration
//! - Extended thinking and prompt caching
//!
//! ## Pipeline Architecture
//!
//! The Messages API is processed through a 6-stage pipeline:
//! 1. Validation - Validate request fields and extract model ID
//! 2. Worker Selection - Select appropriate worker for the request
//! 3. Request Building - Build HTTP request for worker
//! 4. Dispatch Metadata - Generate request ID and timestamps
//! 5. Request Execution - Send request to worker
//! 6. Response Processing - Parse response and record metrics

use std::{any::Any, fmt, sync::Arc, time::Duration};

use async_trait::async_trait;
use axum::{body::Body, extract::Request, http::HeaderMap, response::Response};

use super::{context::SharedComponents, models, pipeline::MessagesPipeline};
use crate::{
    app_context::AppContext,
    protocols::{chat::ChatCompletionRequest, messages::CreateMessageRequest},
    routers::{error, RouterTrait},
};

/// Router for Anthropic-specific APIs
///
/// Handles Anthropic's Messages API with support for:
/// - Streaming and non-streaming responses
/// - Tool use via MCP
/// - Extended thinking
/// - Prompt caching
/// - Citations
pub struct AnthropicRouter {
    context: Arc<AppContext>,
    pipeline: MessagesPipeline,
}

impl fmt::Debug for AnthropicRouter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnthropicRouter")
            .field("context", &"<AppContext>")
            .field("pipeline", &self.pipeline)
            .finish()
    }
}

impl AnthropicRouter {
    pub fn new(context: Arc<AppContext>) -> Self {
        let request_timeout = Duration::from_secs(context.router_config.request_timeout_secs);
        let shared_components = Arc::new(SharedComponents::new(
            context.client.clone(),
            context.worker_registry.clone(),
            request_timeout,
        ));
        let pipeline = MessagesPipeline::new(shared_components);

        Self { context, pipeline }
    }

    pub fn context(&self) -> &Arc<AppContext> {
        &self.context
    }

    pub fn http_client(&self) -> &reqwest::Client {
        &self.context.client
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
        error::not_found(
            "unsupported_endpoint",
            "Chat completions not supported on Anthropic router. Use /v1/messages instead.",
        )
    }

    async fn route_messages(
        &self,
        headers: Option<&HeaderMap>,
        body: &CreateMessageRequest,
        model_id: &str,
    ) -> Response {
        // Clone body into Arc for pipeline (body is borrowed, pipeline needs ownership)
        let request = Arc::new(body.clone());
        let headers_owned = headers.cloned();

        // Execute through pipeline
        self.pipeline
            .execute(request, headers_owned, model_id)
            .await
    }

    /// Get available models from Anthropic API
    async fn get_models(&self, req: Request<Body>) -> Response {
        models::handle_list_models(self, req).await
    }

    fn router_type(&self) -> &'static str {
        "anthropic"
    }
}
