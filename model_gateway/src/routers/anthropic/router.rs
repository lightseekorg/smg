//! Anthropic API router implementation
//!
//! This router handles Anthropic-specific APIs including:
//! - Messages API (/v1/messages) with SSE streaming
//! - Tool use and MCP integration
//! - Extended thinking and prompt caching
//!
//! ## Pipeline Architecture
//!
//! The Messages API is processed through a 4-stage pipeline:
//! 1. Worker Selection - Select appropriate worker for the request
//! 2. Request Building - Build HTTP request for worker
//! 3. Request Execution - Send request to worker
//! 4. Response Processing - Parse response and record metrics

use std::{any::Any, fmt, sync::Arc, time::Duration};

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use tracing::{info, warn};

use super::{
    context::{MessagesContext, SharedComponents},
    messages::tools::ensure_mcp_connection,
    models,
    pipeline::MessagesPipeline,
};
use openai_protocol::{chat::ChatCompletionRequest, messages::CreateMessageRequest};

use crate::{
    app_context::AppContext,
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
    messages_ctx: MessagesContext,
}

impl fmt::Debug for AnthropicRouter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnthropicRouter")
            .field("context", &"<AppContext>")
            .field("pipeline", &self.messages_ctx.pipeline)
            .finish()
    }
}

impl AnthropicRouter {
    pub fn new(context: Arc<AppContext>) -> Result<Self, String> {
        let request_timeout = Duration::from_secs(context.router_config.request_timeout_secs);
        let mcp_orchestrator = context
            .mcp_orchestrator
            .get()
            .ok_or_else(|| "Anthropic router requires MCP orchestrator".to_string())?
            .clone();

        let shared_components = Arc::new(SharedComponents {
            http_client: context.client.clone(),
            request_timeout,
        });

        let pipeline = Arc::new(MessagesPipeline::new(
            shared_components,
            context.worker_registry.clone(),
        ));

        let messages_ctx = MessagesContext::new(pipeline, mcp_orchestrator);

        Ok(Self {
            context,
            messages_ctx,
        })
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
        let mut request = body.clone();
        let headers_owned = headers.cloned();

        let mcp_servers = if request.has_mcp_toolset() {
            match ensure_mcp_connection(&mut request, &self.messages_ctx.mcp_orchestrator).await {
                Ok(servers) => Some(servers),
                Err(response) => {
                    warn!(model = %model_id, "MCP connection setup failed");
                    return response;
                }
            }
        } else {
            None
        };

        let streaming = request.stream.unwrap_or(false);
        info!(
            model = %model_id,
            streaming = %streaming,
            mcp = %mcp_servers.is_some(),
            "Processing Messages API request"
        );

        if streaming {
            return self
                .messages_ctx
                .pipeline
                .execute_streaming(request, headers_owned, model_id)
                .await;
        }

        if let Some(mcp_servers) = mcp_servers {
            return super::messages::non_streaming::execute_tool_loop(
                &self.messages_ctx,
                request,
                headers_owned,
                model_id,
                mcp_servers,
            )
            .await;
        }

        match self
            .messages_ctx
            .pipeline
            .execute_for_messages(request, headers_owned, model_id)
            .await
        {
            Ok(message) => (StatusCode::OK, Json(message)).into_response(),
            Err(response) => response,
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
