//! Anthropic API router implementation
//!
//! This router handles Anthropic-specific APIs including:
//! - Messages API (/v1/messages) with SSE streaming
//! - Tool use and MCP integration
//! - Extended thinking and prompt caching

use std::{any::Any, fmt, sync::Arc, time::Duration};

use async_trait::async_trait;
use axum::{body::Body, extract::Request, http::HeaderMap, response::Response};
use openai_protocol::messages::CreateMessageRequest;
use tracing::{info, warn};

use super::{
    context::{RequestContext, RouterContext},
    mcp, models, non_streaming, streaming, worker,
};
use crate::{app_context::AppContext, routers::RouterTrait};

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
    router_ctx: RouterContext,
}

impl fmt::Debug for AnthropicRouter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnthropicRouter")
            .field("context", &"<AppContext>")
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

        let router_ctx = RouterContext {
            mcp_orchestrator,
            http_client: context.client.clone(),
            worker_registry: context.worker_registry.clone(),
            request_timeout,
        };

        Ok(Self {
            context,
            router_ctx,
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

    async fn route_messages(
        &self,
        headers: Option<&HeaderMap>,
        body: &CreateMessageRequest,
        model_id: &str,
    ) -> Response {
        let mut request = body.clone();
        let headers_owned = headers.cloned();

        let mcp_servers = if request.has_mcp_toolset() {
            match mcp::ensure_connection(&mut request, &self.router_ctx.mcp_orchestrator).await {
                Ok(servers) => Some(servers),
                Err(response) => {
                    warn!(model = %model_id, "MCP connection setup failed");
                    return response;
                }
            }
        } else {
            None
        };

        let is_streaming = request.stream.unwrap_or(false);
        info!(
            model = %model_id,
            streaming = %is_streaming,
            mcp = %mcp_servers.is_some(),
            "Processing Messages API request"
        );

        let selected_worker =
            match worker::select_worker(&self.router_ctx.worker_registry, model_id) {
                Ok(w) => w,
                Err(resp) => return resp,
            };

        let req_ctx = RequestContext {
            request,
            headers: headers_owned,
            model_id: model_id.to_string(),
            mcp_servers,
            worker: selected_worker,
        };

        if is_streaming {
            streaming::execute(&self.router_ctx, req_ctx).await
        } else {
            non_streaming::execute(&self.router_ctx, req_ctx).await
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
