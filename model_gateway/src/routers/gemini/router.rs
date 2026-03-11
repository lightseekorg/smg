//! GeminiRouter — entry point for the Gemini Interactions API.

use std::{
    any::Any,
    sync::{atomic::AtomicBool, Arc},
    time::Duration,
};

use async_trait::async_trait;
use axum::{http::HeaderMap, response::Response};
use openai_protocol::interactions::InteractionsRequest;

use super::{
    context::{RequestContext, SharedComponents},
    driver,
};
use crate::{app_context::AppContext, routers::RouterTrait};

pub struct GeminiRouter {
    shared_components: Arc<SharedComponents>,
    #[expect(dead_code)]
    healthy: AtomicBool,
}

impl std::fmt::Debug for GeminiRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GeminiRouter").finish()
    }
}

impl GeminiRouter {
    /// Create a new `GeminiRouter` from the application context.
    pub fn new(ctx: Arc<AppContext>) -> Result<Self, String> {
        let mcp_orchestrator = ctx
            .mcp_orchestrator
            .get()
            .ok_or_else(|| "Gemini router requires MCP orchestrator".to_string())?
            .clone();

        let request_timeout = Duration::from_secs(ctx.router_config.request_timeout_secs);

        let shared_components = Arc::new(SharedComponents {
            client: ctx.client.clone(),
            worker_registry: ctx.worker_registry.clone(),
            mcp_orchestrator,
            request_timeout,
        });
        Ok(Self {
            shared_components,
            healthy: AtomicBool::new(true),
        })
    }
}

// ============================================================================
// RouterTrait implementation
// ============================================================================

#[async_trait]
impl RouterTrait for GeminiRouter {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn router_type(&self) -> &'static str {
        "gemini"
    }

    async fn route_interactions(
        &self,
        headers: Option<&HeaderMap>,
        body: &InteractionsRequest,
        model_id: Option<&str>,
    ) -> Response {
        let mut ctx = RequestContext::new(
            Arc::new(body.clone()),
            headers.cloned(),
            model_id.map(String::from),
            self.shared_components.clone(),
        );

        driver::execute(&mut ctx).await
    }
}
