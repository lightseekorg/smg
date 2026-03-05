//! GeminiRouter — entry point for the Gemini Interactions API.

use std::{
    any::Any,
    sync::{atomic::AtomicBool, Arc},
};

use async_trait::async_trait;
use axum::{
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use openai_protocol::{chat::ChatCompletionRequest, interactions::InteractionsRequest};
use smg_mcp::McpOrchestrator;

use super::{
    context::{RequestContext, SharedComponents},
    driver,
};
use crate::{core::WorkerRegistry, routers::RouterTrait};

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
    /// Create a new `GeminiRouter`.
    pub fn new(
        worker_registry: Arc<WorkerRegistry>,
        mcp_orchestrator: Arc<McpOrchestrator>,
        client: reqwest::Client,
    ) -> Self {
        let shared_components = Arc::new(SharedComponents {
            client,
            worker_registry,
            mcp_orchestrator,
        });
        Self {
            shared_components,
            healthy: AtomicBool::new(true),
        }
    }

    /// Main handler for `POST /v1/interactions`.
    ///
    /// Builds a `RequestContext` and runs the driver state machine.
    ///
    /// For **non-streaming** requests the driver returns the HTTP response directly.
    ///
    /// For **streaming** requests the terminal streaming step (`StreamRequest` or
    /// `StreamRequestWithTool`) creates an SSE channel internally, spawns the
    /// streaming work in a background task, and returns the SSE `Response` — the
    /// same pattern as `process_streaming_response` in the gRPC router.
    pub async fn route_interactions(
        &self,
        headers: Option<HeaderMap>,
        body: InteractionsRequest,
        model_id: Option<String>,
    ) -> Response {
        let mut ctx = RequestContext::new(
            Arc::new(body),
            headers,
            model_id,
            self.shared_components.clone(),
        );

        driver::execute(&mut ctx).await
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

    async fn route_chat(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &ChatCompletionRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED, "Not implemented").into_response()
    }
}
