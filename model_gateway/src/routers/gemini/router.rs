//! GeminiRouter — entry point for the Gemini Interactions API.

use std::{
    any::Any,
    io,
    sync::{atomic::AtomicBool, Arc},
};

use async_trait::async_trait;
use axum::{
    body::Body,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use openai_protocol::{chat::ChatCompletionRequest, interactions::InteractionsRequest};
use smg_mcp::McpOrchestrator;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

use super::{
    context::{RequestContext, SharedComponents},
    driver,
};
use crate::{core::WorkerRegistry, routers::RouterTrait};

pub struct GeminiRouter {
    shared_components: Arc<SharedComponents>,
    #[allow(dead_code)]
    healthy: AtomicBool,
}

impl std::fmt::Debug for GeminiRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GeminiRouter").finish()
    }
}

impl GeminiRouter {
    /// Create a new `GeminiRouter`.
    pub async fn new(
        worker_registry: Arc<WorkerRegistry>,
        mcp_orchestrator: Arc<McpOrchestrator>,
        client: reqwest::Client,
    ) -> Result<Self, String> {
        let shared_components = Arc::new(SharedComponents {
            client,
            worker_registry,
            mcp_orchestrator,
        });
        Ok(Self {
            shared_components,
            healthy: AtomicBool::new(true),
        })
    }

    /// Main handler for `POST /v1/interactions`.
    ///
    /// Builds a `RequestContext`, then either:
    /// - **Non-streaming**: awaits the driver inline and returns the `Response`.
    /// - **Streaming**: creates an SSE channel, spawns the driver in a background
    ///   task (events flow through `sse_tx`), and returns the SSE `Response` immediately.
    pub async fn route_interactions(
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

        if body.stream {
            // Streaming: create SSE channel, spawn driver, return SSE response.
            let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, io::Error>>();
            ctx.streaming.sse_tx = Some(tx);

            tokio::spawn(async move {
                let response = driver::execute(&mut ctx).await;
                if !response.status().is_success() {
                    tracing::error!(
                        "Streaming request processing failed with status: {}",
                        response.status()
                    );
                }
            });

            let body_stream = UnboundedReceiverStream::new(rx);
            let mut response = Response::new(Body::from_stream(body_stream));
            *response.status_mut() = StatusCode::OK;
            response
                .headers_mut()
                .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
            response
        } else {
            // Non-streaming: run driver synchronously, return the Response.
            driver::execute(&mut ctx).await
        }
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
