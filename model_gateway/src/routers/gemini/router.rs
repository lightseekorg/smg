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
    /// Builds a `RequestContext`, then either:
    /// - **Non-streaming**: awaits the driver inline and returns the `Response`.
    /// - **Streaming**: creates an SSE channel, spawns the driver in a background
    ///   task (events flow through `sse_tx`), and returns the SSE `Response` immediately.
    pub async fn route_interactions(
        &self,
        headers: Option<HeaderMap>,
        body: InteractionsRequest,
        model_id: Option<String>,
    ) -> Response {
        let stream = body.stream;
        let mut ctx = RequestContext::new(
            Arc::new(body),
            headers,
            model_id,
            self.shared_components.clone(),
        );

        if stream {
            // Streaming: create SSE channel, spawn driver, return SSE response.
            let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, io::Error>>();
            let error_tx = tx.clone();
            ctx.streaming.sse_tx = Some(tx);

            // The driver's Response is intentionally discarded: for streaming,
            // the real response flows through `sse_tx` events. The driver returns
            // a dummy Response only to satisfy the state machine's return type.
            // We log errors here so failures aren't completely silent.

            #[expect(
                clippy::disallowed_methods,
                reason = "fire-and-forget SSE streaming driver; gateway shutdown need not wait for individual streams"
            )]
            tokio::spawn(async move {
                let response = driver::execute(&mut ctx).await;
                if !response.status().is_success() {
                    let status = response.status().as_u16();
                    tracing::error!(
                        "Streaming request processing failed with status: {}",
                        status
                    );
                    // Send an SSE error event so the client sees the failure
                    // before the stream closes.
                    let error_event = format!(
                        "event: error\ndata: {{\"status\":{status},\"message\":\"internal pipeline error\"}}\n\n"
                    );
                    let _ = error_tx.send(Ok(Bytes::from(error_event)));
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
