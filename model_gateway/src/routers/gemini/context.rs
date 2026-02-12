//! Context types for the Gemini Interactions router.
//!
//! Two-level context design:
//! - `SharedComponents`: created once per router, `Arc`-cloned for each request.
//! - `RequestContext`: created fresh per request, owned and mutated by steps.

use std::sync::Arc;

use axum::http::HeaderMap;
use serde_json::Value;
use smg_mcp::McpOrchestrator;

use openai_protocol::interactions::InteractionsRequest;

use super::state::RequestState;
use crate::core::{Worker, WorkerRegistry};

// ============================================================================
// SharedComponents (per-router)
// ============================================================================

/// Immutable state shared across all requests.
///
/// Created once during `GeminiRouter::new()` and cheaply `Arc`-cloned
/// into every `RequestContext`.
///
/// TODO: Create InteractionsStorage and add in context
#[allow(dead_code)]
pub(crate) struct SharedComponents {
    /// HTTP client for upstream requests.
    pub client: reqwest::Client,

    /// Worker registry for model → worker resolution.
    pub worker_registry: Arc<WorkerRegistry>,

    /// MCP orchestrator for creating tool sessions.
    pub mcp_orchestrator: Arc<McpOrchestrator>,
}

// ============================================================================
// RequestContext (per-request)
// ============================================================================

/// Per-request mutable state passed through the state machine.
///
/// Steps read and write fields on this struct. The `state` field
/// determines which step the driver executes next.
#[allow(dead_code)]
pub(crate) struct RequestContext {
    /// Immutable request data from the client.
    pub input: RequestInput,

    /// Reference to the per-router shared components.
    pub components: Arc<SharedComponents>,

    /// Current position in the state machine.
    pub state: RequestState,

    /// Mutable processing state populated incrementally by steps.
    pub processing: ProcessingState,

    /// Streaming-specific state (only used when `input.original_request.stream` is true).
    pub streaming: StreamingState,
}

/// Immutable request data captured at the start of processing.
#[allow(dead_code)]
pub(crate) struct RequestInput {
    /// Original client request (`Arc` for cheap cloning into spawned tasks).
    pub original_request: Arc<InteractionsRequest>,

    /// HTTP headers forwarded from the client.
    pub headers: Option<HeaderMap>,

    /// Optional model ID override (e.g. from URL path or query parameter).
    /// When set, takes precedence over `original_request.model`.
    pub model_id: Option<String>,
}

/// Mutable processing state populated incrementally by steps.
#[derive(Default)]
#[allow(dead_code)]
pub(crate) struct ProcessingState {
    /// Selected upstream worker (set by `WorkerSelection`).
    pub worker: Option<Arc<dyn Worker>>,

    /// Upstream URL: `{worker.url()}/v1/interactions` (set by `WorkerSelection`).
    pub upstream_url: Option<String>,

    /// JSON payload to POST upstream (set by `RequestBuilding`, mutated on tool-loop resume).
    pub payload: Option<Value>,

    /// Latest upstream response JSON (set by execution steps).
    pub upstream_response: Option<Value>,
}

/// State specific to streaming responses.
#[derive(Default)]
pub(crate) struct StreamingState {
    /// SSE sender channel (set before spawning the streaming task).
    pub sse_tx: Option<tokio::sync::mpsc::UnboundedSender<Result<bytes::Bytes, std::io::Error>>>,
}

impl RequestContext {
    /// Create a new `RequestContext` in the `SelectWorker` state.
    pub fn new(
        original_request: Arc<InteractionsRequest>,
        headers: Option<HeaderMap>,
        model_id: Option<String>,
        components: Arc<SharedComponents>,
    ) -> Self {
        Self {
            input: RequestInput {
                original_request,
                headers,
                model_id,
            },
            components,
            state: RequestState::SelectWorker,
            processing: ProcessingState::default(),
            streaming: StreamingState::default(),
        }
    }
}
