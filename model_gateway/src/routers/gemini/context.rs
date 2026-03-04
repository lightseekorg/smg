//! Context types for the Gemini Interactions router.
//!
//! Two-level context design:
//! - `SharedComponents`: created once per router, `Arc`-cloned for each request.
//! - `RequestContext`: created fresh per request, owned and mutated by steps.

use std::{io, sync::Arc};

use axum::http::HeaderMap;
use bytes::Bytes;
use openai_protocol::interactions::InteractionsRequest;
use serde_json::Value;
use smg_mcp::McpOrchestrator;
use tokio::sync::mpsc;

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
#[expect(dead_code)]
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
#[expect(dead_code)]
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
#[expect(dead_code)]
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
#[expect(dead_code)]
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
#[expect(dead_code)]
pub(crate) struct StreamingState {
    /// SSE sender channel (set before spawning the streaming task).
    pub sse_tx: Option<mpsc::UnboundedSender<Result<Bytes, io::Error>>>,

    /// Whether this is the first tool-loop iteration.
    ///
    /// Controls dedup of lifecycle events (`interaction.start`, `interaction.in_progress`)
    /// which must only be emitted once across tool-loop iterations.
    /// Set to `false` after the first iteration completes.
    pub is_first_iteration: bool,

    /// Monotonically increasing event sequence number.
    ///
    /// Incremented for every SSE event sent to the client. Ensures events
    /// are sequentially numbered across tool-loop iterations.
    pub sequence_number: u64,

    /// Next output index for sequential output item numbering.
    ///
    /// Tracks the output_index across tool-loop iterations so that each
    /// output item (model response, tool call, tool result) gets a unique
    /// sequential index.
    pub next_output_index: u64,
}

impl Default for StreamingState {
    fn default() -> Self {
        Self {
            sse_tx: None,
            is_first_iteration: true,
            sequence_number: 0,
            next_output_index: 0,
        }
    }
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
