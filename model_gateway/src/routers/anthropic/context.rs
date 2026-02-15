//! Context types for Anthropic router
//!
//! - `RouterContext`: shared infrastructure (HTTP client, worker registry, MCP orchestrator)
//! - `RequestContext`: per-request input (request body, headers, model ID)

use std::{sync::Arc, time::Duration};

use smg_mcp::McpOrchestrator;
use openai_protocol::messages::CreateMessageRequest;
use axum::http::HeaderMap;

use crate::core::WorkerRegistry;

/// Shared context passed to all Anthropic handler functions.
pub(crate) struct RouterContext {
    pub mcp_orchestrator: Arc<McpOrchestrator>,
    pub http_client: reqwest::Client,
    pub worker_registry: Arc<WorkerRegistry>,
    pub request_timeout: Duration,
}

/// Per-request input that flows through handler functions.
pub(crate) struct RequestContext {
    pub request: CreateMessageRequest,
    pub headers: Option<HeaderMap>,
    pub model_id: String,
}
