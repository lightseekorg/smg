//! Context types for Anthropic router
//!
//! - `RouterContext`: shared infrastructure (HTTP client, worker registry, MCP orchestrator)
//! - `RequestContext`: per-request input (request body, headers, model ID)

use std::{sync::Arc, time::Duration};

use axum::http::HeaderMap;

use crate::{
    core::WorkerRegistry, mcp::McpOrchestrator, protocols::messages::CreateMessageRequest,
};

/// Shared context for all Anthropic router request handling.
#[derive(Clone)]
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
