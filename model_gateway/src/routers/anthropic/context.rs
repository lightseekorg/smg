//! Router context for Anthropic router
//!
//! Holds shared infrastructure (HTTP client, worker registry, MCP orchestrator)
//! needed by the handler functions.

use std::{sync::Arc, time::Duration};

use crate::{core::WorkerRegistry, mcp::McpOrchestrator};

/// Shared context for all Anthropic router request handling.
#[derive(Clone)]
pub(crate) struct RouterContext {
    pub mcp_orchestrator: Arc<McpOrchestrator>,
    pub http_client: reqwest::Client,
    pub worker_registry: Arc<WorkerRegistry>,
    pub request_timeout: Duration,
}
