//! Request building stage for Anthropic router pipeline
//!
//! This stage builds the HTTP request to send to the worker:
//! - Constructs target URL
//! - Propagates relevant headers

use async_trait::async_trait;
use axum::http::HeaderMap;
use tracing::{debug, error};

use super::{PipelineStage, StageResult};
use crate::routers::{
    anthropic::{
        context::{HttpRequestState, RequestContext},
        utils::should_propagate_header,
    },
    error,
};

/// Request building stage
pub(crate) struct RequestBuildingStage;

impl RequestBuildingStage {
    /// Create a new request building stage
    pub fn new() -> Self {
        Self
    }
}

impl Default for RequestBuildingStage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PipelineStage for RequestBuildingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> StageResult {
        // Ensure we have a worker selected
        let worker = ctx.state.worker.as_ref().ok_or_else(|| {
            error!("Request building stage called without worker selection");
            error::internal_error("no_worker", "Internal error: no worker selected")
        })?;

        // Build target URL
        let url = format!("{}/v1/messages", worker.url());
        debug!(url = %url, "Building request for worker");

        // Build headers to propagate
        let mut headers = HeaderMap::new();
        if let Some(ref input_headers) = ctx.input.headers {
            for (key, value) in input_headers {
                if should_propagate_header(key.as_str()) {
                    headers.insert(key.clone(), value.clone());
                }
            }
        }

        debug!(
            url = %url,
            header_count = %headers.len(),
            "Request built successfully"
        );

        ctx.state.http_request = Some(HttpRequestState { url, headers });
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "request_building"
    }
}
