//! Dispatch metadata stage for Anthropic router pipeline
//!
//! This stage generates metadata for request tracing and metrics:
//! - Records timestamps
//! - Creates OTEL tracing span

use async_trait::async_trait;
use tracing::debug;

use super::{PipelineStage, StageResult};
use crate::routers::anthropic::context::{DispatchMetadata, RequestContext};

/// Dispatch metadata stage
pub(crate) struct DispatchMetadataStage;

impl DispatchMetadataStage {
    /// Create a new dispatch metadata stage
    pub fn new() -> Self {
        Self
    }
}

impl Default for DispatchMetadataStage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PipelineStage for DispatchMetadataStage {
    async fn execute(&self, ctx: &mut RequestContext) -> StageResult {
        let streaming = ctx.is_streaming();
        let metadata = DispatchMetadata::new(streaming);

        debug!(
            model = %ctx.input.model_id,
            streaming = %metadata.streaming,
            "Generated dispatch metadata"
        );

        ctx.state.dispatch = Some(metadata);
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "dispatch_metadata"
    }
}
