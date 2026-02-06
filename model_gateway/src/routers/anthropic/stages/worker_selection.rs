//! Worker selection stage for Anthropic router pipeline
//!
//! This stage selects an appropriate worker for the request:
//! - Filters workers by model support and provider type
//! - Selects least-loaded healthy worker
//! - Returns 503 if no healthy workers available
//!
//! SECURITY: In multi-provider setups, this stage filters by Anthropic provider
//! to prevent credential leakage to non-Anthropic workers.

use std::sync::Arc;

use async_trait::async_trait;
use tracing::{debug, warn};

use super::{PipelineStage, StageResult};
use crate::{
    core::WorkerRegistry,
    routers::{
        anthropic::{context::RequestContext, utils::find_best_worker_for_model},
        error,
    },
};

/// Worker selection stage
pub(crate) struct WorkerSelectionStage {
    worker_registry: Arc<WorkerRegistry>,
}

impl WorkerSelectionStage {
    /// Create a new worker selection stage
    pub fn new(worker_registry: Arc<WorkerRegistry>) -> Self {
        Self { worker_registry }
    }
}

#[async_trait]
impl PipelineStage for WorkerSelectionStage {
    async fn execute(&self, ctx: &mut RequestContext) -> StageResult {
        let model_id = &ctx.input.model_id;

        debug!(model = %model_id, "Selecting worker for request");

        // Find best worker for the model
        let worker = find_best_worker_for_model(&self.worker_registry, model_id);

        match worker {
            Some(w) => {
                debug!(
                    model = %model_id,
                    worker_url = %w.url(),
                    worker_load = %w.load(),
                    "Selected worker for request"
                );
                ctx.state.worker = Some(w);
                Ok(None)
            }
            None => {
                warn!(model = %model_id, "No healthy workers available for model");
                Err(error::service_unavailable(
                    "no_workers",
                    format!("No healthy workers available for model '{}'", model_id),
                ))
            }
        }
    }

    fn name(&self) -> &'static str {
        "worker_selection"
    }
}
