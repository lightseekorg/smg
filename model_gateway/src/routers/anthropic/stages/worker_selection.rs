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
use axum::{http::StatusCode, response::IntoResponse};
use tracing::{debug, warn};

use super::{PipelineStage, StageResult};
use crate::{
    core::{model_card::ProviderType, WorkerRegistry},
    routers::anthropic::context::RequestContext,
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

    /// Find the best worker for a given model
    ///
    /// This method follows the same pattern as OpenAI router to correctly handle
    /// wildcard workers (workers with empty model lists that accept any model).
    ///
    /// SECURITY: In multi-provider setups, filters by Anthropic provider to prevent
    /// credential leakage (Anthropic X-API-Key, Authorization) to non-Anthropic workers.
    fn find_best_worker_for_model(&self, model_id: &str) -> Option<Arc<dyn crate::core::Worker>> {
        let all_workers = self.worker_registry.get_workers_filtered(
            None, // Don't filter by model in get_workers_filtered
            None, // worker_type
            None, // connection_mode
            None, // runtime_type
            true, // healthy_only
        );

        // SECURITY: In multi-provider setups, filter by Anthropic provider to prevent
        // credential leakage. Use early-exit pattern to avoid allocation.
        let mut first_provider = None;
        let has_multiple_providers = all_workers.iter().any(|w| {
            if let Some(p) = w.default_provider() {
                match first_provider {
                    None => {
                        first_provider = Some(p);
                        false
                    }
                    Some(first) => first != p,
                }
            } else {
                false
            }
        });

        let eligible: Vec<_> = if has_multiple_providers {
            // Multi-provider setup: only use explicitly Anthropic workers
            all_workers
                .into_iter()
                .filter(|w| matches!(w.default_provider(), Some(ProviderType::Anthropic)))
                .collect()
        } else {
            // Single-provider or no-provider setup: use all workers
            all_workers
        };

        eligible
            .into_iter()
            .filter(|w| w.supports_model(model_id))
            .min_by_key(|w| w.load())
    }
}

#[async_trait]
impl PipelineStage for WorkerSelectionStage {
    async fn execute(&self, ctx: &mut RequestContext) -> StageResult {
        let model_id = &ctx.input.model_id;

        debug!(model = %model_id, "Selecting worker for request");

        // Find best worker for the model
        let worker = self.find_best_worker_for_model(model_id);

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
                Err((
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!("No healthy workers available for model '{}'", model_id),
                )
                    .into_response())
            }
        }
    }

    fn name(&self) -> &'static str {
        "worker_selection"
    }
}
