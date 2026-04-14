//! Step to remove workers from worker registry.

use std::collections::HashSet;

use async_trait::async_trait;
use tracing::{debug, warn};
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::{
    observability::metrics::Metrics,
    worker::worker::{ConnectionModeExt, WorkerTypeExt},
    workflow::data::WorkerRemovalWorkflowData,
};

/// Step to remove workers from the worker registry.
///
/// Removes each worker by URL from the central worker registry.
pub struct RemoveFromWorkerRegistryStep;

#[async_trait]
impl StepExecutor<WorkerRemovalWorkflowData> for RemoveFromWorkerRegistryStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerRemovalWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;
        let worker_urls = &context.data.worker_urls;

        debug!(
            "Removing {} worker(s) from worker registry",
            worker_urls.len()
        );

        // Snapshot the unique `(worker_type, connection_mode, model_id)`
        // groups for the workers we are about to remove. We capture this
        // before the removal so the pool-size metric update below can
        // recompute the per-group count after the workers have been
        // pulled from the registry. The cache eviction the old
        // `LoadMonitor` path used to do here is now handled by
        // `WorkerMonitor` reacting to `WorkerEvent::Removed`.
        let mut unique_configs: HashSet<(
            openai_protocol::worker::WorkerType,
            openai_protocol::worker::ConnectionMode,
            String,
        )> = HashSet::new();
        for url in worker_urls {
            if let Some(w) = app_context.worker_registry.get_by_url(url) {
                let meta = w.metadata();
                unique_configs.insert((
                    meta.spec.worker_type,
                    meta.spec.connection_mode,
                    w.model_id().to_string(),
                ));
            }
        }

        let mut removed_count = 0;
        for worker_url in worker_urls {
            if app_context
                .worker_registry
                .remove_by_url(worker_url)
                .is_some()
            {
                removed_count += 1;
            }
        }

        // Log if some workers were already removed (e.g., by another process)
        if removed_count == worker_urls.len() {
            debug!("Removed {} worker(s) from registry", removed_count);
        } else {
            warn!(
                "Removed {} of {} workers (some may have been removed by another process)",
                removed_count,
                worker_urls.len()
            );
        }

        // Update Layer 3 worker pool size metrics for unique
        // configurations. WorkerMonitor subscribes to registry events
        // directly and reconciles per-group polling state on its own,
        // so this step no longer needs to push group-removed
        // notifications.
        for (worker_type, connection_mode, model_id) in &unique_configs {
            // Get labels before moving values into get_workers_filtered
            let worker_type_label = worker_type.as_metric_label();
            let connection_mode_label = connection_mode.as_metric_label();

            let pool_size = app_context
                .worker_registry
                .get_workers_filtered(
                    Some(model_id),
                    Some(*worker_type),
                    Some(*connection_mode),
                    None,
                    false,
                )
                .len();

            Metrics::set_worker_pool_size(
                worker_type_label,
                connection_mode_label,
                model_id,
                pool_size,
            );
        }

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}
