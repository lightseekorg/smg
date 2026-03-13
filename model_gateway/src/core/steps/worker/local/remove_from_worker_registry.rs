//! Step to remove workers from worker registry.

use std::collections::{HashMap, HashSet};

use async_trait::async_trait;
use tracing::{debug, warn};
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::{
    core::{
        steps::workflow_data::WorkerRemovalWorkflowData,
        worker::{ConnectionModeExt, WorkerTypeExt},
        WorkerGroupKey,
    },
    observability::metrics::Metrics,
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

        // Collect unique worker configurations and their URLs before removal.
        // We need the URLs pre-removal so LoadMonitor can clean up stale load entries.
        let mut urls_by_group: HashMap<
            (
                openai_protocol::worker::WorkerType,
                openai_protocol::worker::ConnectionMode,
                String,
            ),
            Vec<String>,
        > = HashMap::new();
        for url in worker_urls {
            if let Some(w) = app_context.worker_registry.get_by_url(url) {
                let meta = w.metadata();
                let key = (
                    meta.spec.worker_type,
                    meta.spec.connection_mode,
                    w.model_id().to_string(),
                );
                urls_by_group.entry(key).or_default().push(url.clone());
            }
        }
        let unique_configs: HashSet<_> = urls_by_group.keys().cloned().collect();

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

        // Update Layer 3 worker pool size metrics for unique configurations
        // and notify LoadMonitor when groups become empty
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

            // If the group is now empty, stop its load monitor and clean up entries
            if pool_size == 0 {
                if let Some(ref load_monitor) = app_context.load_monitor {
                    let key = WorkerGroupKey {
                        model_id: model_id.clone(),
                        worker_type: *worker_type,
                        connection_mode: *connection_mode,
                    };
                    let removed_urls = urls_by_group
                        .get(&(*worker_type, *connection_mode, model_id.clone()))
                        .map(|v| v.as_slice())
                        .unwrap_or_default();
                    load_monitor.on_group_removed(&key, removed_urls).await;
                }
            }
        }

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}
