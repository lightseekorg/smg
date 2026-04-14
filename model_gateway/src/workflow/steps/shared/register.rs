//! Unified worker registration step.

use std::{collections::HashSet, sync::Arc};

use async_trait::async_trait;
use tracing::debug;
use wfaas::{
    StepExecutor, StepResult, WorkflowContext, WorkflowData, WorkflowError, WorkflowResult,
};

use crate::{
    observability::metrics::Metrics,
    worker::{
        worker::{ConnectionModeExt, WorkerTypeExt},
        WorkerRegistry,
    },
    workflow::data::WorkerRegistrationData,
};

/// Unified step to register workers in the registry.
///
/// Works with both single workers and batches. Always expects `workers` key
/// in context containing `Vec<Arc<dyn Worker>>`.
/// Works with any workflow data type that implements `WorkerRegistrationData`.
pub struct RegisterWorkersStep;

#[async_trait]
impl<D: WorkerRegistrationData + WorkflowData> StepExecutor<D> for RegisterWorkersStep {
    async fn execute(&self, context: &mut WorkflowContext<D>) -> WorkflowResult<StepResult> {
        let app_context = context
            .data
            .get_app_context()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?
            .clone();

        let workers = context
            .data
            .get_actual_workers()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("workers".to_string()))?;

        let mut worker_ids = Vec::with_capacity(workers.len());

        for worker in workers {
            let worker_id = app_context
                .worker_registry
                .register_or_replace(Arc::clone(worker));
            debug!(
                "Registered worker {} (model: {}) with ID {:?}",
                worker.url(),
                worker.model_id(),
                worker_id
            );
            worker_ids.push(worker_id);
        }

        // Update per-model retry config (last write wins).
        // Only update if the worker has non-empty retry overrides in its spec.
        for worker in workers {
            let resilience_spec = &worker.metadata().spec.resilience;
            let has_retry_overrides = resilience_spec.max_retries.is_some()
                || resilience_spec.initial_backoff_ms.is_some()
                || resilience_spec.max_backoff_ms.is_some()
                || resilience_spec.backoff_multiplier.is_some()
                || resilience_spec.jitter_factor.is_some()
                || resilience_spec.disable_retry.is_some();

            if has_retry_overrides {
                let resolved = worker.resilience();
                let retry_config = resolved.retry.clone();
                for model_id in WorkerRegistry::worker_model_ids(worker) {
                    app_context.worker_registry.set_model_retry_config(
                        &model_id,
                        retry_config.clone(),
                        resolved.retry_enabled,
                    );
                }
            }
        }

        // Collect unique worker configurations to avoid redundant metric updates
        let unique_configs: HashSet<_> = workers
            .iter()
            .map(|w| {
                let meta = w.metadata();
                (
                    meta.spec.worker_type,
                    meta.spec.connection_mode,
                    w.model_id().to_string(),
                )
            })
            .collect();

        // Update Layer 3 worker pool size metrics per unique
        // `(worker_type, connection_mode, model_id)`.
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

        // WorkerMonitor subscribes to registry events directly (see
        // `worker::monitor::WorkerMonitor::start_event_loop`), so this
        // step no longer has to push group-add notifications. The
        // monitor's event loop reconciles the impacted groups as soon
        // as the `WorkerEvent::Registered` (new worker) or
        // `WorkerEvent::Replaced` (same-URL update) broadcast fires
        // from `worker_registry.register_or_replace()` above.

        // Note: worker_ids are stored for potential future use but not persisted
        // as they are internal registry identifiers
        debug!(
            "Registered {} workers with IDs: {:?}",
            worker_ids.len(),
            worker_ids
        );

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}
