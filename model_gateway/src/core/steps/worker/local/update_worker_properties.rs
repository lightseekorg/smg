//! Step to update worker properties.

use std::sync::Arc;

use async_trait::async_trait;
use tracing::{debug, info};
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::core::{steps::workflow_data::WorkerUpdateWorkflowData, BasicWorkerBuilder, Worker};

/// Step to update worker properties.
///
/// This step creates new worker instances with updated properties and
/// re-registers them to replace the old workers in the registry.
pub struct UpdateWorkerPropertiesStep;

#[async_trait]
impl StepExecutor<WorkerUpdateWorkflowData> for UpdateWorkerPropertiesStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerUpdateWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let request = &context.data.config;
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?
            .clone();
        let workers_to_update = context
            .data
            .workers_to_update
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("workers_to_update".to_string()))?
            .clone();

        debug!(
            "Updating properties for {} worker(s)",
            workers_to_update.len()
        );

        let mut updated_workers: Vec<Arc<dyn Worker>> = Vec::with_capacity(workers_to_update.len());

        for worker in workers_to_update.iter() {
            // Build updated labels - merge new labels into existing ones
            let mut updated_labels = worker.metadata().spec.labels.clone();
            if let Some(ref new_labels) = request.labels {
                for (key, value) in new_labels {
                    updated_labels.insert(key.clone(), value.clone());
                }
            }

            // Resolve priority and cost: use update value if specified, otherwise keep existing
            let updated_priority = request.priority.unwrap_or(worker.priority());
            let updated_cost = request.cost.unwrap_or(worker.cost());

            // Build updated health config from resolved runtime config
            let existing_health = &worker.metadata().health_config;
            let updated_health_config = match &request.health {
                Some(update) => update.apply_to(existing_health),
                None => existing_health.clone(),
            };
            let health_endpoint = worker.metadata().health_endpoint.clone();

            // Determine API key: use new one if provided, otherwise keep existing
            let updated_api_key = request
                .api_key
                .clone()
                .or_else(|| worker.metadata().spec.api_key.clone());

            // Create a new worker with updated properties
            // Use base_url() so DP workers start from the un-suffixed URL
            let mut builder = BasicWorkerBuilder::new(worker.base_url())
                .worker_type(*worker.worker_type())
                .connection_mode(*worker.connection_mode())
                .runtime_type(worker.metadata().spec.runtime_type)
                .labels(updated_labels)
                .health_config(updated_health_config.clone())
                .health_endpoint(&health_endpoint)
                .models(worker.metadata().spec.models.clone())
                .priority(updated_priority)
                .cost(updated_cost);

            if let Some(ref api_key) = updated_api_key {
                builder = builder.api_key(api_key.clone());
            }

            // Preserve DP configuration if the worker is DP-aware
            if worker.is_dp_aware() {
                builder = builder.dp_config(
                    worker.dp_rank().expect("DP-aware worker must have dp_rank"),
                    worker.dp_size().expect("DP-aware worker must have dp_size"),
                );
            }

            let new_worker: Arc<dyn Worker> = Arc::new(builder.build());

            // Re-register the worker (this replaces the old one)
            app_context.worker_registry.register(new_worker.clone());

            updated_workers.push(new_worker);
        }

        // Log result
        if updated_workers.len() == 1 {
            info!("Updated worker {}", updated_workers[0].url());
        } else {
            info!("Updated {} workers", updated_workers.len());
        }

        // Store updated workers for subsequent steps
        context.data.updated_workers = Some(updated_workers);

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}
