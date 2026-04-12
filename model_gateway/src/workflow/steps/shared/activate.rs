//! Unified worker activation step.

use async_trait::async_trait;
use tracing::info;
use wfaas::{
    StepExecutor, StepResult, WorkflowContext, WorkflowData, WorkflowError, WorkflowResult,
};

use crate::workflow::data::WorkerRegistrationData;

/// Unified step to activate workers that have health checks disabled.
///
/// Workers with `disable_health_check == true` are set to Ready immediately.
/// Workers with health checks enabled remain in Pending — the health checker
/// will promote them to Ready after `success_threshold` consecutive passes.
pub struct ActivateWorkersStep;

#[async_trait]
impl<D: WorkerRegistrationData + WorkflowData> StepExecutor<D> for ActivateWorkersStep {
    async fn execute(&self, context: &mut WorkflowContext<D>) -> WorkflowResult<StepResult> {
        let workers = context
            .data
            .get_actual_workers()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("workers".to_string()))?;

        let mut activated = 0;
        for worker in workers {
            if worker.metadata().health_config.disable_health_check {
                // Builder already sets these workers Ready. Guard against
                // redundant mutation to avoid extra state churn and keep
                // future status-change hooks quiet.
                if !worker.is_healthy() {
                    worker.set_healthy(true);
                }
                activated += 1;
            }
            // Health-checked workers stay in Pending — the health checker promotes them.
        }

        info!(
            "Activated {activated}/{} worker(s) (health-check-disabled only)",
            workers.len()
        );

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}
