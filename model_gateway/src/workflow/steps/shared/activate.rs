//! Unified worker activation step.

use async_trait::async_trait;
use openai_protocol::worker::WorkerStatus;
use tracing::info;
use wfaas::{
    StepExecutor, StepResult, WorkflowContext, WorkflowData, WorkflowError, WorkflowResult,
};

use crate::workflow::data::WorkerRegistrationData;

/// Final step in any worker registration workflow: flip Pending → Ready.
pub struct ActivateWorkersStep;

#[async_trait]
impl<D: WorkerRegistrationData + WorkflowData> StepExecutor<D> for ActivateWorkersStep {
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

        let mut activated = false;
        for worker in workers {
            if worker.status() != WorkerStatus::Ready {
                worker.set_status(WorkerStatus::Ready);
                activated = true;
            }
        }

        if activated {
            if let Some(admission) = &app_context.prefill_admission {
                admission.notify_capacity_changed();
            }
        }

        info!("Activated {} worker(s)", workers.len());

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}
