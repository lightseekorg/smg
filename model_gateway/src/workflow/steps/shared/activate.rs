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
        let workers = context
            .data
            .get_actual_workers()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("workers".to_string()))?;

        for worker in workers {
            if worker.status() != WorkerStatus::Ready {
                worker.set_status(WorkerStatus::Ready);
            }
        }

        info!("Activated {} worker(s)", workers.len());

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}
