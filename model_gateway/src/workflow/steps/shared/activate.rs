//! Unified worker activation step.

use async_trait::async_trait;
use tracing::info;
use wfaas::{
    StepExecutor, StepResult, WorkflowContext, WorkflowData, WorkflowError, WorkflowResult,
};

use crate::workflow::data::WorkerRegistrationData;

/// Final step in any worker registration workflow: flip Pending → Ready.
///
/// By the time this step runs, the workflow has already proven the worker
/// is reachable via `DetectBackendStep` (HTTP `/health` GET or gRPC
/// `health_check` call). That probe is functionally equivalent to one
/// successful pass of the state machine's `check_health_async()`. Treating
/// it as the activation signal is what preserves the pre-state-machine
/// behavior at the workflow boundary — workers added via the workflow are
/// routable on the very next request, no startup latency.
///
/// The state machine still applies for ongoing failures: the registry's
/// background health checker will demote the worker (Ready → NotReady →
/// Failed) if probes start failing later. The `Pending` state is reserved
/// for paths that don't go through the workflow's connectivity proof
/// (e.g. mesh-imported workers in a future PR).
pub struct ActivateWorkersStep;

#[async_trait]
impl<D: WorkerRegistrationData + WorkflowData> StepExecutor<D> for ActivateWorkersStep {
    async fn execute(&self, context: &mut WorkflowContext<D>) -> WorkflowResult<StepResult> {
        let workers = context
            .data
            .get_actual_workers()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("workers".to_string()))?;

        for worker in workers {
            if !worker.is_healthy() {
                worker.set_healthy(true);
            }
        }

        info!("Activated {} worker(s)", workers.len());

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}
