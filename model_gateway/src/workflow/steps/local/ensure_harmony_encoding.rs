//! Preload the Harmony encoding when a gpt-oss worker registers.
//!
//! gRPC gpt-oss workers are the only consumers of the Harmony encoding, whose
//! vocab may be read from `TIKTOKEN_ENCODINGS_BASE` or downloaded on first
//! load. Loading here fails this worker's registration (within the step's
//! retry budget) instead of panicking the gateway at router construction;
//! non-gpt-oss and HTTP workers skip.

use async_trait::async_trait;
use tracing::info;
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::{
    routers::grpc::harmony::{try_harmony_encoding, HarmonyDetector},
    worker::ConnectionMode,
    workflow::data::{WorkerKind, WorkerWorkflowData},
};

pub struct EnsureHarmonyEncodingStep;

#[async_trait]
impl StepExecutor<WorkerWorkflowData> for EnsureHarmonyEncodingStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        if context.data.worker_kind != Some(WorkerKind::Local) {
            return Ok(StepResult::Skip);
        }

        // Only the gRPC pipeline serves Harmony natively; HTTP proxies to the backend.
        if context.data.connection_mode != Some(ConnectionMode::Grpc) {
            return Ok(StepResult::Skip);
        }

        let workers = context
            .data
            .actual_workers
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("workers".to_string()))?;

        if !workers
            .iter()
            .any(|worker| HarmonyDetector::is_harmony_worker(worker.as_ref()))
        {
            return Ok(StepResult::Skip);
        }

        // Vocab load may hit disk or network; keep it off the async runtime.
        tokio::task::spawn_blocking(try_harmony_encoding)
            .await
            .map_err(|e| WorkflowError::StepFailed {
                step_id: StepId::new("ensure_harmony_encoding"),
                message: format!("Harmony encoding load task failed: {e}"),
            })?
            .map_err(|message| WorkflowError::StepFailed {
                step_id: StepId::new("ensure_harmony_encoding"),
                message,
            })?;

        info!("Harmony encoding ready for gpt-oss worker registration");
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        true // vocab download can hit transient network failures
    }
}
