//! Unified worker activation step.

use async_trait::async_trait;
use tracing::info;
use wfaas::{
    StepExecutor, StepResult, WorkflowContext, WorkflowData, WorkflowError, WorkflowResult,
};

use crate::workflow::data::WorkerRegistrationData;

/// Unified step to activate workers that won't be promoted by the health checker.
///
/// Two cases require explicit activation:
///
/// 1. **Per-worker `disable_health_check == true`**: the health checker
///    skips this worker entirely, so the builder already initializes it
///    Ready and this step is a no-op (guarded against redundant mutation).
///
/// 2. **Global `disable_health_check == true`**: `server.rs` does not start
///    the health checker at all. Without this step, every worker would
///    stay Pending forever. We force-activate them here so they become
///    routable immediately.
///
/// Workers with health checks enabled (per-worker AND global) remain in
/// Pending — the running health checker will promote them after
/// `success_threshold` consecutive passes.
pub struct ActivateWorkersStep;

#[async_trait]
impl<D: WorkerRegistrationData + WorkflowData> StepExecutor<D> for ActivateWorkersStep {
    async fn execute(&self, context: &mut WorkflowContext<D>) -> WorkflowResult<StepResult> {
        // Check whether the global health checker will run. If it won't,
        // we need to force-activate every worker because no background
        // task will promote them out of Pending.
        let global_health_disabled = context
            .data
            .get_app_context()
            .map(|ctx| ctx.router_config.health_check.disable_health_check)
            .unwrap_or(false);

        let workers = context
            .data
            .get_actual_workers()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("workers".to_string()))?;

        let mut activated = 0;
        for worker in workers {
            let worker_health_disabled = worker.metadata().health_config.disable_health_check;
            // A worker needs explicit activation if either:
            //   - it has health checks disabled (builder already set it Ready,
            //     but we guard against bugs), or
            //   - the global health checker won't run at all.
            if worker_health_disabled || global_health_disabled {
                if !worker.is_healthy() {
                    worker.set_healthy(true);
                }
                activated += 1;
            }
            // Otherwise: health-checked workers stay Pending — the running
            // health checker will promote them.
        }

        info!(
            "Activated {activated}/{} worker(s) (global_health_disabled={global_health_disabled})",
            workers.len()
        );

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}
