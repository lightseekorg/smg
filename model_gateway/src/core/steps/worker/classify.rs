//! Worker type classification step.
//!
//! Classifies a worker as Local or External based on the `runtime_type` field.
//! Only `RuntimeType::External` yields an external worker; all other explicit
//! runtime types (Sglang, Vllm, Trtllm) are local. When `Unspecified`
//! (the default), the step auto-detects by probing the endpoint.

use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use tracing::debug;
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use super::util::{http_base_url, try_grpc_reachable, try_http_reachable};
use crate::core::{
    steps::workflow_data::{WorkerKind, WorkerWorkflowData},
    worker::RuntimeType,
};

/// Quick-probe timeout for classification. Deliberately short — the full
/// connection timeout is applied later by `DetectConnectionModeStep`.
const CLASSIFY_PROBE_TIMEOUT_SECS: u64 = 2;

/// Check if `/v1/models` responds with a non-5xx status.
///
/// Any non-server-error response (2xx, 401, 403, 429, etc.) proves the endpoint
/// is a real API. 5xx is excluded — a starting local backend may return 503.
async fn is_models_endpoint_reachable(
    url: &str,
    timeout_secs: u64,
    client: &Client,
    api_key: Option<&str>,
) -> bool {
    let base = http_base_url(url);
    let models_url = format!("{base}/v1/models");
    let mut req = client
        .get(&models_url)
        .timeout(Duration::from_secs(timeout_secs));
    if let Some(key) = api_key {
        req = req.bearer_auth(key);
    }
    match req.send().await {
        Ok(resp) => !resp.status().is_server_error(),
        Err(_) => false,
    }
}

/// Step 0: Classify the worker as Local or External.
///
/// Detection logic:
/// 1. Explicit `RuntimeType::External` → External
/// 2. Explicit local runtime (Sglang, Vllm, Trtllm) → Local
/// 3. `Unspecified` (default) → auto-detect via endpoint probes:
///    a. `/health` responds → Local (only local backends expose `/health`)
///    b. gRPC health responds → Local (external APIs never use gRPC)
///    c. `/v1/models` returns non-5xx → External (5xx ignored — could be starting local)
///    d. Nothing reachable → default Local (backend may still be starting)
pub struct ClassifyWorkerTypeStep;

#[async_trait]
impl StepExecutor<WorkerWorkflowData> for ClassifyWorkerTypeStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let config = &context.data.config;

        // Explicit External → External
        if config.runtime_type == RuntimeType::External {
            debug!("Worker {} explicitly configured as External", config.url);
            context.data.worker_kind = Some(WorkerKind::External);
            return Ok(StepResult::Success);
        }

        // Explicit local runtime (Sglang, Vllm, Trtllm) → Local
        if config.runtime_type.is_specified() {
            debug!(
                "Worker {} explicitly configured as {} → Local",
                config.url, config.runtime_type
            );
            context.data.worker_kind = Some(WorkerKind::Local);
            return Ok(StepResult::Success);
        }

        // Unspecified — auto-detect with quick probes
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;
        let timeout = CLASSIFY_PROBE_TIMEOUT_SECS;
        let client = &app_context.client;

        // Try /health — only local backends expose this
        if try_http_reachable(&config.url, timeout, client)
            .await
            .is_ok()
        {
            debug!("Worker {} responded to /health → Local", config.url);
            context.data.worker_kind = Some(WorkerKind::Local);
            return Ok(StepResult::Success);
        }

        // Try gRPC — external APIs never use gRPC
        if try_grpc_reachable(&config.url, timeout).await.is_ok() {
            debug!("Worker {} responded to gRPC health → Local", config.url);
            context.data.worker_kind = Some(WorkerKind::Local);
            return Ok(StepResult::Success);
        }

        // /health and gRPC both failed.
        // Try /v1/models — any non-5xx response proves the endpoint is a cloud
        // API. 5xx is excluded (starting local backend may return 503).
        if is_models_endpoint_reachable(&config.url, timeout, client, config.api_key.as_deref())
            .await
        {
            debug!(
                "Worker {} responded to /v1/models (no /health) → External",
                config.url
            );
            context.data.worker_kind = Some(WorkerKind::External);
            return Ok(StepResult::Success);
        }

        // Nothing responded — assume Local (backend may still be starting;
        // detect_connection_mode will retry with the full timeout).
        debug!(
            "Worker {} not reachable on any probe → defaulting to Local",
            config.url
        );
        context.data.worker_kind = Some(WorkerKind::Local);
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}
