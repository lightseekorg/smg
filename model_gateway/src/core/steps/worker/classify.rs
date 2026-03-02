//! Worker type classification step.
//!
//! Auto-detects whether a worker endpoint is a local inference backend
//! (sglang, vllm, trtllm) or an external cloud API (OpenAI, Anthropic, etc.)
//! by probing the endpoint. Users no longer need to set `runtime_type`.

use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use tracing::debug;
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use super::local::{try_grpc_reachable, try_http_reachable};
use crate::core::{
    steps::workflow_data::{WorkerKind, WorkerWorkflowData},
    worker::RuntimeType,
};

/// Check if `/v1/models` is reachable (any HTTP response, including 401/403).
///
/// Returns `true` if we got ANY HTTP response (the endpoint exists).
/// Returns `false` only on connection errors (timeout, refused, DNS failure).
async fn is_models_endpoint_reachable(
    url: &str,
    timeout_secs: u64,
    client: &Client,
    api_key: Option<&str>,
) -> bool {
    let base = url.trim_end_matches('/');
    let models_url = format!("{base}/v1/models");
    let mut req = client
        .get(&models_url)
        .timeout(Duration::from_secs(timeout_secs));
    if let Some(key) = api_key {
        req = req.bearer_auth(key);
    }
    // Any HTTP response (even 401/403) proves the endpoint exists
    req.send().await.is_ok()
}

/// Step 0: Classify the worker as Local or External.
///
/// Detection logic:
/// 1. Explicit `RuntimeType::External` → External
/// 2. Explicit non-default runtime (`Vllm`/`Trtllm`) → Local
/// 3. Auto-detect (default `Sglang` = "auto"):
///    a. `/health` responds → Local (only local backends expose `/health`)
///    b. gRPC health responds → Local (external APIs never use gRPC)
///    c. `/v1/models` returns any HTTP response → External
///    d. Nothing reachable → default Local (backend may still be starting)
pub struct ClassifyWorkerTypeStep;

#[async_trait]
impl StepExecutor<WorkerWorkflowData> for ClassifyWorkerTypeStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let config = &context.data.config;
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;

        // 1. Explicit override: user set runtime_type to External
        if config.runtime_type == RuntimeType::External {
            debug!("Worker {} explicitly configured as External", config.url);
            context.data.worker_kind = Some(WorkerKind::External);
            return Ok(StepResult::Success);
        }

        // 2. Explicit non-default runtime (Vllm/Trtllm) → must be Local
        if config.runtime_type != RuntimeType::default() {
            debug!(
                "Worker {} explicitly configured as {} → Local",
                config.url, config.runtime_type
            );
            context.data.worker_kind = Some(WorkerKind::Local);
            return Ok(StepResult::Success);
        }

        // 3. Auto-detect with quick probes (2s timeout)
        let timeout = 2;
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
        // Try /v1/models — any HTTP response (including 401/403) proves the
        // endpoint is a cloud API. Connection error means unreachable.
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
