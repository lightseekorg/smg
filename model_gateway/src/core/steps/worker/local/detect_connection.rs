//! Connection mode detection step.

use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use tracing::debug;
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use super::strip_protocol;
use crate::{
    core::{steps::workflow_data::LocalWorkerWorkflowData, ConnectionMode},
    routers::grpc::client::GrpcClient,
};

/// Try HTTP health check.
async fn try_http_health_check(
    url: &str,
    timeout_secs: u64,
    client: &Client,
) -> Result<(), String> {
    let is_https = url.starts_with("https://");
    let protocol = if is_https { "https" } else { "http" };
    let clean_url = strip_protocol(url);
    let health_url = format!("{}://{}/health", protocol, clean_url);

    client
        .get(&health_url)
        .timeout(Duration::from_secs(timeout_secs))
        .send()
        .await
        .and_then(reqwest::Response::error_for_status)
        .map_err(|e| format!("Health check failed: {}", e))?;

    Ok(())
}

/// Perform gRPC health check with runtime type.
async fn do_grpc_health_check(
    grpc_url: &str,
    timeout_secs: u64,
    runtime_type: &str,
) -> Result<(), String> {
    let connect_future = GrpcClient::connect(grpc_url, runtime_type);
    let client = tokio::time::timeout(Duration::from_secs(timeout_secs), connect_future)
        .await
        .map_err(|_| "gRPC connection timeout".to_string())?
        .map_err(|e| format!("gRPC connection failed: {}", e))?;

    let health_future = client.health_check();
    tokio::time::timeout(Duration::from_secs(timeout_secs), health_future)
        .await
        .map_err(|_| "gRPC health check timeout".to_string())?
        .map_err(|e| format!("gRPC health check failed: {}", e))?;

    Ok(())
}

/// Try gRPC health check (tries SGLang first, then vLLM, then TensorRT-LLM if not specified).
/// Returns the detected runtime type on success.
async fn try_grpc_health_check(
    url: &str,
    timeout_secs: u64,
    runtime_type: Option<&str>,
) -> Result<String, String> {
    let grpc_url = if url.starts_with("grpc://") {
        url.to_string()
    } else {
        format!("grpc://{}", strip_protocol(url))
    };

    match runtime_type {
        Some(runtime) => {
            do_grpc_health_check(&grpc_url, timeout_secs, runtime).await?;
            Ok(runtime.to_string())
        }
        None => {
            // Try SGLang first, then vLLM, then TensorRT-LLM as fallback
            if do_grpc_health_check(&grpc_url, timeout_secs, "sglang")
                .await
                .is_ok()
            {
                return Ok("sglang".to_string());
            }
            if do_grpc_health_check(&grpc_url, timeout_secs, "vllm")
                .await
                .is_ok()
            {
                return Ok("vllm".to_string());
            }
            do_grpc_health_check(&grpc_url, timeout_secs, "trtllm")
                .await
                .map_err(|e| {
                    format!("gRPC failed (tried SGLang, vLLM, and TensorRT-LLM): {}", e)
                })?;
            Ok("trtllm".to_string())
        }
    }
}

/// Step 1: Detect connection mode by probing HTTP and gRPC.
pub struct DetectConnectionModeStep;

#[async_trait]
impl StepExecutor<LocalWorkerWorkflowData> for DetectConnectionModeStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<LocalWorkerWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let config = &context.data.config;
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;

        debug!(
            "Detecting connection mode for {} (timeout: {:?}s, max_attempts: {})",
            config.url, config.health.timeout_secs, config.max_connection_attempts
        );

        // Try both protocols in parallel
        let url = config.url.clone();
        // Use per-worker timeout override if set, otherwise fall back to router default
        let timeout = config
            .health
            .timeout_secs
            .unwrap_or(app_context.router_config.health_check.timeout_secs);
        let client = &app_context.client;
        // Auto-detect runtime unless explicitly set to non-default
        let runtime_type_str = config.runtime_type.to_string();
        let runtime_hint = if runtime_type_str == "sglang" {
            None
        } else {
            Some(runtime_type_str.as_str())
        };

        let (http_result, grpc_result) = tokio::join!(
            try_http_health_check(&url, timeout, client),
            try_grpc_health_check(&url, timeout, runtime_hint)
        );

        let (connection_mode, detected_runtime) = match (http_result, grpc_result) {
            (Ok(_), _) => {
                debug!("{} detected as HTTP", config.url);
                (ConnectionMode::Http, None)
            }
            (_, Ok(runtime)) => {
                debug!("{} detected as gRPC (runtime: {})", config.url, runtime);
                (ConnectionMode::Grpc, Some(runtime))
            }
            (Err(http_err), Err(grpc_err)) => {
                return Err(WorkflowError::StepFailed {
                    step_id: StepId::new("detect_connection_mode"),
                    message: format!(
                        "Both HTTP and gRPC health checks failed for {}: HTTP: {}, gRPC: {}",
                        config.url, http_err, grpc_err
                    ),
                });
            }
        };

        context.data.connection_mode = Some(connection_mode);
        // Save detected runtime type from health check for use in metadata discovery
        if let Some(runtime) = detected_runtime {
            context.data.detected_runtime_type = Some(runtime);
        }
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        true
    }
}
