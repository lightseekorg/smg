//! Connection mode detection step.
//!
//! Determines whether a worker communicates via HTTP or gRPC.
//! This step only answers "HTTP or gRPC?" — backend runtime detection
//! (sglang vs vllm vs trtllm) is handled by the separate DetectBackendStep.

use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use smg_grpc_client::health::{
    proto::health_check_response::ServingStatus as HealthServingStatus, HealthClient,
};
use tracing::debug;
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::{
    core::{
        steps::{
            worker::util::strip_protocol,
            workflow_data::{WorkerKind, WorkerWorkflowData},
        },
        ConnectionMode, WorkerType,
    },
    routers::grpc::client::GrpcClient,
};

/// Try HTTP health check.
async fn try_http_reachable(url: &str, timeout_secs: u64, client: &Client) -> Result<(), String> {
    let is_https = url.starts_with("https://");
    let protocol = if is_https { "https" } else { "http" };
    let clean_url = strip_protocol(url);
    let health_url = format!("{protocol}://{clean_url}/health");

    client
        .get(&health_url)
        .timeout(Duration::from_secs(timeout_secs))
        .send()
        .await
        .and_then(reqwest::Response::error_for_status)
        .map_err(|e| format!("Health check failed: {e}"))?;

    Ok(())
}

/// Perform a single gRPC health check with a specific runtime type.
///
/// Shared with `detect_backend` which uses this for runtime identification.
pub(super) async fn do_grpc_health_check(
    grpc_url: &str,
    timeout_secs: u64,
    runtime_type: &str,
) -> Result<(), String> {
    let connect_future = GrpcClient::connect(grpc_url, runtime_type);
    let client = tokio::time::timeout(Duration::from_secs(timeout_secs), connect_future)
        .await
        .map_err(|_| "gRPC connection timeout".to_string())?
        .map_err(|e| format!("gRPC connection failed: {e}"))?;

    let health_future = client.health_check();
    tokio::time::timeout(Duration::from_secs(timeout_secs), health_future)
        .await
        .map_err(|_| "gRPC health check timeout".to_string())?
        .map_err(|e| format!("gRPC health check failed: {e}"))?;

    Ok(())
}

/// Try gRPC health check for encoder workers (uses grpc.health.v1)
async fn try_encoder_grpc_health_check(url: &str, timeout_secs: u64) -> Result<(), String> {
    let grpc_url = if url.starts_with("grpc://") {
        url.to_string()
    } else {
        format!("grpc://{}", strip_protocol(url))
    };

    let connect_future = HealthClient::connect(&grpc_url);
    let client = tokio::time::timeout(Duration::from_secs(timeout_secs), connect_future)
        .await
        .map_err(|_| "Encoder gRPC connection timeout".to_string())?
        .map_err(|e| format!("Encoder gRPC connection failed: {e}"))?;

    let health_future = client.check("");
    let response = tokio::time::timeout(Duration::from_secs(timeout_secs), health_future)
        .await
        .map_err(|_| "Encoder gRPC health check timeout".to_string())?
        .map_err(|e| format!("Encoder gRPC health check failed: {e}"))?;

    let serving = response.status == HealthServingStatus::Serving as i32;
    if serving {
        Ok(())
    } else {
        Err("Encoder not serving".to_string())
    }
}

/// Check if gRPC is reachable by trying all known runtime types in parallel.
///
/// We don't care which runtime it is here — that's detect_backend's job.
/// We just need to know: does this endpoint speak gRPC at all?
async fn try_grpc_reachable(
    url: &str,
    timeout_secs: u64,
    include_encoder: bool,
) -> Result<(), String> {
    let grpc_url = if url.starts_with("grpc://") {
        url.to_string()
    } else {
        format!("grpc://{}", strip_protocol(url))
    };

    if include_encoder {
        let (encoder, sglang, vllm, trtllm) = tokio::join!(
            try_encoder_grpc_health_check(&grpc_url, timeout_secs),
            do_grpc_health_check(&grpc_url, timeout_secs, "sglang"),
            do_grpc_health_check(&grpc_url, timeout_secs, "vllm"),
            do_grpc_health_check(&grpc_url, timeout_secs, "trtllm"),
        );

        match (encoder, sglang, vllm, trtllm) {
            (Ok(()), _, _, _)
            | (_, Ok(()), _, _)
            | (_, _, Ok(()), _)
            | (_, _, _, Ok(())) => Ok(()),
            (Err(e0), Err(e1), Err(e2), Err(e3)) => Err(format!(
                "gRPC not reachable (tried encoder, sglang, vllm, trtllm): encoder={e0}, sglang={e1}, vllm={e2}, trtllm={e3}",
            )),
        }
    } else {
        let (sglang, vllm, trtllm) = tokio::join!(
            do_grpc_health_check(&grpc_url, timeout_secs, "sglang"),
            do_grpc_health_check(&grpc_url, timeout_secs, "vllm"),
            do_grpc_health_check(&grpc_url, timeout_secs, "trtllm"),
        );

        match (sglang, vllm, trtllm) {
            (Ok(()), _, _) | (_, Ok(()), _) | (_, _, Ok(())) => Ok(()),
            (Err(e0), Err(e1), Err(e2)) => Err(format!(
                "gRPC not reachable (tried sglang, vllm, trtllm): sglang={e0}, vllm={e1}, trtllm={e2}",
            )),
        }
    }
}
/// Step 1: Detect connection mode (HTTP vs gRPC).
///
/// Probes both protocols in parallel. HTTP takes priority if both succeed.
/// Does NOT detect backend runtime — that's handled by DetectBackendStep.
pub struct DetectConnectionModeStep;

#[async_trait]
impl StepExecutor<WorkerWorkflowData> for DetectConnectionModeStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        if context.data.worker_kind != Some(WorkerKind::Local) {
            return Ok(StepResult::Skip);
        }

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

        let url = config.url.clone();
        let timeout = config
            .health
            .timeout_secs
            .unwrap_or(app_context.router_config.health_check.timeout_secs);
        let client = &app_context.client;

        let include_encoder = config.worker_type == WorkerType::Encode;
        let (http_result, grpc_result) = tokio::join!(
            try_http_reachable(&url, timeout, client),
            try_grpc_reachable(&url, timeout, include_encoder)
        );

        let connection_mode = match (http_result, grpc_result) {
            (Ok(()), _) => {
                debug!("{} detected as HTTP", config.url);
                ConnectionMode::Http
            }
            (_, Ok(())) => {
                debug!("{} detected as gRPC", config.url);
                ConnectionMode::Grpc
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
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        true
    }
}
