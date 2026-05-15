//! Connection mode detection step.
//!
//! Determines whether a worker communicates via HTTP or gRPC.
//! This step only answers "HTTP or gRPC?" — backend runtime detection
//! (sglang vs vllm vs trtllm) is handled by the separate DetectBackendStep.

use async_trait::async_trait;
use tracing::debug;
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::{
    worker::ConnectionMode,
    workflow::{
        data::{WorkerKind, WorkerWorkflowData},
        steps::util::{try_grpc_reachable, try_http_reachable},
    },
};

fn explicit_connection_mode(url: &str) -> Option<ConnectionMode> {
    if url.starts_with("grpc://") || url.starts_with("grpcs://") {
        Some(ConnectionMode::Grpc)
    } else if url.starts_with("http://") || url.starts_with("https://") {
        Some(ConnectionMode::Http)
    } else {
        None
    }
}

/// Step 1: Detect connection mode (HTTP vs gRPC).
///
/// Explicit URL schemes are honored. For bare host:port URLs, probes both
/// protocols in parallel and HTTP takes priority if both succeed.
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

        if let Some(connection_mode) = explicit_connection_mode(&url) {
            let result = match connection_mode {
                ConnectionMode::Http => try_http_reachable(&url, timeout, client).await,
                ConnectionMode::Grpc => try_grpc_reachable(&url, timeout).await,
            };

            match result {
                Ok(()) => {
                    debug!(
                        "{} explicitly configured as {}",
                        config.url, connection_mode
                    );
                    context.data.connection_mode = Some(connection_mode);
                    return Ok(StepResult::Success);
                }
                Err(err) => {
                    return Err(WorkflowError::StepFailed {
                        step_id: StepId::new("detect_connection_mode"),
                        message: format!(
                            "{connection_mode} health check failed for explicitly configured worker URL {}: {}",
                            config.url, err
                        ),
                    });
                }
            }
        }

        let (http_result, grpc_result) = tokio::join!(
            try_http_reachable(&url, timeout, client),
            try_grpc_reachable(&url, timeout)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_connection_mode_honors_grpc_scheme() {
        assert_eq!(
            explicit_connection_mode("grpc://localhost:30001"),
            Some(ConnectionMode::Grpc)
        );
        assert_eq!(
            explicit_connection_mode("grpcs://localhost:30001"),
            Some(ConnectionMode::Grpc)
        );
    }

    #[test]
    fn explicit_connection_mode_honors_http_schemes() {
        assert_eq!(
            explicit_connection_mode("http://localhost:30000"),
            Some(ConnectionMode::Http)
        );
        assert_eq!(
            explicit_connection_mode("https://example.com"),
            Some(ConnectionMode::Http)
        );
    }

    #[test]
    fn explicit_connection_mode_leaves_bare_urls_for_probe_detection() {
        assert_eq!(explicit_connection_mode("localhost:30000"), None);
    }
}
