//! Data Parallel (DP) information discovery step.

use std::collections::HashMap;

use async_trait::async_trait;
use tracing::{debug, warn};
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use super::discover_metadata::get_server_info;
use crate::{
    worker::{ConnectionMode, UNKNOWN_MODEL_ID},
    workflow::data::{WorkerKind, WorkerWorkflowData},
};

/// DP (Data Parallel) information for a worker.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DpInfo {
    pub dp_size: usize,
    pub model_id: String,
}

/// Get DP info for a worker URL.
pub async fn get_dp_info(url: &str, api_key: Option<&str>) -> Result<DpInfo, String> {
    let info = get_server_info(url, api_key).await?;

    let dp_size = info
        .dp_size
        .ok_or_else(|| format!("No dp_size in response from {url}"))?;

    let model_id = info
        .model_id
        .filter(|s| !s.is_empty())
        .or_else(|| info.served_model_name.filter(|s| !s.is_empty()))
        .or_else(|| {
            info.model_path
                .and_then(|path| path.split('/').next_back().map(|s| s.to_string()))
        })
        .unwrap_or_else(|| UNKNOWN_MODEL_ID.to_string());

    Ok(DpInfo { dp_size, model_id })
}

/// Build DP info from gRPC-discovered metadata labels (`dp_size` is reported
/// via GetServerInfo and normalized into the label set by metadata discovery).
pub fn dp_info_from_labels(labels: &HashMap<String, String>) -> Result<DpInfo, String> {
    let dp_size = labels
        .get("dp_size")
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .ok_or_else(|| "no positive dp_size in discovered metadata".to_string())?;

    let model_id = labels
        .get("model_id")
        .or_else(|| labels.get("served_model_name"))
        .filter(|s| !s.is_empty())
        .cloned()
        .or_else(|| {
            labels
                .get("model_path")
                .and_then(|p| p.rsplit('/').next())
                .filter(|s| !s.is_empty())
                .map(str::to_string)
        })
        .unwrap_or_else(|| UNKNOWN_MODEL_ID.to_string());

    Ok(DpInfo { dp_size, model_id })
}

/// Step 2b: Discover DP (Data Parallel) information (only for DP-aware workers).
pub struct DiscoverDPInfoStep;

#[async_trait]
impl StepExecutor<WorkerWorkflowData> for DiscoverDPInfoStep {
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

        if !app_context.router_config.dp_aware {
            debug!(
                "Worker {} is not DP-aware, skipping DP discovery",
                config.url
            );
            return Ok(StepResult::Success);
        }

        debug!("Discovering DP info for {} (DP-aware)", config.url);

        let dp_info = match context.data.connection_mode {
            // gRPC workers report dp_size through GetServerInfo, already
            // flattened into the discovered metadata labels
            Some(ConnectionMode::Grpc) => dp_info_from_labels(&context.data.discovered_labels)
                .map_err(|e| WorkflowError::StepFailed {
                    step_id: StepId::new("discover_dp_info"),
                    message: format!("Failed to get DP info for {}: {e}", config.url),
                })?,
            _ => {
                // HTTP DP discovery uses SGLang's /server_info which exposes
                // dp_size — other runtimes don't expose DP info this way
                let runtime = context.data.detected_runtime_type.as_deref();
                if runtime != Some("sglang") {
                    warn!(
                        "DP discovery is not supported for {} HTTP workers ({}), skipping",
                        runtime.unwrap_or("unknown"),
                        config.url
                    );
                    return Ok(StepResult::Success);
                }
                get_dp_info(&config.url, config.api_key.as_deref())
                    .await
                    .map_err(|e| WorkflowError::StepFailed {
                        step_id: StepId::new("discover_dp_info"),
                        message: format!("Failed to get DP info: {e}"),
                    })?
            }
        };

        debug!(
            "Discovered DP size {} for {} (model: {})",
            dp_info.dp_size, config.url, dp_info.model_id
        );

        context.data.dp_info = Some(dp_info);
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn dp_info_from_labels_reads_dp_size_and_model() {
        let info =
            dp_info_from_labels(&labels(&[("dp_size", "4"), ("served_model_name", "m")])).unwrap();
        assert_eq!(info.dp_size, 4);
        assert_eq!(info.model_id, "m");
    }

    #[test]
    fn dp_info_from_labels_model_path_fallback() {
        let info =
            dp_info_from_labels(&labels(&[("dp_size", "2"), ("model_path", "org/repo")])).unwrap();
        assert_eq!(info.model_id, "repo");
    }

    #[test]
    fn dp_info_from_labels_unknown_model_when_absent() {
        let info = dp_info_from_labels(&labels(&[("dp_size", "2")])).unwrap();
        assert_eq!(info.model_id, UNKNOWN_MODEL_ID);
    }

    #[test]
    fn dp_info_from_labels_rejects_missing_or_zero_dp_size() {
        assert!(dp_info_from_labels(&labels(&[])).is_err());
        assert!(dp_info_from_labels(&labels(&[("dp_size", "0")])).is_err());
        assert!(dp_info_from_labels(&labels(&[("dp_size", "abc")])).is_err());
    }
}
