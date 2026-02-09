//! Metadata discovery step for local workers.

use std::{collections::HashMap, time::Duration};

use async_trait::async_trait;
use once_cell::sync::Lazy;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{debug, warn};

use super::strip_protocol;
use crate::{
    core::{steps::workflow_data::LocalWorkerWorkflowData, ConnectionMode},
    routers::grpc::client::GrpcClient,
    workflow::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult},
};

// HTTP client for metadata fetching
static HTTP_CLIENT: Lazy<Client> = Lazy::new(|| {
    Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .expect("Failed to create HTTP client")
});

/// Server information returned from /server_info endpoint.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ServerInfo {
    #[serde(alias = "model")]
    pub model_id: Option<String>,
    pub model_path: Option<String>,
    pub served_model_name: Option<String>,
    pub tp_size: Option<usize>,
    pub dp_size: Option<usize>,
    pub load_balance_method: Option<String>,
    pub disaggregation_mode: Option<String>,
    pub version: Option<String>,
    pub max_batch_size: Option<usize>,
    pub max_total_tokens: Option<usize>,
    pub max_prefill_tokens: Option<usize>,
    pub max_running_requests: Option<usize>,
    pub max_num_reqs: Option<usize>,
}

/// Model information returned from /model_info endpoint.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelInfo {
    pub model_path: Option<String>,
    pub tokenizer_path: Option<String>,
    pub is_generation: Option<bool>,
    pub model_type: Option<String>,
    pub architectures: Option<Vec<String>>,
}

/// Fallback function to GET JSON from old endpoint (with "get_" prefix) for backward compatibility.
async fn get_json_fallback(
    base_url: &str,
    endpoint: &str,
    api_key: Option<&str>,
) -> Result<Value, String> {
    // FIXME: This fallback logic should be removed together with /get_server_info
    // and /get_model_info endpoints in http_server.py
    warn!(
        concat!(
            "Endpoint '/{}' returned 404, falling back to '/get_{}' for backward compatibility. ",
            "The '/get_{}' endpoint is deprecated and will be removed in a future version. ",
            "Please use '/{}' instead."
        ),
        endpoint, endpoint, endpoint, endpoint
    );

    let old_url = format!("{}/get_{}", base_url, endpoint);
    let mut req = HTTP_CLIENT.get(&old_url);
    if let Some(key) = api_key {
        req = req.bearer_auth(key);
    }

    let response = req
        .send()
        .await
        .map_err(|e| format!("Failed to connect to {}: {}", old_url, e))?;

    if !response.status().is_success() {
        return Err(format!(
            "Server returned status {} from {}",
            response.status(),
            old_url
        ));
    }

    response
        .json::<Value>()
        .await
        .map_err(|e| format!("Failed to parse response from {}: {}", old_url, e))
}

/// Get server info from /server_info endpoint.
pub async fn get_server_info(url: &str, api_key: Option<&str>) -> Result<ServerInfo, String> {
    let base_url = url.trim_end_matches('/');
    let server_info_url = format!("{}/server_info", base_url);

    let mut req = HTTP_CLIENT.get(&server_info_url);
    if let Some(key) = api_key {
        req = req.bearer_auth(key);
    }

    let response = req
        .send()
        .await
        .map_err(|e| format!("Failed to connect to {}: {}", server_info_url, e))?;

    // If /server_info returns 404, fallback to /get_server_info for backward compatibility
    if response.status() == reqwest::StatusCode::NOT_FOUND {
        let json = get_json_fallback(base_url, "server_info", api_key).await?;
        return serde_json::from_value(json)
            .map_err(|e| format!("Failed to parse server info: {}", e));
    }

    if !response.status().is_success() {
        return Err(format!(
            "Server returned status {} from {}",
            response.status(),
            server_info_url
        ));
    }

    let json = response
        .json::<Value>()
        .await
        .map_err(|e| format!("Failed to parse response from {}: {}", server_info_url, e))?;

    serde_json::from_value(json).map_err(|e| format!("Failed to parse server info: {}", e))
}

/// Get model info from /model_info endpoint.
pub async fn get_model_info(url: &str, api_key: Option<&str>) -> Result<ModelInfo, String> {
    let base_url = url.trim_end_matches('/');
    let model_info_url = format!("{}/model_info", base_url);

    let mut req = HTTP_CLIENT.get(&model_info_url);
    if let Some(key) = api_key {
        req = req.bearer_auth(key);
    }

    let response = req
        .send()
        .await
        .map_err(|e| format!("Failed to connect to {}: {}", model_info_url, e))?;

    // If /model_info returns 404, fallback to /get_model_info for backward compatibility
    if response.status() == reqwest::StatusCode::NOT_FOUND {
        let json = get_json_fallback(base_url, "model_info", api_key).await?;
        return serde_json::from_value(json)
            .map_err(|e| format!("Failed to parse model info: {}", e));
    }

    if !response.status().is_success() {
        return Err(format!(
            "Server returned status {} from {}",
            response.status(),
            model_info_url
        ));
    }

    response
        .json::<ModelInfo>()
        .await
        .map_err(|e| format!("Failed to parse response from {}: {}", model_info_url, e))
}

/// Fetch gRPC metadata (returns labels and runtime type).
/// The runtime_type should be pre-detected by the connection detection step.
async fn fetch_grpc_metadata(
    url: &str,
    runtime_type: &str,
) -> Result<(HashMap<String, String>, String), String> {
    let grpc_url = if url.starts_with("grpc://") {
        url.to_string()
    } else {
        format!("grpc://{}", strip_protocol(url))
    };

    let client = GrpcClient::connect(&grpc_url, runtime_type)
        .await
        .map_err(|e| format!("Failed to connect to gRPC: {}", e))?;

    // Fetch model info for labels
    let model_info = client
        .get_model_info()
        .await
        .map_err(|e| format!("Failed to fetch gRPC model info: {}", e))?;

    let mut labels = model_info.to_labels();

    // Fetch server info for KV transfer config (PD disaggregation)
    match client.get_server_info().await {
        Ok(server_info) => {
            if let Some(kv_connector) = server_info.kv_connector() {
                debug!("Discovered kv_connector: {}", kv_connector);
                labels.insert("kv_connector".to_string(), kv_connector);
            }
            if let Some(kv_role) = server_info.kv_role() {
                debug!("Discovered kv_role: {}", kv_role);
                labels.insert("kv_role".to_string(), kv_role);
            }
        }
        Err(e) => {
            // Server info is optional - log warning but don't fail
            warn!(
                "Failed to fetch gRPC server info (KV config may not be available): {}",
                e
            );
        }
    }

    Ok((labels, runtime_type.to_string()))
}

/// Step 2a: Discover metadata from worker.
pub struct DiscoverMetadataStep;

#[async_trait]
impl StepExecutor<LocalWorkerWorkflowData> for DiscoverMetadataStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<LocalWorkerWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let config = &context.data.config;
        let connection_mode =
            context.data.connection_mode.as_ref().ok_or_else(|| {
                WorkflowError::ContextValueNotFound("connection_mode".to_string())
            })?;

        debug!(
            "Discovering metadata for {} ({:?})",
            config.url, connection_mode
        );

        let (discovered_labels, detected_runtime) = match connection_mode {
            ConnectionMode::Http => {
                let mut labels = HashMap::new();

                // Fetch from /server_info for server-related metadata
                if let Ok(server_info) =
                    get_server_info(&config.url, config.api_key.as_deref()).await
                {
                    if let Some(model_path) = server_info.model_path.filter(|s| !s.is_empty()) {
                        labels.insert("model_path".to_string(), model_path);
                    }
                    if let Some(served_model_name) =
                        server_info.served_model_name.filter(|s| !s.is_empty())
                    {
                        labels.insert("served_model_name".to_string(), served_model_name);
                    }
                    if let Some(tp_size) = server_info.tp_size {
                        labels.insert("tp_size".to_string(), tp_size.to_string());
                    }
                    if let Some(dp_size) = server_info.dp_size {
                        labels.insert("dp_size".to_string(), dp_size.to_string());
                    }
                    if let Some(load_balance_method) = server_info.load_balance_method {
                        labels.insert("load_balance_method".to_string(), load_balance_method);
                    }
                    if let Some(disaggregation_mode) = server_info.disaggregation_mode {
                        labels.insert("disaggregation_mode".to_string(), disaggregation_mode);
                    }
                }

                // Fetch from /model_info for model-related metadata
                if let Ok(model_info) = get_model_info(&config.url, config.api_key.as_deref()).await
                {
                    if let Some(model_type) = model_info.model_type.filter(|s| !s.is_empty()) {
                        labels.insert("model_type".to_string(), model_type);
                    }
                    if let Some(architectures) = model_info.architectures.filter(|a| !a.is_empty())
                    {
                        if let Ok(json_str) = serde_json::to_string(&architectures) {
                            labels.insert("architectures".to_string(), json_str);
                        }
                    }
                }

                Ok((labels, None))
            }
            ConnectionMode::Grpc { .. } => {
                // Use pre-detected runtime type from connection detection step,
                // falling back to config.runtime if not available
                let runtime_type = context
                    .data
                    .detected_runtime_type
                    .as_deref()
                    .or(config.runtime.as_deref())
                    .unwrap_or("sglang"); // Fallback to sglang if somehow not detected
                debug!(
                    "Using runtime type '{}' for gRPC metadata fetch",
                    runtime_type
                );
                fetch_grpc_metadata(&config.url, runtime_type)
                    .await
                    .map(|(labels, runtime)| (labels, Some(runtime)))
            }
        }
        .unwrap_or_else(|e| {
            warn!("Failed to fetch metadata for {}: {}", config.url, e);
            (HashMap::new(), None)
        });

        let url = config.url.clone();
        debug!(
            "Discovered {} metadata labels for {}",
            discovered_labels.len(),
            url
        );

        // Update workflow data
        context.data.discovered_labels = discovered_labels;
        if let Some(runtime) = detected_runtime {
            debug!("Detected runtime type: {}", runtime);
            context.data.detected_runtime_type = Some(runtime);
        }

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        true
    }
}
