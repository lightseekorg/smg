//! Metadata discovery step for local workers.

use std::{collections::HashMap, time::Duration};

use async_trait::async_trait;
use once_cell::sync::Lazy;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use super::{grpc_base_url, http_base_url};
use crate::{
    core::{steps::workflow_data::LocalWorkerWorkflowData, ConnectionMode},
    routers::grpc::client::{flat_labels, GrpcClient},
};

static HTTP_CLIENT: Lazy<Client> = Lazy::new(|| {
    Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .expect("Failed to create HTTP client")
});

// ---------------------------------------------------------------------------
// HTTP response structs (sglang /server_info, /model_info; vllm /v1/models)
// ---------------------------------------------------------------------------

/// SGLang `/server_info` response — curated subset of the full response (~800 fields).
/// Uses `deny_unknown_fields = false` (the default) so extra fields are silently ignored.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ServerInfo {
    #[serde(alias = "model")]
    pub model_id: Option<String>,
    pub model_path: Option<String>,
    pub served_model_name: Option<String>,
    pub tp_size: Option<usize>,
    pub dp_size: Option<usize>,
    pub pp_size: Option<usize>,
    pub load_balance_method: Option<String>,
    pub disaggregation_mode: Option<String>,
    pub version: Option<String>,
    pub is_embedding: Option<bool>,
    pub context_length: Option<usize>,
    pub max_total_tokens: Option<usize>,
    pub weight_version: Option<String>,
}

/// SGLang `/model_info` response.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelInfo {
    pub model_path: Option<String>,
    pub tokenizer_path: Option<String>,
    pub is_generation: Option<bool>,
    pub has_image_understanding: Option<bool>,
    pub has_audio_understanding: Option<bool>,
    pub model_type: Option<String>,
    pub architectures: Option<Vec<String>>,
}

/// Single entry from `/v1/models` (shared by sglang and vllm).
#[derive(Debug, Clone, Deserialize)]
pub(super) struct ModelsResponseEntry {
    pub owned_by: Option<String>,
    pub id: Option<String>,
    pub root: Option<String>,
    pub max_model_len: Option<usize>,
}

/// `/v1/models` response wrapper.
#[derive(Debug, Clone, Deserialize)]
pub(super) struct ModelsResponse {
    pub data: Vec<ModelsResponseEntry>,
}

/// vLLM `/version` response.
#[derive(Debug, Deserialize)]
struct VersionResponse {
    version: String,
}

// ---------------------------------------------------------------------------
// HTTP fetchers
// ---------------------------------------------------------------------------

/// GET JSON with optional bearer auth, with 404 fallback to `/get_<endpoint>`.
async fn get_json_with_fallback<T: serde::de::DeserializeOwned>(
    base_url: &str,
    endpoint: &str,
    api_key: Option<&str>,
) -> Result<T, String> {
    let url = format!("{}/{}", base_url, endpoint);
    let mut req = HTTP_CLIENT.get(&url);
    if let Some(key) = api_key {
        req = req.bearer_auth(key);
    }

    let response = req
        .send()
        .await
        .map_err(|e| format!("Failed to connect to {}: {}", url, e))?;

    if response.status() == reqwest::StatusCode::NOT_FOUND {
        // Fallback to deprecated /get_<endpoint> prefix
        warn!("'/{endpoint}' returned 404, falling back to deprecated '/get_{endpoint}'");
        let old_url = format!("{}/get_{}", base_url, endpoint);
        let mut req = HTTP_CLIENT.get(&old_url);
        if let Some(key) = api_key {
            req = req.bearer_auth(key);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| format!("Failed to connect to {}: {}", old_url, e))?;
        if !resp.status().is_success() {
            return Err(format!("status {} from {}", resp.status(), old_url));
        }
        return resp
            .json::<T>()
            .await
            .map_err(|e| format!("Failed to parse {}: {}", old_url, e));
    }

    if !response.status().is_success() {
        return Err(format!("status {} from {}", response.status(), url));
    }

    response
        .json::<T>()
        .await
        .map_err(|e| format!("Failed to parse {}: {}", url, e))
}

/// GET JSON (no fallback).
async fn http_get_json<T: serde::de::DeserializeOwned>(
    url: &str,
    api_key: Option<&str>,
) -> Result<T, String> {
    let mut req = HTTP_CLIENT.get(url);
    if let Some(key) = api_key {
        req = req.bearer_auth(key);
    }
    let resp = req
        .send()
        .await
        .map_err(|e| format!("Failed to connect to {}: {}", url, e))?;
    if !resp.status().is_success() {
        return Err(format!("status {} from {}", resp.status(), url));
    }
    resp.json::<T>()
        .await
        .map_err(|e| format!("Failed to parse {}: {}", url, e))
}

pub async fn get_server_info(url: &str, api_key: Option<&str>) -> Result<ServerInfo, String> {
    get_json_with_fallback(&http_base_url(url), "server_info", api_key).await
}

pub async fn get_model_info(url: &str, api_key: Option<&str>) -> Result<ModelInfo, String> {
    get_json_with_fallback(&http_base_url(url), "model_info", api_key).await
}

// ---------------------------------------------------------------------------
// Per-backend metadata fetchers
// ---------------------------------------------------------------------------

async fn fetch_sglang_http_metadata(url: &str, api_key: Option<&str>) -> HashMap<String, String> {
    let base = http_base_url(url);
    let mut labels = HashMap::new();

    if let Ok(info) = get_server_info(&base, api_key).await {
        labels.extend(flat_labels(&info));
    }
    if let Ok(info) = get_model_info(&base, api_key).await {
        labels.extend(flat_labels(&info));
    }

    // /v1/models gives us max_model_len (fills context_length when /server_info returns null)
    if let Ok(models) =
        http_get_json::<ModelsResponse>(&format!("{}/v1/models", base), api_key).await
    {
        if let Some(m) = models.data.first() {
            if let Some(len) = m.max_model_len.filter(|&n| n > 0) {
                labels
                    .entry("max_model_len".to_string())
                    .or_insert_with(|| len.to_string());
            }
        }
    }

    labels
}

async fn fetch_vllm_http_metadata(url: &str, api_key: Option<&str>) -> HashMap<String, String> {
    let base = http_base_url(url);
    let mut labels = HashMap::new();

    // /v1/models — vLLM uses `root` as model_path, `id` as served_model_name
    if let Ok(models) =
        http_get_json::<ModelsResponse>(&format!("{}/v1/models", base), api_key).await
    {
        if let Some(m) = models.data.first() {
            if let Some(ref root) = m.root {
                labels.insert("model_path".to_string(), root.clone());
            }
            if let Some(ref id) = m.id {
                labels.insert("served_model_name".to_string(), id.clone());
            }
            if let Some(len) = m.max_model_len.filter(|&n| n > 0) {
                labels.insert("max_model_len".to_string(), len.to_string());
            }
        }
    }

    // /version
    if let Ok(v) = http_get_json::<VersionResponse>(&format!("{}/version", base), api_key).await {
        if !v.version.is_empty() {
            labels.insert("version".to_string(), v.version);
        }
    }

    labels
}

async fn fetch_grpc_metadata(
    url: &str,
    runtime_type: &str,
) -> Result<(HashMap<String, String>, String), String> {
    let grpc_url = grpc_base_url(url);

    let client = GrpcClient::connect(&grpc_url, runtime_type)
        .await
        .map_err(|e| format!("Failed to connect to gRPC: {}", e))?;

    let mut labels = client
        .get_model_info()
        .await
        .map_err(|e| format!("Failed to fetch gRPC model info: {}", e))?
        .to_labels();

    match client.get_server_info().await {
        Ok(info) => labels.extend(info.to_labels()),
        Err(e) => warn!("Failed to fetch gRPC server info: {}", e),
    }

    normalize_grpc_keys(&mut labels);
    Ok((labels, runtime_type.to_string()))
}

/// Rename gRPC-specific keys to canonical names and strip transient state.
fn normalize_grpc_keys(labels: &mut HashMap<String, String>) {
    for &(from, to) in &[
        ("tensor_parallel_size", "tp_size"),
        ("pipeline_parallel_size", "pp_size"),
        ("context_parallel_size", "cp_size"),
    ] {
        if let Some(val) = labels.remove(from) {
            labels.entry(to.to_string()).or_insert(val);
        }
    }
    for key in [
        "active_requests",
        "is_paused",
        "last_receive_timestamp",
        "uptime_seconds",
        "server_type",
    ] {
        labels.remove(key);
    }
}

// ---------------------------------------------------------------------------
// Step executor
// ---------------------------------------------------------------------------

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
                let runtime = context
                    .data
                    .detected_runtime_type
                    .as_deref()
                    .unwrap_or_else(|| {
                        warn!(
                            "No detected_runtime_type for {}, defaulting to sglang",
                            config.url
                        );
                        "sglang"
                    });
                let labels = match runtime {
                    "vllm" => {
                        fetch_vllm_http_metadata(&config.url, config.api_key.as_deref()).await
                    }
                    _ => fetch_sglang_http_metadata(&config.url, config.api_key.as_deref()).await,
                };
                Ok((labels, None))
            }
            ConnectionMode::Grpc => {
                let config_runtime = config.runtime_type.to_string();
                let runtime_type = context
                    .data
                    .detected_runtime_type
                    .as_deref()
                    .unwrap_or(&config_runtime);
                fetch_grpc_metadata(&config.url, runtime_type)
                    .await
                    .map(|(labels, rt)| (labels, Some(rt)))
            }
        }
        .unwrap_or_else(|e| {
            warn!("Failed to fetch metadata for {}: {}", config.url, e);
            (HashMap::new(), None)
        });

        debug!(
            "Discovered {} labels for {}",
            discovered_labels.len(),
            config.url
        );
        context.data.discovered_labels = discovered_labels;
        if let Some(runtime) = detected_runtime {
            context.data.detected_runtime_type = Some(runtime);
        }

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dump_labels(title: &str, labels: &HashMap<String, String>) {
        println!("\n=== {} ({} labels) ===", title, labels.len());
        let mut keys: Vec<_> = labels.keys().collect();
        keys.sort();
        for key in keys {
            println!("  {}: {}", key, labels[key]);
        }
    }

    #[tokio::test]
    #[ignore]
    async fn test_sglang_http_metadata() {
        let labels = fetch_sglang_http_metadata("http://0.0.0.0:30000", None).await;
        dump_labels("SGLang HTTP combined", &labels);
        assert!(labels.contains_key("model_path"));
        assert!(labels.contains_key("tokenizer_path"));
    }

    #[tokio::test]
    #[ignore]
    async fn test_vllm_http_metadata() {
        let labels = fetch_vllm_http_metadata("http://0.0.0.0:20000", None).await;
        dump_labels("vLLM HTTP", &labels);
        assert!(labels.contains_key("model_path"));
        assert!(labels.contains_key("version"));
    }

    #[tokio::test]
    #[ignore]
    async fn test_sglang_grpc_metadata() {
        let (labels, _) = fetch_grpc_metadata("grpc://0.0.0.0:30001", "sglang")
            .await
            .expect("grpc metadata");
        dump_labels("SGLang gRPC", &labels);
        assert!(labels.contains_key("model_path"));
    }

    #[tokio::test]
    #[ignore]
    async fn test_vllm_grpc_metadata() {
        let (labels, _) = fetch_grpc_metadata("grpc://0.0.0.0:20001", "vllm")
            .await
            .expect("grpc metadata");
        dump_labels("vLLM gRPC", &labels);
        assert!(!labels.is_empty());
    }
}
