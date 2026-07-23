//! Metadata discovery step for local workers.

use std::{collections::HashMap, time::Duration};

use async_trait::async_trait;
use once_cell::sync::Lazy;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::{
    routers::grpc::client::{flat_labels, GrpcClient},
    worker::{
        sampling_defaults::SamplingDefaults, worker::metrics_authority, ConnectionMode,
        DEFAULT_SAMPLING_PARAMS_LABEL,
    },
    workflow::{
        data::{WorkerKind, WorkerWorkflowData},
        steps::util::{grpc_base_url, http_base_url},
    },
};

#[expect(
    clippy::expect_used,
    reason = "Lazy static initialization — reqwest::Client::build() only fails on TLS backend misconfiguration which is unrecoverable"
)]
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
    /// Per-instance concurrency cap. CLI flag `--max-running-requests` on SGLang.
    /// Already extracted by the SGLang gRPC label pipeline; surfacing it here
    /// closes the HTTP-only path so capacity-aware consumers (e.g. WorkerCapacity)
    /// see the same label regardless of transport.
    pub max_running_requests: Option<usize>,
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
    let url = format!("{base_url}/{endpoint}");
    let mut req = HTTP_CLIENT.get(&url);
    if let Some(key) = api_key {
        req = req.bearer_auth(key);
    }

    let response = req
        .send()
        .await
        .map_err(|e| format!("Failed to connect to {url}: {e}"))?;

    if response.status() == reqwest::StatusCode::NOT_FOUND {
        // Fallback to deprecated /get_<endpoint> prefix
        warn!("'/{endpoint}' returned 404, falling back to deprecated '/get_{endpoint}'");
        let old_url = format!("{base_url}/get_{endpoint}");
        let mut req = HTTP_CLIENT.get(&old_url);
        if let Some(key) = api_key {
            req = req.bearer_auth(key);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| format!("Failed to connect to {old_url}: {e}"))?;
        if !resp.status().is_success() {
            return Err(format!("status {} from {}", resp.status(), old_url));
        }
        return resp
            .json::<T>()
            .await
            .map_err(|e| format!("Failed to parse {old_url}: {e}"));
    }

    if !response.status().is_success() {
        return Err(format!("status {} from {}", response.status(), url));
    }

    response
        .json::<T>()
        .await
        .map_err(|e| format!("Failed to parse {url}: {e}"))
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
        .map_err(|e| format!("Failed to connect to {url}: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("status {} from {}", resp.status(), url));
    }
    resp.json::<T>()
        .await
        .map_err(|e| format!("Failed to parse {url}: {e}"))
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

    // /v1/models gives us model identity and max_model_len when a compatible
    // local frontend does not expose the SGLang-specific metadata endpoints.
    if let Ok(models) = http_get_json::<ModelsResponse>(&format!("{base}/v1/models"), api_key).await
    {
        if let Some(m) = models.data.first() {
            if let Some(id) = m.id.as_ref().filter(|id| !id.is_empty()) {
                labels
                    .entry("served_model_name".to_string())
                    .or_insert_with(|| id.clone());
            }
            if let Some(root) = m.root.as_ref().filter(|root| !root.is_empty()) {
                labels
                    .entry("model_path".to_string())
                    .or_insert_with(|| root.clone());
            }
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
    if let Ok(models) = http_get_json::<ModelsResponse>(&format!("{base}/v1/models"), api_key).await
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
    if let Ok(v) = http_get_json::<VersionResponse>(&format!("{base}/version"), api_key).await {
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
        .map_err(|e| format!("Failed to connect to gRPC: {e}"))?;

    let mut labels = client
        .get_model_info()
        .await
        .map_err(|e| format!("Failed to fetch gRPC model info: {e}"))?
        .to_labels();

    match client.get_server_info().await {
        Ok(info) => labels.extend(info.to_labels()),
        Err(e) => warn!("Failed to fetch gRPC server info: {}", e),
    }

    normalize_grpc_keys(&mut labels);
    derive_grpc_metrics_url(&mut labels, &grpc_url);
    Ok((labels, runtime_type.to_string()))
}

/// Resolve the engine `/metrics` scrape endpoint for a gRPC worker into a
/// canonical `metrics_url` label, consuming the transient `server_args` keys it
/// derives from.
///
/// Precedence:
/// 1. explicit `metrics_url` — but only when its host matches the worker's gRPC
///    host (see below),
/// 2. `prometheus_port` (+ gRPC host),
/// 3. derived `http://{host}:{port}/metrics` — only when an `enable_metrics`
///    flag is truthy, so a dark port is never advertised.
///
/// The host always comes from the gRPC worker URL, never from the `server_args`
/// `host` (which is the bind address — often `0.0.0.0`/`::`/empty — and not a
/// routable scrape target). This keeps the scrape and its bearer token on the
/// worker's own host. The `host` key is consumed but not used.
///
/// An explicit `metrics_url` is backend-advertised and therefore untrusted; we
/// accept it only when its host matches the worker's gRPC host so a backend
/// cannot redirect the scrape (and any attached credentials) to an arbitrary
/// origin. Scheme and port are not constrained — the metrics port differs from
/// the gRPC port by design. A mismatching `metrics_url` is dropped and we fall
/// back to the derived endpoint (or none).
fn derive_grpc_metrics_url(labels: &mut HashMap<String, String>, grpc_url: &str) {
    let enable_metrics = labels.remove("enable_metrics");
    let prometheus_port = labels.remove("prometheus_port").filter(|s| !s.is_empty());
    labels.remove("host");
    let port = labels.remove("port").filter(|s| !s.is_empty());

    let worker_host = grpc_host(grpc_url);
    let explicit = labels
        .get("metrics_url")
        .filter(|s| !s.is_empty())
        .filter(|url| url_host(url).as_deref() == Some(worker_host.as_str()))
        .cloned();

    let resolved = explicit
        .or_else(|| {
            prometheus_port
                .map(|p| format!("http://{}/metrics", metrics_authority(&worker_host, &p)))
        })
        .or_else(|| {
            (is_truthy(enable_metrics.as_deref()) && port.is_some()).then(|| {
                let p = port.unwrap_or_default();
                format!("http://{}/metrics", metrics_authority(&worker_host, &p))
            })
        });

    match resolved {
        Some(url) => {
            labels.insert("metrics_url".to_string(), url);
        }
        None => {
            labels.remove("metrics_url");
        }
    }
}

/// Truthy flag values as emitted by `server_args` (bools become `"true"`, ints
/// like a port number become e.g. `"1"`).
fn is_truthy(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("true" | "1" | "yes" | "on")
    )
}

/// Host portion of a gRPC URL (`grpc://host:port` → `host`,
/// `grpc://[::1]:port` → `::1`), falling back to the stripped input.
fn grpc_host(grpc_url: &str) -> String {
    let stripped = grpc_url
        .strip_prefix("grpc://")
        .or_else(|| grpc_url.strip_prefix("grpcs://"))
        .unwrap_or(grpc_url);
    // `rsplit_once` keeps bracketed IPv6 literals intact (`[::1]:port`); a
    // plain `split_once(':')` would cut at the first colon of the address.
    let host = stripped.rsplit_once(':').map_or(stripped, |(h, _)| h);
    host.strip_prefix('[')
        .and_then(|h| h.strip_suffix(']'))
        .unwrap_or(host)
        .to_string()
}

/// Host of an arbitrary URL in unbracketed form, for comparison against
/// [`grpc_host`] (which is also unbracketed). Returns `None` if the URL has no
/// parsable host, so an unparsable explicit `metrics_url` is rejected rather
/// than silently trusted.
fn url_host(url: &str) -> Option<String> {
    url::Url::parse(url)
        .ok()
        .and_then(|u| u.host_str().map(str::to_string))
        .map(|h| {
            h.strip_prefix('[')
                .and_then(|s| s.strip_suffix(']'))
                .unwrap_or(&h)
                .to_string()
        })
}

/// Rename gRPC-specific keys to canonical names and strip transient state.
fn normalize_grpc_keys(labels: &mut HashMap<String, String>) {
    for &(from, to) in &[
        ("tensor_parallel_size", "tp_size"),
        ("pipeline_parallel_size", "pp_size"),
        ("context_parallel_size", "cp_size"),
        ("data_parallel_size", "dp_size"),
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
    normalize_default_sampling_params_label(labels);
}

fn normalize_default_sampling_params_label(labels: &mut HashMap<String, String>) {
    let Some(raw) = labels.get(DEFAULT_SAMPLING_PARAMS_LABEL).cloned() else {
        return;
    };

    match SamplingDefaults::canonical_json_from_str(&raw) {
        Ok(Some(canonical)) => {
            labels.insert(DEFAULT_SAMPLING_PARAMS_LABEL.to_string(), canonical);
        }
        Ok(None) => {
            labels.remove(DEFAULT_SAMPLING_PARAMS_LABEL);
        }
        Err(e) => {
            warn!(
                error = %e,
                "Ignoring invalid default sampling params label"
            );
            labels.remove(DEFAULT_SAMPLING_PARAMS_LABEL);
        }
    }
}

// ---------------------------------------------------------------------------
// Step executor
// ---------------------------------------------------------------------------

pub struct DiscoverMetadataStep;

#[async_trait]
impl StepExecutor<WorkerWorkflowData> for DiscoverMetadataStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        if context.data.worker_kind != Some(WorkerKind::Local) {
            return Ok(StepResult::Skip);
        }

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

    #[expect(clippy::print_stderr)]
    fn dump_labels(title: &str, labels: &HashMap<String, String>) {
        eprintln!("\n=== {title} ({} labels) ===", labels.len());
        let mut keys: Vec<_> = labels.keys().collect();
        keys.sort();
        for key in keys {
            eprintln!("  {key}: {}", labels[key]);
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

    #[test]
    fn test_sglang_server_info_surfaces_max_running_requests_label() {
        // Subset of an actual SGLang /server_info response. The full payload has
        // many more fields; `serde(deny_unknown_fields)` is off by default so
        // they're silently ignored.
        let body = serde_json::json!({
            "model_path": "Qwen/Qwen3-8B",
            "tp_size": 1,
            "dp_size": 1,
            "max_running_requests": 256,
            "context_length": 32768,
        });
        let info: ServerInfo = serde_json::from_value(body).expect("deserialize ServerInfo");
        assert_eq!(info.max_running_requests, Some(256));

        let labels = flat_labels(&info);
        assert_eq!(
            labels.get("max_running_requests").map(String::as_str),
            Some("256")
        );
    }

    fn labels_of(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect()
    }

    #[test]
    fn derive_grpc_metrics_url_prefers_explicit() {
        // Explicit URL on the worker's own host (only the port differs) is
        // trusted and wins over the derived candidates.
        let mut labels = labels_of(&[
            ("metrics_url", "http://host:9000/metrics"),
            ("prometheus_port", "9100"),
            ("enable_metrics", "true"),
        ]);
        derive_grpc_metrics_url(&mut labels, "grpc://host:30001");
        assert_eq!(
            labels.get("metrics_url").map(String::as_str),
            Some("http://host:9000/metrics")
        );
        // Transient derivation keys are consumed.
        assert!(!labels.contains_key("prometheus_port"));
        assert!(!labels.contains_key("enable_metrics"));
    }

    #[test]
    fn derive_grpc_metrics_url_rejects_explicit_on_foreign_host() {
        // A backend-advertised metrics_url pointing at a different host is an
        // SSRF vector; it must be dropped and the derived endpoint used instead.
        let mut labels = labels_of(&[
            ("metrics_url", "http://evil.example.com:9000/metrics"),
            ("prometheus_port", "9100"),
        ]);
        derive_grpc_metrics_url(&mut labels, "grpc://host:30001");
        assert_eq!(
            labels.get("metrics_url").map(String::as_str),
            Some("http://host:9100/metrics")
        );
    }

    #[test]
    fn derive_grpc_metrics_url_drops_foreign_explicit_without_fallback() {
        // No derivable fallback: a foreign explicit URL is removed entirely
        // rather than scraped.
        let mut labels = labels_of(&[("metrics_url", "http://evil.example.com:9000/metrics")]);
        derive_grpc_metrics_url(&mut labels, "grpc://host:30001");
        assert!(!labels.contains_key("metrics_url"));
    }

    #[test]
    fn derive_grpc_metrics_url_from_prometheus_port() {
        let mut labels = labels_of(&[("prometheus_port", "9100")]);
        derive_grpc_metrics_url(&mut labels, "grpc://10.0.0.5:30001");
        assert_eq!(
            labels.get("metrics_url").map(String::as_str),
            Some("http://10.0.0.5:9100/metrics")
        );
    }

    #[test]
    fn derive_grpc_metrics_url_derives_when_enabled() {
        // The bind `host` (often a wildcard) is dropped; the scrape host comes
        // from the worker's gRPC URL so the endpoint is routable.
        let mut labels = labels_of(&[
            ("enable_metrics", "true"),
            ("host", "0.0.0.0"),
            ("port", "30000"),
        ]);
        derive_grpc_metrics_url(&mut labels, "grpc://node-7:30001");
        assert_eq!(
            labels.get("metrics_url").map(String::as_str),
            Some("http://node-7:30000/metrics")
        );
    }

    #[test]
    fn derive_grpc_metrics_url_ipv6_host_from_grpc_url() {
        // An IPv6 gRPC host must stay bracketed in the scrape URL authority
        // (`http://[::1]:9100/metrics`), otherwise the target is an invalid URI.
        let mut labels = labels_of(&[("prometheus_port", "9100")]);
        derive_grpc_metrics_url(&mut labels, "grpc://[::1]:30001");
        assert_eq!(
            labels.get("metrics_url").map(String::as_str),
            Some("http://[::1]:9100/metrics")
        );

        // Same for the enable_metrics + port branch.
        let mut labels = labels_of(&[("enable_metrics", "true"), ("port", "30000")]);
        derive_grpc_metrics_url(&mut labels, "grpc://[2001:db8::1]:30001");
        assert_eq!(
            labels.get("metrics_url").map(String::as_str),
            Some("http://[2001:db8::1]:30000/metrics")
        );
    }

    #[test]
    fn derive_grpc_metrics_url_accepts_explicit_ipv6_same_host() {
        // Explicit metrics_url on the same IPv6 host (brackets vs. grpc_host's
        // unbracketed form must still compare equal) is trusted.
        let mut labels = labels_of(&[("metrics_url", "http://[::1]:9100/metrics")]);
        derive_grpc_metrics_url(&mut labels, "grpc://[::1]:30001");
        assert_eq!(
            labels.get("metrics_url").map(String::as_str),
            Some("http://[::1]:9100/metrics")
        );
    }

    #[test]
    fn derive_grpc_metrics_url_uses_grpc_host_when_host_absent() {
        let mut labels = labels_of(&[("enable_metrics", "true"), ("port", "30000")]);
        derive_grpc_metrics_url(&mut labels, "grpc://node-7:30001");
        assert_eq!(
            labels.get("metrics_url").map(String::as_str),
            Some("http://node-7:30000/metrics")
        );
    }

    #[test]
    fn derive_grpc_metrics_url_skips_when_disabled() {
        // enable_metrics absent/falsy and no explicit port keys: never advertise
        // a dark scrape endpoint.
        let mut labels = labels_of(&[("host", "0.0.0.0"), ("port", "30000")]);
        derive_grpc_metrics_url(&mut labels, "grpc://host:30001");
        assert!(!labels.contains_key("metrics_url"));

        let mut labels = labels_of(&[
            ("enable_metrics", "false"),
            ("host", "0.0.0.0"),
            ("port", "30000"),
        ]);
        derive_grpc_metrics_url(&mut labels, "grpc://host:30001");
        assert!(!labels.contains_key("metrics_url"));
    }

    #[test]
    fn test_sglang_server_info_max_running_requests_optional() {
        // Older SGLang versions or special configurations may omit the field.
        let body = serde_json::json!({
            "model_path": "Qwen/Qwen3-8B",
            "tp_size": 1,
        });
        let info: ServerInfo = serde_json::from_value(body).expect("deserialize ServerInfo");
        assert_eq!(info.max_running_requests, None);

        let labels = flat_labels(&info);
        assert!(!labels.contains_key("max_running_requests"));
    }

    #[tokio::test]
    async fn test_sglang_http_metadata_uses_models_identity() {
        use axum::{routing::get, Json, Router};
        use serde_json::json;
        use tokio::net::TcpListener;

        async fn models() -> Json<serde_json::Value> {
            Json(json!({
                "data": [{
                    "id": "test-model",
                    "max_model_len": 4096,
                    "object": "model",
                    "owned_by": "nvidia",
                    "root": "test-root"
                }],
                "object": "list"
            }))
        }

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        #[expect(
            clippy::disallowed_methods,
            reason = "test-only mock /v1/models server; handle is aborted at test end"
        )]
        let server = tokio::spawn(async move {
            axum::serve(listener, Router::new().route("/v1/models", get(models)))
                .await
                .unwrap();
        });

        let labels = fetch_sglang_http_metadata(&format!("http://{addr}"), None).await;
        server.abort();

        assert_eq!(
            labels.get("served_model_name").map(String::as_str),
            Some("test-model")
        );
        assert_eq!(
            labels.get("model_path").map(String::as_str),
            Some("test-root")
        );
        assert_eq!(
            labels.get("max_model_len").map(String::as_str),
            Some("4096")
        );
    }
}
