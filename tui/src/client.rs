use anyhow::Result;
use openai_protocol::{
    messages::ListModelsResponse,
    worker::{WorkerLoadResponse, WorkerSpec, WorkerUpdateRequest},
};
use serde::Deserialize;

/// HTTP client for the SMG gateway REST API.
#[derive(Debug, Clone)]
pub struct SmgClient {
    http: reqwest::Client,
    stream_http: reqwest::Client,
    gateway_url: String,
    metrics_url: String,
    api_key: Option<String>,
}

// ── Local response types matching the actual wire format ──
// The server uses custom IntoResponse impls that produce JSON different from
// the protocol structs' Serialize output, so we define our own Deserialize types.

/// Mirrors the JSON produced by `ListWorkersResult::into_response` in model_gateway.
#[derive(Debug, Clone, Deserialize)]
pub struct WorkersResponse {
    pub workers: Vec<WorkerInfo>,
    pub total: usize,
    #[serde(default)]
    pub stats: WorkerStatsWire,
}

/// Worker info as it appears on the wire (WorkerSpec fields are flattened).
#[derive(Debug, Clone, Deserialize)]
pub struct WorkerInfo {
    pub id: String,
    pub url: String,
    #[serde(default)]
    pub worker_type: String,
    #[serde(default)]
    pub connection_mode: String,
    #[serde(default)]
    pub runtime_type: String,
    #[serde(default)]
    pub models: Vec<ModelRef>,
    #[serde(default)]
    pub is_healthy: bool,
    #[serde(default)]
    pub load: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelRef {
    pub id: String,
    #[serde(default)]
    pub model_type: Vec<String>,
}

/// Stats block: `{ prefill_count, decode_count, regular_count }`.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct WorkerStatsWire {
    #[serde(default)]
    pub prefill_count: usize,
    #[serde(default)]
    pub decode_count: usize,
    #[serde(default)]
    pub regular_count: usize,
}

/// Mirrors the JSON from `WorkerLoadsResult::into_response`:
/// `{ "workers": [{"worker": "...", "load": N}] }`
#[derive(Debug, Clone, Deserialize)]
pub struct LoadsResponse {
    pub workers: Vec<WorkerLoad>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct WorkerLoad {
    pub worker: String,
    #[serde(default)]
    pub worker_type: Option<String>,
    pub load: isize,
    #[serde(default)]
    pub details: Option<WorkerLoadResponse>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ClusterStatusResponse {
    pub node_name: Option<String>,
    pub cluster_size: Option<usize>,
    pub stores: Option<Vec<StoreStatus>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StoreStatus {
    pub name: String,
    pub healthy: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MeshHealthResponse {
    pub status: String,
    pub node_count: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RateLimitStats {
    pub limit: Option<u64>,
    pub current: Option<u64>,
    pub remaining: Option<u64>,
}

impl SmgClient {
    pub fn new(gateway_url: String, metrics_url: String, api_key: Option<String>) -> Self {
        // reqwest::Client::builder().build() only fails on TLS backend init failure,
        // which is an unrecoverable startup error.
        #[expect(clippy::expect_used)]
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .expect("failed to build HTTP client");

        #[expect(clippy::expect_used)]
        let stream_http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .expect("failed to build streaming HTTP client");

        Self {
            http,
            stream_http,
            gateway_url,
            metrics_url,
            api_key,
        }
    }

    fn request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}{}", self.gateway_url, path);
        let mut req = self.http.request(method, &url);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }
        req
    }

    pub async fn check_health(&self) -> Result<()> {
        self.request(reqwest::Method::GET, "/readiness")
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }

    /// Check if the gateway is alive (accepting connections), regardless of worker readiness.
    pub async fn check_alive(&self) -> Result<()> {
        // Any response (even 503) means the server is up
        self.request(reqwest::Method::GET, "/readiness")
            .send()
            .await?;
        Ok(())
    }

    pub async fn list_workers(&self) -> Result<WorkersResponse> {
        Ok(self
            .request(reqwest::Method::GET, "/workers")
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?)
    }

    pub async fn add_worker(&self, spec: &WorkerSpec) -> Result<serde_json::Value> {
        // Build JSON manually because WorkerSpec.api_key has skip_serializing
        let mut body = serde_json::to_value(spec)?;
        if let Some(ref key) = spec.api_key {
            body["api_key"] = serde_json::Value::String(key.clone());
        }
        Ok(self
            .request(reqwest::Method::POST, "/workers")
            .json(&body)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?)
    }

    pub async fn delete_worker(&self, id: &str) -> Result<serde_json::Value> {
        Ok(self
            .request(reqwest::Method::DELETE, &format!("/workers/{id}"))
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?)
    }

    pub async fn get_loads(&self) -> Result<LoadsResponse> {
        Ok(self
            .request(reqwest::Method::GET, "/get_loads")
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?)
    }

    pub async fn get_cluster_status(&self) -> Result<ClusterStatusResponse> {
        Ok(self
            .request(reqwest::Method::GET, "/ha/status")
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?)
    }

    pub async fn get_mesh_health(&self) -> Result<MeshHealthResponse> {
        Ok(self
            .request(reqwest::Method::GET, "/ha/health")
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?)
    }

    pub async fn get_rate_limit_stats(&self) -> Result<RateLimitStats> {
        Ok(self
            .request(reqwest::Method::GET, "/ha/rate-limit/stats")
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?)
    }

    pub async fn list_models(&self) -> Result<ListModelsResponse> {
        Ok(self
            .request(reqwest::Method::GET, "/v1/models")
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?)
    }

    pub async fn update_worker(
        &self,
        id: &str,
        update: &WorkerUpdateRequest,
    ) -> Result<serde_json::Value> {
        let resp = self
            .request(reqwest::Method::PATCH, &format!("/workers/{id}"))
            .json(update)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        Ok(resp)
    }

    /// Send a streaming POST request, returning the raw response for SSE processing.
    /// Uses a longer timeout and passes API key as bearer token.
    pub async fn stream_request(
        &self,
        path: &str,
        body: &serde_json::Value,
    ) -> Result<reqwest::Response> {
        let url = format!("{}{}", self.gateway_url, path);
        let mut req = self.stream_http.post(&url).json(body);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }
        Ok(req.send().await?.error_for_status()?)
    }

    /// Fetch raw Prometheus metrics text from the metrics endpoint.
    pub async fn fetch_metrics(&self) -> Result<String> {
        let url = format!("{}/metrics", self.metrics_url);
        Ok(self
            .http
            .get(&url)
            .send()
            .await?
            .error_for_status()?
            .text()
            .await?)
    }

    pub async fn flush_worker_cache(&self, id: &str) -> Result<serde_json::Value> {
        let resp = self
            .request(reqwest::Method::POST, &format!("/workers/{id}/flush_cache"))
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        Ok(resp)
    }
}
