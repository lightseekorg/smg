use std::{
    fmt,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, LazyLock, RwLock as StdRwLock,
    },
    time::Duration,
};

use async_trait::async_trait;
use axum::body::Body;
// Re-export protocol types as the canonical types for the gateway
pub use openai_protocol::worker::{ConnectionMode, RuntimeType, WorkerType};
use openai_protocol::{
    model_card::ModelCard,
    model_type::{Endpoint, ModelType},
    worker::{HealthCheckConfig, ProviderType, WorkerInfo, WorkerModels, WorkerSpec},
};
use tokio::{sync::OnceCell, time};

use super::{CircuitBreaker, WorkerError, WorkerResult, UNKNOWN_MODEL_ID};
use crate::{
    observability::metrics::{metrics_labels, Metrics},
    routers::grpc::client::GrpcClient,
};

/// Default HTTP client timeout for worker requests (in seconds)
pub const DEFAULT_WORKER_HTTP_TIMEOUT_SECS: u64 = 30;

/// Default bootstrap port for PD disaggregation (used by SGLang and vLLM Mooncake)
pub const DEFAULT_BOOTSTRAP_PORT: u16 = 8998;

/// vLLM Mooncake KV connector name
pub const MOONCAKE_CONNECTOR: &str = "MooncakeConnector";

static WORKER_CLIENT: LazyLock<reqwest::Client> = LazyLock::new(|| {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(DEFAULT_WORKER_HTTP_TIMEOUT_SECS))
        .build()
        .expect("Failed to create worker HTTP client")
});

pub struct WorkerRoutingKeyLoad {
    url: String,
    active_routing_keys: dashmap::DashMap<String, usize>,
}

impl WorkerRoutingKeyLoad {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            active_routing_keys: dashmap::DashMap::new(),
        }
    }

    pub fn value(&self) -> usize {
        self.active_routing_keys.len()
    }

    pub fn increment(&self, routing_key: &str) {
        *self
            .active_routing_keys
            .entry(routing_key.to_string())
            .or_insert(0) += 1;
        self.update_metrics();
    }

    pub fn decrement(&self, routing_key: &str) {
        use dashmap::mapref::entry::Entry;

        match self.active_routing_keys.entry(routing_key.to_string()) {
            Entry::Occupied(mut entry) => {
                let counter = entry.get_mut();
                if *counter > 0 {
                    *counter -= 1;
                    if *counter == 0 {
                        entry.remove();
                    }
                } else {
                    tracing::warn!(
                        worker_url = %self.url,
                        routing_key = %routing_key,
                        "Attempted to decrement routing key counter that is already at 0"
                    );
                }
            }
            Entry::Vacant(_) => {
                tracing::warn!(
                    worker_url = %self.url,
                    routing_key = %routing_key,
                    "Attempted to decrement non-existent routing key"
                );
            }
        }
        self.update_metrics();
    }

    fn update_metrics(&self) {
        Metrics::set_worker_routing_keys_active(&self.url, self.value());
    }
}

impl fmt::Debug for WorkerRoutingKeyLoad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorkerRoutingKeyLoad")
            .field("url", &self.url)
            .field("active_routing_keys", &self.value())
            .finish()
    }
}

/// Core worker abstraction that represents a backend service
#[async_trait]
pub trait Worker: Send + Sync + fmt::Debug {
    /// Get the worker's URL
    fn url(&self) -> &str;
    /// Get the worker's API key
    fn api_key(&self) -> &Option<String>;
    /// Get the worker's type (Regular, Prefill, or Decode)
    /// Returns a reference to avoid cloning on every access
    fn worker_type(&self) -> &WorkerType;

    /// Get the worker's connection mode (HTTP or gRPC)
    /// Returns a reference to avoid cloning on every access
    fn connection_mode(&self) -> &ConnectionMode;

    /// Get the bootstrap hostname for PD mode
    /// Returns cached hostname parsed from URL at construction time
    fn bootstrap_host(&self) -> &str {
        &self.metadata().spec.bootstrap_host
    }

    /// Get the bootstrap port for PD mode
    /// Returns cached port from WorkerType::Prefill
    fn bootstrap_port(&self) -> Option<u16> {
        self.metadata().spec.bootstrap_port
    }

    /// Check if the worker is currently healthy
    fn is_healthy(&self) -> bool;

    /// Set the worker's health status
    fn set_healthy(&self, healthy: bool);

    /// Perform an async health check on the worker
    async fn check_health_async(&self) -> WorkerResult<()>;

    /// Get the current load (number of active requests)
    fn load(&self) -> usize;

    /// Increment the load counter
    fn increment_load(&self);

    /// Decrement the load counter
    fn decrement_load(&self);

    /// Reset the load counter to 0 (for sync/recovery)
    fn reset_load(&self) {}

    /// Get the worker routing key load tracker
    fn worker_routing_key_load(&self) -> &WorkerRoutingKeyLoad;

    /// Get the number of processed requests
    fn processed_requests(&self) -> usize;

    /// Increment the processed requests counter
    fn increment_processed(&self);

    /// Get worker-specific metadata
    fn metadata(&self) -> &WorkerMetadata;

    /// Get the circuit breaker for this worker
    fn circuit_breaker(&self) -> &CircuitBreaker;

    /// Check if the worker is available (healthy + circuit closed/half-open)
    fn is_available(&self) -> bool {
        self.is_healthy() && self.circuit_breaker().can_execute()
    }

    /// Record the outcome of a request to this worker
    fn record_outcome(&self, success: bool) {
        self.circuit_breaker().record_outcome(success);
    }

    /// Check if this worker is DP-aware
    fn is_dp_aware(&self) -> bool {
        false
    }

    /// Get the base URL without any DP rank suffix
    fn base_url(&self) -> &str {
        self.url()
    }

    /// Get DP rank if this is a DP-aware worker
    fn dp_rank(&self) -> Option<usize> {
        None
    }

    /// Get DP size if this worker is part of a DP group
    fn dp_size(&self) -> Option<usize> {
        None
    }

    /// Transform a request for DP-aware routing
    async fn prepare_request(&self, req: serde_json::Value) -> WorkerResult<serde_json::Value> {
        Ok(req)
    }

    /// Get the actual endpoint URL for requests
    fn endpoint_url(&self, route: &str) -> String {
        format!("{}{}", self.base_url(), route)
    }

    /// Check if this worker can handle a specific request
    fn can_handle(&self, _req: &serde_json::Value) -> bool {
        true
    }

    /// Get the model ID this worker serves
    /// Checks ModelCards first, then falls back to labels
    fn model_id(&self) -> &str {
        // Check ModelCards first
        self.metadata()
            .spec
            .models
            .primary()
            .map(|m| m.id.as_str())
            .or_else(|| {
                // Fall back to labels
                self.metadata()
                    .spec
                    .labels
                    .get("model_id")
                    .map(|s| s.as_str())
            })
            .unwrap_or(UNKNOWN_MODEL_ID)
    }

    /// Get the priority of this worker (higher value = higher priority)
    fn priority(&self) -> u32 {
        self.metadata().spec.priority
    }

    /// Get the cost factor of this worker (baseline = 1.0)
    fn cost(&self) -> f32 {
        self.metadata().spec.cost
    }

    /// Get tokenizer path for a specific model.
    fn tokenizer_path(&self, model_id: &str) -> Option<&str> {
        self.metadata()
            .find_model(model_id)
            .and_then(|m| m.tokenizer_path.as_deref())
    }

    /// Get reasoning parser for a specific model.
    fn reasoning_parser(&self, model_id: &str) -> Option<&str> {
        self.metadata()
            .find_model(model_id)
            .and_then(|m| m.reasoning_parser.as_deref())
    }

    /// Get tool parser for a specific model.
    fn tool_parser(&self, model_id: &str) -> Option<&str> {
        self.metadata()
            .find_model(model_id)
            .and_then(|m| m.tool_parser.as_deref())
    }

    /// Get chat template for a specific model.
    fn chat_template(&self, model_id: &str) -> Option<&str> {
        self.metadata()
            .find_model(model_id)
            .and_then(|m| m.chat_template.as_deref())
    }

    /// Get the default provider type for this worker.
    /// `None` means native/passthrough.
    fn default_provider(&self) -> Option<&ProviderType> {
        self.metadata().spec.provider.as_ref()
    }

    /// Get provider for a specific model.
    /// Priority: ModelCard.provider > worker.default_provider
    fn provider_for_model(&self, model_id: &str) -> Option<&ProviderType> {
        self.metadata().provider_for_model(model_id)
    }

    /// Check if a model is a classifier (has id2label mapping).
    fn is_classifier(&self, model_id: &str) -> bool {
        self.metadata()
            .find_model(model_id)
            .map(|m| m.is_classifier())
            .unwrap_or(false)
    }

    /// Get the id2label mapping for a classification model.
    /// Returns None if model is not a classifier or not found.
    fn id2label(&self, model_id: &str) -> Option<&std::collections::HashMap<u32, String>> {
        self.metadata()
            .find_model(model_id)
            .filter(|m| m.is_classifier())
            .map(|m| &m.id2label)
    }

    /// Get the number of classification labels for a model.
    fn num_labels(&self, model_id: &str) -> u32 {
        self.metadata()
            .find_model(model_id)
            .map(|m| m.num_labels)
            .unwrap_or(0)
    }

    /// Get label for a class index from a classification model.
    /// Returns generic label (LABEL_N) if model not found or index not in mapping.
    fn get_label(&self, model_id: &str, class_idx: u32) -> String {
        self.metadata()
            .find_model(model_id)
            .map(|m| m.get_label(class_idx))
            .unwrap_or_else(|| format!("LABEL_{}", class_idx))
    }

    /// Check if this worker supports a specific model.
    /// If models list is empty, worker accepts any model.
    fn supports_model(&self, model_id: &str) -> bool {
        self.metadata().supports_model(model_id)
    }

    /// Check if this worker supports an endpoint for a given model.
    /// Falls back to default_model_type if model not found.
    fn supports_endpoint(&self, model_id: &str, endpoint: Endpoint) -> bool {
        self.metadata().supports_endpoint(model_id, endpoint)
    }

    /// Get all models this worker can serve.
    fn models(&self) -> &[ModelCard] {
        self.metadata().spec.models.all()
    }

    /// Set models for this worker (for lazy discovery).
    /// Default implementation does nothing - only BasicWorker supports this.
    fn set_models(&self, _models: Vec<ModelCard>) {
        // Default: no-op. BasicWorker overrides this.
    }

    /// Check if models have been discovered for this worker.
    /// Returns true if models were set via set_models() or if metadata has models.
    fn has_models_discovered(&self) -> bool {
        !self.metadata().spec.models.is_wildcard()
    }

    /// Get or create a gRPC client for this worker
    /// Returns None for HTTP workers, Some(client) for gRPC workers
    async fn get_grpc_client(&self) -> WorkerResult<Option<Arc<GrpcClient>>>;

    /// Reset the gRPC client connection (for reconnection scenarios)
    /// No-op for HTTP workers
    async fn reset_grpc_client(&self) -> WorkerResult<()> {
        Ok(())
    }
    async fn grpc_health_check(&self) -> WorkerResult<bool>;
    async fn http_health_check(&self) -> WorkerResult<bool>;
}

/// Extension trait for model_gateway-specific ConnectionMode methods.
pub(crate) trait ConnectionModeExt {
    fn as_metric_label(&self) -> &'static str;
}

impl ConnectionModeExt for ConnectionMode {
    fn as_metric_label(&self) -> &'static str {
        match self {
            ConnectionMode::Http => metrics_labels::CONNECTION_HTTP,
            ConnectionMode::Grpc => metrics_labels::CONNECTION_GRPC,
        }
    }
}

/// Extension trait for model_gateway-specific WorkerType methods.
pub(crate) trait WorkerTypeExt {
    fn as_metric_label(&self) -> &'static str;
}

impl WorkerTypeExt for WorkerType {
    fn as_metric_label(&self) -> &'static str {
        match self {
            WorkerType::Regular => metrics_labels::WORKER_REGULAR,
            WorkerType::Prefill => metrics_labels::WORKER_PREFILL,
            WorkerType::Decode => metrics_labels::WORKER_DECODE,
        }
    }
}

/// Metadata associated with a worker.
///
/// Embeds [`WorkerSpec`] for identity/config fields shared with the
/// protocol layer, plus internal-only fields for health checking and
/// endpoint routing.
#[derive(Debug, Clone)]
pub struct WorkerMetadata {
    /// Protocol-level worker identity and configuration.
    pub spec: WorkerSpec,
    /// Resolved health check config (router defaults + per-worker overrides).
    /// This is the concrete config used at runtime; `spec.health` only stores
    /// the partial overrides from the API layer.
    pub health_config: HealthCheckConfig,
    /// Health check endpoint path (internal-only, from router config).
    pub health_endpoint: String,
    /// Default model type for unknown models (defaults to LLM capabilities).
    pub default_model_type: ModelType,
}

impl WorkerMetadata {
    /// Find a model card by ID (including aliases)
    pub fn find_model(&self, model_id: &str) -> Option<&ModelCard> {
        self.spec.models.find(model_id)
    }

    /// Check if this worker can serve a given model.
    /// Wildcard workers accept any model.
    pub fn supports_model(&self, model_id: &str) -> bool {
        self.spec.models.supports(model_id)
    }

    /// Check if this worker supports an endpoint for a given model.
    /// Falls back to default_model_type if model not found.
    pub fn supports_endpoint(&self, model_id: &str, endpoint: Endpoint) -> bool {
        if let Some(model) = self.find_model(model_id) {
            model.supports_endpoint(endpoint)
        } else {
            self.default_model_type.supports_endpoint(endpoint)
        }
    }

    /// Get the provider for a given model.
    /// Returns the model's provider if found, otherwise the worker's default provider.
    pub fn provider_for_model(&self, model_id: &str) -> Option<&ProviderType> {
        self.find_model(model_id)
            .and_then(|m| m.provider.as_ref())
            .or(self.spec.provider.as_ref())
    }

    /// Get all model IDs this worker can serve
    pub fn model_ids(&self) -> impl Iterator<Item = &str> {
        self.spec.models.iter().map(|m| m.id.as_str())
    }

    /// Check if this worker is in wildcard mode (accepts any model).
    pub fn is_wildcard(&self) -> bool {
        self.spec.models.is_wildcard()
    }
}

/// Basic worker implementation
#[derive(Clone)]
pub struct BasicWorker {
    pub metadata: WorkerMetadata,
    pub load_counter: Arc<AtomicUsize>,
    pub worker_routing_key_load: Arc<WorkerRoutingKeyLoad>,
    pub processed_counter: Arc<AtomicUsize>,
    pub healthy: Arc<AtomicBool>,
    pub consecutive_failures: Arc<AtomicUsize>,
    pub consecutive_successes: Arc<AtomicUsize>,
    pub circuit_breaker: CircuitBreaker,
    /// Lazily initialized gRPC client for gRPC workers.
    /// Uses OnceCell for lock-free reads after initialization.
    pub grpc_client: Arc<OnceCell<Arc<GrpcClient>>>,
    /// Runtime-mutable models override (for lazy discovery)
    /// When set, overrides metadata.models for routing decisions.
    /// Uses std::sync::RwLock for synchronous access in supports_model().
    pub models_override: Arc<StdRwLock<Option<WorkerModels>>>,
    /// DP rank for this worker (None = not DP-aware)
    pub dp_rank: Option<usize>,
    /// Total DP size (None = not DP-aware)
    pub dp_size: Option<usize>,
    /// Base URL without DP suffix (None = not DP-aware)
    pub dp_base_url: Option<String>,
}

impl fmt::Debug for BasicWorker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = f.debug_struct("BasicWorker");
        s.field("metadata", &self.metadata)
            .field("healthy", &self.healthy.load(Ordering::Relaxed))
            .field("circuit_breaker", &self.circuit_breaker)
            .field("grpc_client", &"<RwLock>");
        if let Some(rank) = self.dp_rank {
            s.field("dp_rank", &rank);
        }
        if let Some(size) = self.dp_size {
            s.field("dp_size", &size);
        }
        if let Some(ref base_url) = self.dp_base_url {
            s.field("dp_base_url", base_url);
        }
        s.finish()
    }
}

impl BasicWorker {
    fn update_running_requests_metrics(&self) {
        let load = self.load();
        Metrics::set_worker_requests_active(self.url(), load);
    }
}

#[async_trait]
impl Worker for BasicWorker {
    fn url(&self) -> &str {
        &self.metadata.spec.url
    }

    fn api_key(&self) -> &Option<String> {
        &self.metadata.spec.api_key
    }

    fn worker_type(&self) -> &WorkerType {
        &self.metadata.spec.worker_type
    }

    fn connection_mode(&self) -> &ConnectionMode {
        &self.metadata.spec.connection_mode
    }

    fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Acquire)
    }

    fn set_healthy(&self, healthy: bool) {
        self.healthy.store(healthy, Ordering::Release);
        Metrics::set_worker_health(self.url(), healthy);
    }

    async fn check_health_async(&self) -> WorkerResult<()> {
        if self.metadata.health_config.disable_health_check {
            if !self.is_healthy() {
                self.set_healthy(true);
            }
            return Ok(());
        }

        let health_result = match &self.metadata.spec.connection_mode {
            ConnectionMode::Http => self.http_health_check().await?,
            ConnectionMode::Grpc => self.grpc_health_check().await?,
        };

        // Get worker type label for metrics
        let worker_type_str = self.metadata.spec.worker_type.as_metric_label();

        if health_result {
            self.consecutive_failures.store(0, Ordering::Release);
            let successes = self.consecutive_successes.fetch_add(1, Ordering::AcqRel) + 1;

            // Record health check success metric
            Metrics::record_worker_health_check(worker_type_str, metrics_labels::CB_SUCCESS);

            if !self.is_healthy()
                && successes >= self.metadata.health_config.success_threshold as usize
            {
                self.set_healthy(true);
                self.consecutive_successes.store(0, Ordering::Release);
            }
            Ok(())
        } else {
            self.consecutive_successes.store(0, Ordering::Release);
            let failures = self.consecutive_failures.fetch_add(1, Ordering::AcqRel) + 1;

            // Record health check failure metric
            Metrics::record_worker_health_check(worker_type_str, metrics_labels::CB_FAILURE);

            if self.is_healthy()
                && failures >= self.metadata.health_config.failure_threshold as usize
            {
                self.set_healthy(false);
                self.consecutive_failures.store(0, Ordering::Release);
            }

            Err(WorkerError::HealthCheckFailed {
                url: self.metadata.spec.url.clone(),
                reason: format!("Health check failed (consecutive failures: {})", failures),
            })
        }
    }

    fn load(&self) -> usize {
        self.load_counter.load(Ordering::Relaxed)
    }

    fn increment_load(&self) {
        self.load_counter.fetch_add(1, Ordering::Relaxed);
        self.update_running_requests_metrics();
    }

    fn decrement_load(&self) {
        if self
            .load_counter
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_sub(1)
            })
            .is_err()
        {
            tracing::warn!(
                worker_url = %self.metadata.spec.url,
                "Attempted to decrement load counter that is already at 0"
            );
        }
        self.update_running_requests_metrics();
    }

    fn reset_load(&self) {
        self.load_counter.store(0, Ordering::Relaxed);
        self.update_running_requests_metrics();
    }

    fn worker_routing_key_load(&self) -> &WorkerRoutingKeyLoad {
        &self.worker_routing_key_load
    }

    fn processed_requests(&self) -> usize {
        self.processed_counter.load(Ordering::Relaxed)
    }

    fn increment_processed(&self) {
        self.processed_counter.fetch_add(1, Ordering::Relaxed);
    }

    fn metadata(&self) -> &WorkerMetadata {
        &self.metadata
    }

    fn circuit_breaker(&self) -> &CircuitBreaker {
        &self.circuit_breaker
    }

    fn is_dp_aware(&self) -> bool {
        self.dp_rank.is_some()
    }

    fn base_url(&self) -> &str {
        self.dp_base_url.as_deref().unwrap_or(self.url())
    }

    fn dp_rank(&self) -> Option<usize> {
        self.dp_rank
    }

    fn dp_size(&self) -> Option<usize> {
        self.dp_size
    }

    async fn prepare_request(&self, mut req: serde_json::Value) -> WorkerResult<serde_json::Value> {
        if let Some(rank) = self.dp_rank {
            if let Some(map) = req.as_object_mut() {
                map.insert("data_parallel_rank".to_string(), serde_json::json!(rank));
                Ok(req)
            } else {
                Err(WorkerError::InvalidConfiguration {
                    message: "Request must be a JSON object for DP-aware routing".to_string(),
                })
            }
        } else {
            Ok(req)
        }
    }

    fn endpoint_url(&self, route: &str) -> String {
        format!("{}{}", self.base_url(), route)
    }

    fn supports_model(&self, model_id: &str) -> bool {
        // Check models_override first (for lazy discovery)
        if let Ok(guard) = self.models_override.read() {
            if let Some(ref models) = *guard {
                // Models were discovered - check if this model is supported
                return models.supports(model_id);
            }
        }
        // Fall back to metadata.models
        self.metadata.supports_model(model_id)
    }

    fn set_models(&self, models: Vec<ModelCard>) {
        if let Ok(mut guard) = self.models_override.write() {
            tracing::debug!(
                "Setting {} models for worker {} via lazy discovery",
                models.len(),
                self.metadata.spec.url
            );
            *guard = Some(WorkerModels::from(models));
        }
    }

    fn has_models_discovered(&self) -> bool {
        // Check if models_override has been set
        if let Ok(guard) = self.models_override.read() {
            if guard.is_some() {
                return true;
            }
        }
        // Fall back to checking metadata.models
        !self.metadata.spec.models.is_wildcard()
    }

    async fn get_grpc_client(&self) -> WorkerResult<Option<Arc<GrpcClient>>> {
        match self.metadata.spec.connection_mode {
            ConnectionMode::Http => Ok(None),
            ConnectionMode::Grpc => {
                // OnceCell provides lock-free reads after initialization.
                // get_or_try_init only acquires internal lock on first call.
                let client = self
                    .grpc_client
                    .get_or_try_init(|| async {
                        let runtime_str = self.metadata.spec.runtime_type.to_string();
                        tracing::info!(
                            "Lazily initializing gRPC client ({}) for worker: {}",
                            runtime_str,
                            self.metadata.spec.url
                        );
                        match GrpcClient::connect(&self.metadata.spec.url, &runtime_str).await {
                            Ok(client) => {
                                tracing::info!(
                                    "Successfully connected gRPC client ({}) for worker: {}",
                                    runtime_str,
                                    self.metadata.spec.url
                                );
                                Ok(Arc::new(client))
                            }
                            Err(e) => {
                                tracing::error!(
                                    "Failed to connect gRPC client for worker {}: {}",
                                    self.metadata.spec.url,
                                    e
                                );
                                Err(WorkerError::ConnectionFailed {
                                    url: self.metadata.spec.url.clone(),
                                    reason: format!("Failed to connect to gRPC server: {}", e),
                                })
                            }
                        }
                    })
                    .await?;
                Ok(Some(Arc::clone(client)))
            }
        }
    }

    async fn reset_grpc_client(&self) -> WorkerResult<()> {
        // OnceCell doesn't support resetting. This is intentional for lock-free performance.
        // If a connection fails, the worker should be removed and re-added.
        tracing::debug!(
            "reset_grpc_client called for {} (no-op with OnceCell)",
            self.metadata.spec.url
        );
        Ok(())
    }

    async fn grpc_health_check(&self) -> WorkerResult<bool> {
        let timeout = Duration::from_secs(self.metadata.health_config.timeout_secs);
        let maybe = self.get_grpc_client().await?;
        let Some(grpc_client) = maybe else {
            tracing::error!(
                "Worker {} is not a gRPC worker but connection mode is gRPC",
                self.metadata.spec.url
            );
            return Ok(false);
        };

        match time::timeout(timeout, grpc_client.health_check()).await {
            Ok(Ok(resp)) => {
                tracing::debug!(
                    "gRPC health OK for {}: healthy={}",
                    self.metadata.spec.url,
                    resp.healthy
                );
                Ok(resp.healthy)
            }
            Ok(Err(err)) => {
                tracing::warn!(
                    "gRPC health RPC error for {}: {err:?}",
                    self.metadata.spec.url
                );
                Ok(false)
            }
            Err(_) => {
                tracing::warn!("gRPC health timed out for {}", self.metadata.spec.url);
                Ok(false)
            }
        }
    }

    async fn http_health_check(&self) -> WorkerResult<bool> {
        let timeout = Duration::from_secs(self.metadata.health_config.timeout_secs);

        let health_url = format!("{}{}", self.base_url(), self.metadata.health_endpoint);

        let mut req = WORKER_CLIENT.get(&health_url).timeout(timeout);
        if let Some(api_key) = &self.metadata.spec.api_key {
            req = req.bearer_auth(api_key);
        }

        match req.send().await {
            Ok(resp) => {
                let status = resp.status();
                if status.is_success() {
                    Ok(true)
                } else {
                    tracing::warn!(
                        "HTTP health check returned non-success status for {}: {}",
                        health_url,
                        status
                    );
                    Ok(false)
                }
            }
            Err(err) => {
                tracing::warn!("HTTP health check failed for {}: {err:?}", health_url);
                Ok(false)
            }
        }
    }
}

/// RAII guard for worker load management
///
/// Automatically decrements worker load when dropped. Can be attached to
/// an axum Response to tie the guard's lifetime to the response body,
/// which is essential for streaming responses where the function returns
/// immediately but the stream continues in the background.
pub struct WorkerLoadGuard {
    worker: Arc<dyn Worker>,
    routing_key: Option<String>,
}

impl WorkerLoadGuard {
    pub fn new(worker: Arc<dyn Worker>, headers: Option<&http::HeaderMap>) -> Self {
        use crate::routers::header_utils::extract_routing_key;

        worker.increment_load();

        let routing_key = extract_routing_key(headers).map(String::from);

        if let Some(ref key) = routing_key {
            worker.worker_routing_key_load().increment(key);
        }

        Self {
            worker,
            routing_key,
        }
    }
}

impl Drop for WorkerLoadGuard {
    fn drop(&mut self) {
        self.worker.decrement_load();
        if let Some(ref key) = self.routing_key {
            self.worker.worker_routing_key_load().decrement(key);
        }
    }
}

/// Body wrapper that holds an attached value.
///
/// When this body is dropped (stream ends or client disconnects),
/// the attached value is dropped automatically. This is useful for RAII guards
/// like WorkerLoadGuard that need to be tied to a response body's lifetime.
pub struct AttachedBody<T> {
    inner: Body,
    _attached: T,
}

impl<T> AttachedBody<T> {
    pub fn new(inner: Body, attached: T) -> Self {
        Self {
            inner,
            _attached: attached,
        }
    }
}

impl<T: Send + Unpin + 'static> AttachedBody<T> {
    pub fn wrap_response(
        response: axum::response::Response,
        attached: T,
    ) -> axum::response::Response {
        let (parts, body) = response.into_parts();
        axum::response::Response::from_parts(parts, Body::new(Self::new(body, attached)))
    }
}

impl<T: Send + Unpin + 'static> http_body::Body for AttachedBody<T> {
    type Data = bytes::Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Result<http_body::Frame<Self::Data>, Self::Error>>> {
        let this = self.get_mut();
        std::pin::Pin::new(&mut this.inner).poll_frame(cx)
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }
}

/// Health checker handle with graceful shutdown.
///
/// The checker sleeps until the next worker is due for a health check,
/// so it wakes only when there is actual work to do.
pub(crate) struct HealthChecker {
    #[allow(dead_code)]
    handle: tokio::task::JoinHandle<()>,
    shutdown_notify: Arc<tokio::sync::Notify>,
}

impl fmt::Debug for HealthChecker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HealthChecker").finish()
    }
}

impl HealthChecker {
    pub fn new(
        handle: tokio::task::JoinHandle<()>,
        shutdown_notify: Arc<tokio::sync::Notify>,
    ) -> Self {
        Self {
            handle,
            shutdown_notify,
        }
    }

    /// Shutdown the health checker gracefully.
    /// Wakes the sleeping task immediately so it can exit.
    #[allow(dead_code)]
    pub async fn shutdown(self) {
        self.shutdown_notify.notify_one();
        let _ = self.handle.await;
    }
}

/// Helper to convert Worker trait object to WorkerInfo struct
pub fn worker_to_info(worker: &Arc<dyn Worker>) -> WorkerInfo {
    let metadata = worker.metadata();
    let spec = metadata.spec.clone();

    WorkerInfo {
        id: worker.url().to_string(),
        spec,
        is_healthy: worker.is_healthy(),
        load: worker.load(),
        job_status: None,
    }
}

#[cfg(test)]
mod tests {
    use std::{thread, time::Duration};

    use super::*;
    use crate::core::{
        circuit_breaker::{CircuitBreakerConfig, CircuitState},
        BasicWorkerBuilder,
    };

    #[test]
    fn test_worker_type_display() {
        assert_eq!(WorkerType::Regular.to_string(), "regular");
        assert_eq!(WorkerType::Prefill.to_string(), "prefill");
        assert_eq!(WorkerType::Decode.to_string(), "decode");
    }

    #[test]
    fn test_worker_type_equality() {
        assert_eq!(WorkerType::Regular, WorkerType::Regular);
        assert_ne!(WorkerType::Regular, WorkerType::Decode);
        assert_eq!(WorkerType::Prefill, WorkerType::Prefill);
    }

    #[test]
    fn test_worker_type_clone() {
        let original = WorkerType::Prefill;
        let cloned = original;
        assert_eq!(original, cloned);
    }

    #[test]
    fn test_health_config_default() {
        use openai_protocol::worker::HealthCheckConfig;
        let config = HealthCheckConfig::default();
        assert_eq!(config.timeout_secs, 30);
        assert_eq!(config.check_interval_secs, 60);
        assert_eq!(config.failure_threshold, 3);
        assert_eq!(config.success_threshold, 2);
        assert!(!config.disable_health_check);
    }

    #[test]
    fn test_health_config_custom() {
        use openai_protocol::worker::HealthCheckConfig;
        let config = HealthCheckConfig {
            timeout_secs: 10,
            check_interval_secs: 60,
            failure_threshold: 5,
            success_threshold: 3,
            disable_health_check: true,
        };
        assert_eq!(config.timeout_secs, 10);
        assert_eq!(config.check_interval_secs, 60);
        assert_eq!(config.failure_threshold, 5);
        assert_eq!(config.success_threshold, 3);
        assert!(config.disable_health_check);
    }

    #[test]
    fn test_basic_worker_creation() {
        use crate::core::BasicWorkerBuilder;
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .build();
        assert_eq!(worker.url(), "http://test:8080");
        assert_eq!(worker.worker_type(), &WorkerType::Regular);
        assert!(worker.is_healthy());
        assert_eq!(worker.load(), 0);
        assert_eq!(worker.processed_requests(), 0);
    }

    #[test]
    fn test_worker_with_labels() {
        let mut labels = std::collections::HashMap::new();
        labels.insert("env".to_string(), "prod".to_string());
        labels.insert("zone".to_string(), "us-west".to_string());

        use crate::core::BasicWorkerBuilder;
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .labels(labels.clone())
            .build();

        assert_eq!(worker.metadata().spec.labels, labels);
    }

    #[test]
    fn test_worker_with_health_config() {
        use openai_protocol::worker::HealthCheckConfig;
        let custom_config = HealthCheckConfig {
            timeout_secs: 15,
            check_interval_secs: 45,
            failure_threshold: 4,
            success_threshold: 2,
            disable_health_check: false,
        };

        use crate::core::BasicWorkerBuilder;
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .health_config(custom_config.clone())
            .health_endpoint("/custom-health")
            .build();

        assert_eq!(worker.metadata().health_config.timeout_secs, 15);
        assert_eq!(worker.metadata().health_config.check_interval_secs, 45);
        assert_eq!(worker.metadata().health_endpoint, "/custom-health");
    }

    #[test]
    fn test_worker_url() {
        use crate::core::BasicWorkerBuilder;
        let worker = BasicWorkerBuilder::new("http://worker1:8080")
            .worker_type(WorkerType::Regular)
            .build();
        assert_eq!(worker.url(), "http://worker1:8080");
    }

    #[test]
    fn test_worker_type_getter() {
        use crate::core::BasicWorkerBuilder;
        let regular = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .build();
        assert_eq!(regular.worker_type(), &WorkerType::Regular);

        let prefill = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Prefill)
            .bootstrap_port(Some(9090))
            .build();
        assert_eq!(prefill.worker_type(), &WorkerType::Prefill);
        assert_eq!(prefill.bootstrap_port(), Some(9090));

        let decode = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Decode)
            .build();
        assert_eq!(decode.worker_type(), &WorkerType::Decode);
    }

    #[test]
    fn test_health_status() {
        use crate::core::BasicWorkerBuilder;
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .build();

        assert!(worker.is_healthy());

        worker.set_healthy(false);
        assert!(!worker.is_healthy());

        worker.set_healthy(true);
        assert!(worker.is_healthy());
    }

    #[test]
    fn test_load_counter_operations() {
        use crate::core::BasicWorkerBuilder;
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .build();

        assert_eq!(worker.load(), 0);

        worker.increment_load();
        assert_eq!(worker.load(), 1);

        worker.increment_load();
        worker.increment_load();
        assert_eq!(worker.load(), 3);

        worker.decrement_load();
        assert_eq!(worker.load(), 2);

        worker.decrement_load();
        worker.decrement_load();
        assert_eq!(worker.load(), 0);

        worker.decrement_load();
        assert_eq!(worker.load(), 0);
    }

    #[test]
    fn test_processed_counter() {
        use crate::core::BasicWorkerBuilder;
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .build();

        assert_eq!(worker.processed_requests(), 0);

        for i in 1..=100 {
            worker.increment_processed();
            assert_eq!(worker.processed_requests(), i);
        }
    }

    #[tokio::test]
    async fn test_concurrent_load_increments() {
        use crate::core::BasicWorkerBuilder;
        let worker = Arc::new(
            BasicWorkerBuilder::new("http://test:8080")
                .worker_type(WorkerType::Regular)
                .build(),
        );

        let mut handles = vec![];

        for _ in 0..100 {
            let worker_clone = Arc::clone(&worker);
            let handle = tokio::spawn(async move {
                worker_clone.increment_load();
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.await.unwrap();
        }

        assert_eq!(worker.load(), 100);
    }

    #[tokio::test]
    async fn test_concurrent_load_decrements() {
        use crate::core::BasicWorkerBuilder;
        let worker = Arc::new(
            BasicWorkerBuilder::new("http://test:8080")
                .worker_type(WorkerType::Regular)
                .build(),
        );

        for _ in 0..100 {
            worker.increment_load();
        }
        assert_eq!(worker.load(), 100);

        let mut handles = vec![];

        for _ in 0..100 {
            let worker_clone = Arc::clone(&worker);
            let handle = tokio::spawn(async move {
                worker_clone.decrement_load();
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.await.unwrap();
        }

        assert_eq!(worker.load(), 0);
    }

    #[tokio::test]
    async fn test_concurrent_health_updates() {
        use crate::core::BasicWorkerBuilder;
        let worker = Arc::new(
            BasicWorkerBuilder::new("http://test:8080")
                .worker_type(WorkerType::Regular)
                .build(),
        );

        let mut handles = vec![];

        for i in 0..100 {
            let worker_clone = Arc::clone(&worker);
            let handle = tokio::spawn(async move {
                worker_clone.set_healthy(i % 2 == 0);
                time::sleep(Duration::from_micros(10)).await;
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.await.unwrap();
        }
    }

    #[test]
    fn test_create_regular_worker() {
        use crate::core::BasicWorkerBuilder;
        let worker: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://regular:8080")
                .worker_type(WorkerType::Regular)
                .build(),
        );
        assert_eq!(worker.url(), "http://regular:8080");
        assert_eq!(worker.worker_type(), &WorkerType::Regular);
    }

    #[test]
    fn test_create_prefill_worker() {
        use crate::core::BasicWorkerBuilder;
        let worker1: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://prefill:8080")
                .worker_type(WorkerType::Prefill)
                .bootstrap_port(Some(9090))
                .build(),
        );
        assert_eq!(worker1.url(), "http://prefill:8080");
        assert_eq!(worker1.worker_type(), &WorkerType::Prefill);
        assert_eq!(worker1.bootstrap_port(), Some(9090));

        let worker2: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://prefill:8080")
                .worker_type(WorkerType::Prefill)
                .build(),
        );
        assert_eq!(worker2.worker_type(), &WorkerType::Prefill);
        assert_eq!(worker2.bootstrap_port(), None);
    }

    #[test]
    fn test_create_decode_worker() {
        use crate::core::BasicWorkerBuilder;
        let worker: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://decode:8080")
                .worker_type(WorkerType::Decode)
                .build(),
        );
        assert_eq!(worker.url(), "http://decode:8080");
        assert_eq!(worker.worker_type(), &WorkerType::Decode);
    }

    #[tokio::test]
    async fn test_check_health_async() {
        use crate::core::BasicWorkerBuilder;
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .build();

        // Health check should fail since there's no actual server
        let result = worker.check_health_async().await;
        assert!(result.is_err());
    }

    #[test]
    fn test_load_counter_performance() {
        use std::time::Instant;

        use crate::core::BasicWorkerBuilder;

        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .build();
        let iterations = 1_000_000;

        let start = Instant::now();
        for _ in 0..iterations {
            worker.increment_load();
        }
        let duration = start.elapsed();

        let ops_per_sec = iterations as f64 / duration.as_secs_f64();
        println!("Load counter operations per second: {:.0}", ops_per_sec);

        assert!(ops_per_sec > 1_000_000.0);
    }

    #[test]
    fn test_dp_aware_worker_creation() {
        let dp_worker = BasicWorkerBuilder::new("http://worker1:8080")
            .dp_config(2, 4)
            .worker_type(WorkerType::Regular)
            .build();

        assert_eq!(dp_worker.url(), "http://worker1:8080@2");
        assert_eq!(dp_worker.base_url(), "http://worker1:8080");
        assert!(dp_worker.is_dp_aware());
        assert_eq!(dp_worker.dp_rank(), Some(2));
        assert_eq!(dp_worker.dp_size(), Some(4));
        assert_eq!(dp_worker.worker_type(), &WorkerType::Regular);
    }

    #[test]
    fn test_dp_aware_worker_creation_prefill() {
        let dp_worker = BasicWorkerBuilder::new("http://worker1:8080")
            .dp_config(1, 2)
            .worker_type(WorkerType::Prefill)
            .build();

        assert_eq!(dp_worker.url(), "http://worker1:8080@1");
        assert!(dp_worker.is_dp_aware());
        assert_eq!(dp_worker.worker_type(), &WorkerType::Prefill);
    }

    #[test]
    fn test_dp_aware_worker_creation_decode() {
        let dp_worker = BasicWorkerBuilder::new("http://worker1:8080")
            .dp_config(0, 4)
            .worker_type(WorkerType::Decode)
            .build();

        assert_eq!(dp_worker.url(), "http://worker1:8080@0");
        assert!(dp_worker.is_dp_aware());
        assert_eq!(dp_worker.worker_type(), &WorkerType::Decode);
    }

    #[tokio::test]
    async fn test_dp_aware_prepare_request() {
        let dp_worker = BasicWorkerBuilder::new("http://worker1:8080")
            .dp_config(3, 8)
            .worker_type(WorkerType::Regular)
            .build();

        let original_req = serde_json::json!({
            "prompt": "Hello",
            "max_tokens": 100
        });

        let prepared_req = dp_worker.prepare_request(original_req).await.unwrap();

        assert_eq!(prepared_req["prompt"], "Hello");
        assert_eq!(prepared_req["max_tokens"], 100);
        assert_eq!(prepared_req["data_parallel_rank"], 3);
    }

    #[tokio::test]
    async fn test_dp_aware_prepare_request_invalid() {
        let dp_worker = BasicWorkerBuilder::new("http://worker1:8080")
            .dp_config(0, 4)
            .worker_type(WorkerType::Regular)
            .build();

        // Non-object JSON should fail
        let invalid_req = serde_json::json!("not an object");
        let result = dp_worker.prepare_request(invalid_req).await;

        assert!(result.is_err());
        match result.unwrap_err() {
            WorkerError::InvalidConfiguration { message } => {
                assert!(message.contains("JSON object"));
            }
            _ => panic!("Expected InvalidConfiguration error"),
        }
    }

    #[test]
    fn test_dp_aware_endpoint_url() {
        let dp_worker = BasicWorkerBuilder::new("http://worker1:8080")
            .dp_config(1, 4)
            .worker_type(WorkerType::Regular)
            .build();

        assert_eq!(
            dp_worker.endpoint_url("/generate"),
            "http://worker1:8080/generate"
        );
        assert_eq!(
            dp_worker.endpoint_url("/health"),
            "http://worker1:8080/health"
        );
    }

    #[test]
    fn test_dp_aware_worker_delegated_methods() {
        let dp_worker = BasicWorkerBuilder::new("http://worker1:8080")
            .dp_config(0, 2)
            .worker_type(WorkerType::Regular)
            .build();

        assert!(dp_worker.is_healthy());
        dp_worker.set_healthy(false);
        assert!(!dp_worker.is_healthy());

        assert_eq!(dp_worker.load(), 0);
        dp_worker.increment_load();
        assert_eq!(dp_worker.load(), 1);
        dp_worker.decrement_load();
        assert_eq!(dp_worker.load(), 0);

        assert_eq!(dp_worker.processed_requests(), 0);
        dp_worker.increment_processed();
        assert_eq!(dp_worker.processed_requests(), 1);
    }

    #[test]
    fn test_worker_circuit_breaker() {
        use crate::core::BasicWorkerBuilder;
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .build();

        assert!(worker.is_available());
        assert_eq!(worker.circuit_breaker().state(), CircuitState::Closed);

        worker.record_outcome(false);
        worker.record_outcome(false);

        assert!(worker.is_available());

        worker.record_outcome(false);
        worker.record_outcome(false);
        worker.record_outcome(false);

        assert!(!worker.is_available());
        assert!(worker.is_healthy());
        assert!(!worker.circuit_breaker().can_execute());
    }

    #[test]
    fn test_worker_with_circuit_breaker_config() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 1,
            timeout_duration: Duration::from_millis(100),
            window_duration: Duration::from_secs(60),
        };

        use crate::core::BasicWorkerBuilder;
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .circuit_breaker_config(config)
            .build();

        worker.record_outcome(false);
        assert!(worker.is_available());
        worker.record_outcome(false);
        assert!(!worker.is_available());

        thread::sleep(Duration::from_millis(150));

        assert!(worker.is_available());
        assert_eq!(worker.circuit_breaker().state(), CircuitState::HalfOpen);

        worker.record_outcome(true);
        assert_eq!(worker.circuit_breaker().state(), CircuitState::Closed);
    }

    #[test]
    fn test_dp_aware_worker_circuit_breaker() {
        let dp_worker = BasicWorkerBuilder::new("http://worker:8080")
            .dp_config(0, 2)
            .worker_type(WorkerType::Regular)
            .build();

        assert!(dp_worker.is_available());

        for _ in 0..5 {
            dp_worker.record_outcome(false);
        }

        assert!(!dp_worker.is_available());
        assert_eq!(dp_worker.circuit_breaker().state(), CircuitState::Open);
    }

    #[tokio::test]
    async fn test_mixed_worker_types() {
        use crate::core::BasicWorkerBuilder;
        let regular: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://regular:8080")
                .worker_type(WorkerType::Regular)
                .build(),
        );
        let prefill: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://prefill:8080")
                .worker_type(WorkerType::Prefill)
                .bootstrap_port(Some(9090))
                .build(),
        );
        let decode: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://decode:8080")
                .worker_type(WorkerType::Decode)
                .build(),
        );
        let dp_aware_regular: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://dp:8080")
                .dp_config(0, 2)
                .worker_type(WorkerType::Regular)
                .api_key("test_api_key")
                .build(),
        );
        let dp_aware_prefill: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://dp-prefill:8080")
                .dp_config(1, 2)
                .worker_type(WorkerType::Prefill)
                .api_key("test_api_key")
                .build(),
        );
        let dp_aware_decode: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://dp-decode:8080")
                .dp_config(0, 4)
                .worker_type(WorkerType::Decode)
                .api_key("test_api_key")
                .build(),
        );

        let workers: Vec<Box<dyn Worker>> = vec![
            regular,
            prefill,
            decode,
            dp_aware_regular,
            dp_aware_prefill,
            dp_aware_decode,
        ];

        for worker in &workers {
            assert!(worker.is_healthy());
            assert_eq!(worker.load(), 0);
            assert_eq!(worker.processed_requests(), 0);
        }

        assert!(!workers[0].is_dp_aware());
        assert!(!workers[1].is_dp_aware());
        assert!(!workers[2].is_dp_aware());
        assert!(workers[3].is_dp_aware());
        assert!(workers[4].is_dp_aware());
        assert!(workers[5].is_dp_aware());

        assert_eq!(workers[0].worker_type(), &WorkerType::Regular);
        assert_eq!(workers[1].worker_type(), &WorkerType::Prefill);
        assert_eq!(workers[2].worker_type(), &WorkerType::Decode);
        assert_eq!(workers[3].worker_type(), &WorkerType::Regular);
        assert_eq!(workers[4].worker_type(), &WorkerType::Prefill);
        assert_eq!(workers[5].worker_type(), &WorkerType::Decode);
    }

    // === Phase 1.3: WorkerMetadata model methods tests ===

    #[test]
    fn test_worker_metadata_empty_models_accepts_all() {
        let metadata = WorkerMetadata {
            spec: WorkerSpec::new("http://test:8080"),
            health_config: HealthCheckConfig::default(),
            health_endpoint: "/health".to_string(),
            default_model_type: ModelType::LLM,
        };

        // Empty models list should accept any model
        assert!(metadata.supports_model("any-model"));
        assert!(metadata.supports_model("gpt-4"));
        assert!(metadata.supports_model("llama-3.1"));
    }

    #[test]
    fn test_worker_metadata_find_model() {
        use super::ModelCard;

        let model1 = ModelCard::new("meta-llama/Llama-3.1-8B")
            .with_alias("llama-3.1-8b")
            .with_alias("llama3.1");
        let model2 = ModelCard::new("gpt-4o");

        let mut spec = WorkerSpec::new("http://test:8080");
        spec.models = WorkerModels::from(vec![model1, model2]);
        let metadata = WorkerMetadata {
            spec,
            health_config: HealthCheckConfig::default(),
            health_endpoint: "/health".to_string(),
            default_model_type: ModelType::LLM,
        };

        // Find by primary ID
        assert!(metadata.find_model("meta-llama/Llama-3.1-8B").is_some());
        assert!(metadata.find_model("gpt-4o").is_some());

        // Find by alias
        assert!(metadata.find_model("llama-3.1-8b").is_some());
        assert!(metadata.find_model("llama3.1").is_some());

        // Not found
        assert!(metadata.find_model("unknown-model").is_none());
    }

    #[test]
    fn test_worker_routing_key_load_increment_decrement() {
        let load = WorkerRoutingKeyLoad::new("http://test:8000");
        assert_eq!(load.value(), 0);

        load.increment("key1");
        assert_eq!(load.value(), 1);

        load.increment("key2");
        assert_eq!(load.value(), 2);

        load.increment("key1");
        assert_eq!(load.value(), 2);

        load.decrement("key1");
        assert_eq!(load.value(), 2);

        load.decrement("key1");
        assert_eq!(load.value(), 1);

        load.decrement("key2");
        assert_eq!(load.value(), 0);
    }

    #[test]
    fn test_worker_routing_key_load_cleanup_on_zero() {
        let load = WorkerRoutingKeyLoad::new("http://test:8000");

        load.increment("key1");
        load.increment("key2");
        load.increment("key3");
        assert_eq!(load.active_routing_keys.len(), 3);

        load.decrement("key1");
        assert_eq!(load.active_routing_keys.len(), 2);

        load.decrement("key2");
        assert_eq!(load.active_routing_keys.len(), 1);

        load.decrement("key3");
        assert_eq!(load.active_routing_keys.len(), 0);
    }

    #[test]
    fn test_worker_routing_key_load_multiple_requests_same_key() {
        let load = WorkerRoutingKeyLoad::new("http://test:8000");

        load.increment("key-1");
        load.increment("key-1");
        load.increment("key-1");
        assert_eq!(load.value(), 1);

        load.decrement("key-1");
        assert_eq!(load.value(), 1);

        load.decrement("key-1");
        assert_eq!(load.value(), 1);

        load.decrement("key-1");
        assert_eq!(load.value(), 0);
        assert_eq!(load.active_routing_keys.len(), 0);
    }

    #[test]
    fn test_worker_routing_key_load_decrement_nonexistent() {
        let load = WorkerRoutingKeyLoad::new("http://test:8000");
        load.decrement("nonexistent");
        assert_eq!(load.value(), 0);
    }

    #[test]
    fn test_worker_load_guard_with_routing_key() {
        use crate::core::BasicWorkerBuilder;

        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://test:8000")
                .worker_type(WorkerType::Regular)
                .build(),
        );

        assert_eq!(worker.load(), 0);
        assert_eq!(worker.worker_routing_key_load().value(), 0);

        let mut headers = http::HeaderMap::new();
        headers.insert("x-smg-routing-key", "key-123".parse().unwrap());

        {
            let _guard = WorkerLoadGuard::new(worker.clone(), Some(&headers));
            assert_eq!(worker.load(), 1);
            assert_eq!(worker.worker_routing_key_load().value(), 1);
        }

        assert_eq!(worker.load(), 0);
        assert_eq!(worker.worker_routing_key_load().value(), 0);
    }

    #[test]
    fn test_worker_load_guard_without_routing_key() {
        use crate::core::BasicWorkerBuilder;

        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://test:8000")
                .worker_type(WorkerType::Regular)
                .build(),
        );

        assert_eq!(worker.load(), 0);
        assert_eq!(worker.worker_routing_key_load().value(), 0);

        {
            let _guard = WorkerLoadGuard::new(worker.clone(), None);
            assert_eq!(worker.load(), 1);
            assert_eq!(worker.worker_routing_key_load().value(), 0);
        }

        assert_eq!(worker.load(), 0);
        assert_eq!(worker.worker_routing_key_load().value(), 0);
    }

    #[test]
    fn test_worker_load_guard_multiple_same_routing_key() {
        use crate::core::BasicWorkerBuilder;

        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://test:8000")
                .worker_type(WorkerType::Regular)
                .build(),
        );

        let mut headers = http::HeaderMap::new();
        headers.insert("x-smg-routing-key", "key-123".parse().unwrap());

        let guard1 = WorkerLoadGuard::new(worker.clone(), Some(&headers));
        assert_eq!(worker.load(), 1);
        assert_eq!(worker.worker_routing_key_load().value(), 1);

        let guard2 = WorkerLoadGuard::new(worker.clone(), Some(&headers));
        assert_eq!(worker.load(), 2);
        assert_eq!(worker.worker_routing_key_load().value(), 1);

        drop(guard1);
        assert_eq!(worker.load(), 1);
        assert_eq!(worker.worker_routing_key_load().value(), 1);

        drop(guard2);
        assert_eq!(worker.load(), 0);
        assert_eq!(worker.worker_routing_key_load().value(), 0);
    }
}
