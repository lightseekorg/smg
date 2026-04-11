use std::{fmt, sync::Arc};

use async_trait::async_trait;
// Re-export protocol types as the canonical types for the gateway
pub use openai_protocol::worker::{ConnectionMode, RuntimeType, WorkerType};
use openai_protocol::{
    model_card::ModelCard,
    model_type::Endpoint,
    worker::{ProviderType, WorkerInfo, WorkerStatus},
};

use super::{
    metadata::{WorkerMetadata, WorkerRoutingKeyLoad},
    CircuitBreaker, ResolvedResilience, WorkerError, WorkerResult, UNKNOWN_MODEL_ID,
};
use crate::{observability::metrics::metrics_labels, routers::grpc::client::GrpcClient};

/// Core worker abstraction that represents a backend service
#[async_trait]
pub trait Worker: Send + Sync + fmt::Debug {
    /// Get the worker's URL
    fn url(&self) -> &str;
    /// Get the worker's API key
    fn api_key(&self) -> Option<&String>;
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

    /// Get the circuit breaker for this worker.
    ///
    /// **Do not call from routers.** Use `record_outcome(status_code)` to
    /// record request outcomes and `is_available()` to check worker health.
    fn circuit_breaker(&self) -> &CircuitBreaker;

    /// Check if the worker is available (healthy + circuit closed/half-open)
    fn is_available(&self) -> bool {
        self.is_healthy() && self.circuit_breaker().can_execute()
    }

    /// Record the outcome of a request based on the HTTP status code.
    ///
    /// The worker decides whether the status is a CB failure using its
    /// per-worker `retryable_status_codes` set (default: 408, 429, 5xx).
    /// Callers just pass the status — no need to interpret it.
    ///
    /// For transport/connection errors where no HTTP response is received,
    /// pass the status code returned to the client (e.g., 502 for a send
    /// error, 504 for a timeout).
    fn record_outcome(&self, status_code: u16) {
        let is_failure = self
            .resilience()
            .retryable_status_codes
            .contains(&status_code);
        self.circuit_breaker().record_outcome(!is_failure);
    }

    /// Get the resolved resilience config for this worker.
    fn resilience(&self) -> &ResolvedResilience;

    /// Get the per-worker HTTP client.
    fn http_client(&self) -> &reqwest::Client;

    /// Check if this worker is DP-aware
    fn is_dp_aware(&self) -> bool {
        self.metadata().spec.dp_rank.is_some()
    }

    /// Get the base URL without any DP rank suffix
    fn base_url(&self) -> &str {
        self.metadata()
            .spec
            .dp_base_url
            .as_deref()
            .unwrap_or_else(|| self.url())
    }

    /// Get DP rank if this is a DP-aware worker
    fn dp_rank(&self) -> Option<usize> {
        self.metadata().spec.dp_rank
    }

    /// Get DP size if this worker is part of a DP group
    fn dp_size(&self) -> Option<usize> {
        self.metadata().spec.dp_size
    }

    /// Transform a request for DP-aware routing
    async fn prepare_request(&self, mut req: serde_json::Value) -> WorkerResult<serde_json::Value> {
        if let Some(rank) = self.metadata().spec.dp_rank {
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
            .unwrap_or_else(|| format!("LABEL_{class_idx}"))
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
    fn models(&self) -> Vec<ModelCard> {
        self.metadata().spec.models.all().to_vec()
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

/// Helper to convert Worker trait object to WorkerInfo struct
pub fn worker_to_info(worker: &Arc<dyn Worker>) -> WorkerInfo {
    let metadata = worker.metadata();
    let spec = metadata.spec.clone();
    let is_healthy = worker.is_healthy();

    WorkerInfo {
        id: worker.url().to_string(),
        model_id: spec.models.primary().map(|m| m.id.clone()),
        spec,
        is_healthy,
        status: Some(if is_healthy {
            WorkerStatus::Ready
        } else {
            WorkerStatus::NotReady
        }),
        load: worker.load(),
        job_status: None,
    }
}
