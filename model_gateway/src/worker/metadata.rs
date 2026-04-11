use std::fmt;

use openai_protocol::{
    model_card::ModelCard,
    model_type::{Endpoint, ModelType},
    worker::{HealthCheckConfig, ProviderType, WorkerSpec},
};

use crate::observability::metrics::Metrics;

pub struct WorkerRoutingKeyLoad {
    url: String,
    pub(crate) active_routing_keys: dashmap::DashMap<String, usize>,
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
    /// Falls back to LLM capabilities if model not found — this is safe because
    /// non-LLM workers (embeddings, rerank) are always registered with explicit
    /// models via discovery, never as wildcards.
    pub fn supports_endpoint(&self, model_id: &str, endpoint: Endpoint) -> bool {
        if let Some(model) = self.find_model(model_id) {
            model.supports_endpoint(endpoint)
        } else {
            ModelType::LLM.supports_endpoint(endpoint)
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
