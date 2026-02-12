use std::collections::HashMap;

use super::{
    circuit_breaker::{CircuitBreaker, CircuitBreakerConfig},
    worker::{
        BasicWorker, ConnectionMode, DPAwareWorker, RuntimeType, WorkerMetadata,
        WorkerRoutingKeyLoad, WorkerType,
    },
};
use crate::{
    observability::metrics::Metrics,
    protocols::{
        model_card::ModelCard,
        model_type::ModelType,
        worker::{HealthCheckConfig, WorkerModels, WorkerSpec},
    },
    routers::grpc::client::GrpcClient,
};

/// Builder for creating BasicWorker instances with fluent API.
///
/// Internally stores a [`WorkerSpec`] for identity/config fields.
/// Callers with a pre-built `WorkerSpec` can use [`from_spec()`](Self::from_spec).
pub struct BasicWorkerBuilder {
    spec: WorkerSpec,
    health_endpoint: String,
    circuit_breaker_config: CircuitBreakerConfig,
    grpc_client: Option<GrpcClient>,
}

impl BasicWorkerBuilder {
    /// Create a new builder with only the URL (uses default WorkerSpec)
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            spec: WorkerSpec::new(url),
            health_endpoint: "/health".to_string(),
            circuit_breaker_config: CircuitBreakerConfig::default(),
            grpc_client: None,
        }
    }

    /// Create a builder from an existing WorkerSpec.
    pub fn from_spec(spec: WorkerSpec) -> Self {
        Self {
            spec,
            health_endpoint: "/health".to_string(),
            circuit_breaker_config: CircuitBreakerConfig::default(),
            grpc_client: None,
        }
    }

    /// Create a new builder with URL and worker type (for backwards compatibility)
    pub fn new_with_type(url: impl Into<String>, worker_type: WorkerType) -> Self {
        let mut spec = WorkerSpec::new(url);
        spec.worker_type = worker_type;
        Self {
            spec,
            health_endpoint: "/health".to_string(),
            circuit_breaker_config: CircuitBreakerConfig::default(),
            grpc_client: None,
        }
    }

    /// Set the bootstrap port (for prefill workers in PD disaggregation)
    pub fn bootstrap_port(mut self, port: Option<u16>) -> Self {
        self.spec.bootstrap_port = port;
        self
    }

    /// Set the API key
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.spec.api_key = Some(api_key.into());
        self
    }

    /// Set the worker type (Regular, Prefill, or Decode)
    pub fn worker_type(mut self, worker_type: WorkerType) -> Self {
        self.spec.worker_type = worker_type;
        self
    }

    /// Set the connection mode (HTTP or gRPC)
    pub fn connection_mode(mut self, mode: ConnectionMode) -> Self {
        self.spec.connection_mode = mode;
        self
    }

    /// Set the runtime type (SGLang or vLLM)
    pub fn runtime_type(mut self, runtime_type: RuntimeType) -> Self {
        self.spec.runtime_type = runtime_type;
        self
    }

    /// Set labels for worker identification
    pub fn labels(mut self, labels: HashMap<String, String>) -> Self {
        self.spec.labels = labels;
        self
    }

    /// Add a single label
    pub fn label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.spec.labels.insert(key.into(), value.into());
        self
    }

    /// Set health check configuration (protocol-level fields).
    pub fn health_config(mut self, config: HealthCheckConfig) -> Self {
        self.spec.health = config;
        self
    }

    /// Set health check endpoint path (internal-only, from router config).
    pub fn health_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.health_endpoint = endpoint.into();
        self
    }

    /// Set circuit breaker configuration
    pub fn circuit_breaker_config(mut self, config: CircuitBreakerConfig) -> Self {
        self.circuit_breaker_config = config;
        self
    }

    /// Set gRPC client for gRPC workers
    pub fn grpc_client(mut self, client: GrpcClient) -> Self {
        self.grpc_client = Some(client);
        self
    }

    /// Set KV connector type (e.g., "MooncakeConnector", "NixlConnector")
    pub fn kv_connector(mut self, connector: impl Into<String>) -> Self {
        self.spec.kv_connector = Some(connector.into());
        self
    }

    /// Set KV role (e.g., "kv_producer", "kv_consumer", "kv_both")
    pub fn kv_role(mut self, role: impl Into<String>) -> Self {
        self.spec.kv_role = Some(role.into());
        self
    }

    /// Set worker priority (higher value = higher priority)
    pub fn priority(mut self, priority: u32) -> Self {
        self.spec.priority = priority;
        self
    }

    /// Set worker cost factor (baseline = 1.0)
    pub fn cost(mut self, cost: f32) -> Self {
        self.spec.cost = cost;
        self
    }

    /// Set models this worker can serve
    pub fn models(mut self, models: impl Into<WorkerModels>) -> Self {
        self.spec.models = models.into();
        self
    }

    /// Set a single model this worker can serve
    pub fn model(mut self, model: ModelCard) -> Self {
        self.spec.models = WorkerModels::Single(Box::new(model));
        self
    }

    /// Override the URL (used internally by DPAwareWorkerBuilder)
    fn url(mut self, url: impl Into<String>) -> Self {
        self.spec.url = url.into();
        self
    }

    /// Build the BasicWorker instance
    pub fn build(mut self) -> BasicWorker {
        use std::sync::{
            atomic::{AtomicBool, AtomicUsize},
            Arc, RwLock as StdRwLock,
        };

        use tokio::sync::OnceCell;

        // Derive bootstrap_host from URL at construction time
        self.spec.bootstrap_host = parse_bootstrap_host(&self.spec.url);

        let metadata = WorkerMetadata {
            spec: self.spec,
            health_endpoint: self.health_endpoint,
            default_model_type: ModelType::LLM,
        };

        // Use OnceCell for lock-free gRPC client access after initialization
        let grpc_client = Arc::new(match self.grpc_client {
            Some(client) => {
                let cell = OnceCell::new();
                // Pre-set the client if provided (blocking set is fine during construction)
                cell.set(Arc::new(client)).ok();
                cell
            }
            None => OnceCell::new(),
        });

        let healthy = true;
        Metrics::set_worker_health(&metadata.spec.url, healthy);

        BasicWorker {
            load_counter: Arc::new(AtomicUsize::new(0)),
            worker_routing_key_load: Arc::new(WorkerRoutingKeyLoad::new(&metadata.spec.url)),
            processed_counter: Arc::new(AtomicUsize::new(0)),
            healthy: Arc::new(AtomicBool::new(healthy)),
            consecutive_failures: Arc::new(AtomicUsize::new(0)),
            consecutive_successes: Arc::new(AtomicUsize::new(0)),
            circuit_breaker: CircuitBreaker::with_config_and_label(
                self.circuit_breaker_config,
                metadata.spec.url.clone(),
            ),
            metadata,
            grpc_client,
            models_override: Arc::new(StdRwLock::new(None)),
        }
    }
}

/// Parse bootstrap hostname from a URL, falling back to "localhost".
fn parse_bootstrap_host(url: &str) -> String {
    match url::Url::parse(url) {
        Ok(parsed) => parsed.host_str().unwrap_or("localhost").to_string(),
        Err(_) if !url.contains("://") => match url::Url::parse(&format!("http://{}", url)) {
            Ok(parsed) => parsed.host_str().unwrap_or("localhost").to_string(),
            Err(_) => {
                tracing::warn!("Failed to parse URL '{}', defaulting to localhost", url);
                "localhost".to_string()
            }
        },
        Err(_) => {
            tracing::warn!("Failed to parse URL '{}', defaulting to localhost", url);
            "localhost".to_string()
        }
    }
}

/// Builder for creating DPAwareWorker instances with fluent API.
///
/// Delegates to [`BasicWorkerBuilder`] for all shared configuration,
/// adding only DP-specific fields (base_url, dp_rank, dp_size).
pub struct DPAwareWorkerBuilder {
    inner: BasicWorkerBuilder,
    base_url: String,
    dp_rank: usize,
    dp_size: usize,
}

impl DPAwareWorkerBuilder {
    pub fn new(base_url: impl Into<String>, dp_rank: usize, dp_size: usize) -> Self {
        let base_url = base_url.into();
        Self {
            inner: BasicWorkerBuilder::new(&base_url),
            base_url,
            dp_rank,
            dp_size,
        }
    }

    pub fn new_with_type(
        base_url: impl Into<String>,
        dp_rank: usize,
        dp_size: usize,
        worker_type: WorkerType,
    ) -> Self {
        let base_url = base_url.into();
        Self {
            inner: BasicWorkerBuilder::new_with_type(&base_url, worker_type),
            base_url,
            dp_rank,
            dp_size,
        }
    }

    // Delegate all shared setter methods to inner BasicWorkerBuilder
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.inner = self.inner.api_key(api_key);
        self
    }
    pub fn worker_type(mut self, worker_type: WorkerType) -> Self {
        self.inner = self.inner.worker_type(worker_type);
        self
    }
    pub fn connection_mode(mut self, mode: ConnectionMode) -> Self {
        self.inner = self.inner.connection_mode(mode);
        self
    }
    pub fn runtime_type(mut self, runtime_type: RuntimeType) -> Self {
        self.inner = self.inner.runtime_type(runtime_type);
        self
    }
    pub fn labels(mut self, labels: HashMap<String, String>) -> Self {
        self.inner = self.inner.labels(labels);
        self
    }
    pub fn label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.inner = self.inner.label(key, value);
        self
    }
    pub fn health_config(mut self, config: HealthCheckConfig) -> Self {
        self.inner = self.inner.health_config(config);
        self
    }
    pub fn health_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.inner = self.inner.health_endpoint(endpoint);
        self
    }
    pub fn circuit_breaker_config(mut self, config: CircuitBreakerConfig) -> Self {
        self.inner = self.inner.circuit_breaker_config(config);
        self
    }
    pub fn grpc_client(mut self, client: GrpcClient) -> Self {
        self.inner = self.inner.grpc_client(client);
        self
    }
    pub fn models(mut self, models: impl Into<WorkerModels>) -> Self {
        self.inner = self.inner.models(models);
        self
    }
    pub fn model(mut self, model: ModelCard) -> Self {
        self.inner = self.inner.model(model);
        self
    }
    pub fn kv_connector(mut self, connector: impl Into<String>) -> Self {
        self.inner = self.inner.kv_connector(connector);
        self
    }
    pub fn kv_role(mut self, role: impl Into<String>) -> Self {
        self.inner = self.inner.kv_role(role);
        self
    }
    pub fn priority(mut self, priority: u32) -> Self {
        self.inner = self.inner.priority(priority);
        self
    }
    pub fn cost(mut self, cost: f32) -> Self {
        self.inner = self.inner.cost(cost);
        self
    }
    pub fn bootstrap_port(mut self, port: Option<u16>) -> Self {
        self.inner = self.inner.bootstrap_port(port);
        self
    }

    pub fn build(self) -> DPAwareWorker {
        let worker_url = format!("{}@{}", self.base_url, self.dp_rank);
        let base_worker = self.inner.url(worker_url).build();
        DPAwareWorker::with_base_worker(base_worker, self.base_url, self.dp_rank, self.dp_size)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::core::worker::Worker;

    #[test]
    fn test_basic_worker_builder_minimal() {
        let worker = BasicWorkerBuilder::new("http://localhost:8080").build();

        assert_eq!(worker.url(), "http://localhost:8080");
        assert_eq!(worker.worker_type(), &WorkerType::Regular);
        assert_eq!(worker.connection_mode(), &ConnectionMode::Http);
        assert!(worker.is_healthy());
    }

    #[test]
    fn test_basic_worker_builder_with_type() {
        let worker = BasicWorkerBuilder::new("http://localhost:8080")
            .worker_type(WorkerType::Decode)
            .build();

        assert_eq!(worker.url(), "http://localhost:8080");
        assert_eq!(worker.worker_type(), &WorkerType::Decode);
        assert_eq!(worker.connection_mode(), &ConnectionMode::Http);
        assert!(worker.is_healthy());
    }

    #[test]
    fn test_basic_worker_builder_full() {
        let mut labels = HashMap::new();
        labels.insert("env".to_string(), "prod".to_string());
        labels.insert("region".to_string(), "us-east".to_string());

        let health_config = HealthCheckConfig {
            timeout_secs: 30,
            check_interval_secs: 60,
            failure_threshold: 3,
            success_threshold: 2,
            disable_health_check: false,
        };

        let cb_config = CircuitBreakerConfig {
            failure_threshold: 10,
            success_threshold: 5,
            timeout_duration: Duration::from_millis(2000),
            window_duration: Duration::from_millis(30000),
        };

        let worker = BasicWorkerBuilder::new("http://localhost:8080")
            .worker_type(WorkerType::Prefill)
            .connection_mode(ConnectionMode::Grpc)
            .labels(labels.clone())
            .health_config(health_config.clone())
            .health_endpoint("/health")
            .circuit_breaker_config(cb_config)
            .build();

        assert_eq!(worker.url(), "http://localhost:8080");
        assert_eq!(worker.worker_type(), &WorkerType::Prefill);
        assert_eq!(worker.connection_mode(), &ConnectionMode::Grpc);
        assert_eq!(worker.metadata().spec.labels, labels);
        assert_eq!(worker.metadata().health_endpoint, "/health");
        assert_eq!(
            worker.metadata().spec.health.timeout_secs,
            health_config.timeout_secs
        );
        assert_eq!(
            worker.metadata().spec.health.check_interval_secs,
            health_config.check_interval_secs
        );
        assert_eq!(
            worker.metadata().spec.health.failure_threshold,
            health_config.failure_threshold
        );
        assert_eq!(
            worker.metadata().spec.health.success_threshold,
            health_config.success_threshold
        );
    }

    #[test]
    fn test_basic_worker_builder_with_single_label() {
        let worker = BasicWorkerBuilder::new("http://localhost:8080")
            .worker_type(WorkerType::Decode)
            .label("env", "staging")
            .label("version", "v1.2.3")
            .build();

        assert_eq!(
            worker.metadata().spec.labels.get("env"),
            Some(&"staging".to_string())
        );
        assert_eq!(
            worker.metadata().spec.labels.get("version"),
            Some(&"v1.2.3".to_string())
        );
    }

    #[test]
    fn test_dp_aware_worker_builder_minimal() {
        let worker = DPAwareWorkerBuilder::new("http://localhost:8080", 2, 8).build();

        assert_eq!(worker.url(), "http://localhost:8080@2");
        assert_eq!(worker.dp_rank(), Some(2));
        assert_eq!(worker.dp_size(), Some(8));
        assert_eq!(worker.worker_type(), &WorkerType::Regular);
    }

    #[test]
    fn test_dp_aware_worker_builder_full() {
        let mut labels = HashMap::new();
        labels.insert("cluster".to_string(), "main".to_string());

        let health_config = HealthCheckConfig {
            timeout_secs: 20,
            check_interval_secs: 45,
            failure_threshold: 5,
            success_threshold: 3,
            disable_health_check: false,
        };

        let worker = DPAwareWorkerBuilder::new("http://localhost:8080", 3, 16)
            .worker_type(WorkerType::Prefill)
            .bootstrap_port(Some(9090))
            .connection_mode(ConnectionMode::Http)
            .labels(labels.clone())
            .health_config(health_config.clone())
            .health_endpoint("/status")
            .api_key("test_api_key")
            .build();

        assert_eq!(worker.url(), "http://localhost:8080@3");
        assert_eq!(worker.dp_rank(), Some(3));
        assert_eq!(worker.dp_size(), Some(16));
        assert_eq!(worker.metadata().spec.labels, labels);
        assert_eq!(worker.metadata().health_endpoint, "/status");
        assert_eq!(
            worker.metadata().spec.health.timeout_secs,
            health_config.timeout_secs
        );
        assert_eq!(
            worker.metadata().spec.health.check_interval_secs,
            health_config.check_interval_secs
        );
        assert_eq!(
            worker.metadata().spec.health.failure_threshold,
            health_config.failure_threshold
        );
        assert_eq!(
            worker.metadata().spec.health.success_threshold,
            health_config.success_threshold
        );
    }

    #[test]
    fn test_dp_aware_worker_with_grpc() {
        let worker = DPAwareWorkerBuilder::new("grpc://cluster.local", 1, 4)
            .worker_type(WorkerType::Decode)
            .connection_mode(ConnectionMode::Grpc)
            .label("transport", "grpc")
            .build();

        assert_eq!(worker.url(), "grpc://cluster.local@1");
        assert_eq!(worker.dp_rank(), Some(1));
        assert_eq!(worker.dp_size(), Some(4));
        assert_eq!(worker.worker_type(), &WorkerType::Decode);
        assert_eq!(worker.connection_mode(), &ConnectionMode::Grpc);
        assert_eq!(
            worker.metadata().spec.labels.get("transport"),
            Some(&"grpc".to_string())
        );
    }
}
