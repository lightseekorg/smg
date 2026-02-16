use std::collections::HashMap;

use openai_protocol::{
    model_card::ModelCard,
    model_type::ModelType,
    worker::{HealthCheckConfig, WorkerModels, WorkerSpec},
};

use super::{
    circuit_breaker::{CircuitBreaker, CircuitBreakerConfig},
    worker::{
        BasicWorker, ConnectionMode, RuntimeType, WorkerMetadata, WorkerRoutingKeyLoad, WorkerType,
    },
};
use crate::{observability::metrics::Metrics, routers::grpc::client::GrpcClient};

/// Builder for creating BasicWorker instances with fluent API.
///
/// Internally stores a [`WorkerSpec`] for identity/config fields.
/// Callers with a pre-built `WorkerSpec` can use [`from_spec()`](Self::from_spec).
pub struct BasicWorkerBuilder {
    spec: WorkerSpec,
    /// Resolved health config (router defaults + per-worker overrides).
    /// If not set, falls back to `HealthCheckConfig::default()`.
    health_config: Option<HealthCheckConfig>,
    health_endpoint: String,
    circuit_breaker_config: CircuitBreakerConfig,
    grpc_client: Option<GrpcClient>,
    dp_rank: Option<usize>,
    dp_size: Option<usize>,
    dp_base_url: Option<String>,
}

impl BasicWorkerBuilder {
    /// Create a new builder with only the URL (uses default WorkerSpec)
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            spec: WorkerSpec::new(url),
            health_config: None,
            health_endpoint: "/health".to_string(),
            circuit_breaker_config: CircuitBreakerConfig::default(),
            grpc_client: None,
            dp_rank: None,
            dp_size: None,
            dp_base_url: None,
        }
    }

    /// Create a builder from an existing WorkerSpec.
    pub fn from_spec(spec: WorkerSpec) -> Self {
        Self {
            spec,
            health_config: None,
            health_endpoint: "/health".to_string(),
            circuit_breaker_config: CircuitBreakerConfig::default(),
            grpc_client: None,
            dp_rank: None,
            dp_size: None,
            dp_base_url: None,
        }
    }

    /// Create a new builder with URL and worker type (for backwards compatibility)
    pub fn new_with_type(url: impl Into<String>, worker_type: WorkerType) -> Self {
        let mut spec = WorkerSpec::new(url);
        spec.worker_type = worker_type;
        Self {
            spec,
            health_config: None,
            health_endpoint: "/health".to_string(),
            circuit_breaker_config: CircuitBreakerConfig::default(),
            grpc_client: None,
            dp_rank: None,
            dp_size: None,
            dp_base_url: None,
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

    /// Set the resolved health check configuration.
    ///
    /// This is the fully-resolved config (router defaults + per-worker overrides)
    /// stored on `WorkerMetadata` for runtime use.
    pub fn health_config(mut self, config: HealthCheckConfig) -> Self {
        self.health_config = Some(config);
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

    /// Configure data-parallel routing.
    /// Captures the current URL as the base URL, then formats it as `{base}@{rank}`.
    pub fn dp_config(mut self, rank: usize, size: usize) -> Self {
        self.dp_base_url = Some(self.spec.url.clone());
        self.spec.url = format!("{}@{}", self.dp_base_url.as_ref().unwrap(), rank);
        self.dp_rank = Some(rank);
        self.dp_size = Some(size);
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

        // Resolve health config: use explicit config if set, otherwise
        // apply per-worker overrides from spec.health to defaults.
        let health_config = self
            .health_config
            .unwrap_or_else(|| self.spec.health.apply_to(&HealthCheckConfig::default()));

        let metadata = WorkerMetadata {
            spec: self.spec,
            health_config,
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
            dp_rank: self.dp_rank,
            dp_size: self.dp_size,
            dp_base_url: self.dp_base_url,
        }
    }
}

/// Parse bootstrap hostname from a URL, falling back to "localhost".
///
/// Handles DP-aware URLs like `http://host:8080@3` by stripping the `@rank`
/// suffix before parsing, since `@` is otherwise interpreted as a userinfo
/// delimiter per RFC 3986.
fn parse_bootstrap_host(url: &str) -> String {
    // Strip DP rank suffix (e.g., "http://host:8080@3" -> "http://host:8080")
    let clean_url = match url.rfind('@') {
        Some(at_pos)
            if !url[at_pos + 1..].is_empty()
                && url[at_pos + 1..].chars().all(|c| c.is_ascii_digit()) =>
        {
            &url[..at_pos]
        }
        _ => url,
    };

    // Try parsing as-is first. If the URL lacks a scheme (e.g., "worker1:8080"),
    // Url::parse may treat the host as a scheme — detect this via missing host_str()
    // and fall back to prefixing "http://".
    let try_parse = |u: &str| -> Option<String> {
        url::Url::parse(u)
            .ok()
            .and_then(|p| p.host_str().map(|h| h.to_string()))
    };

    if let Some(host) = try_parse(clean_url) {
        host
    } else if !clean_url.contains("://") {
        try_parse(&format!("http://{}", clean_url)).unwrap_or_else(|| {
            tracing::warn!("Failed to parse URL '{}', defaulting to localhost", url);
            "localhost".to_string()
        })
    } else {
        tracing::warn!("Failed to parse URL '{}', defaulting to localhost", url);
        "localhost".to_string()
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
            worker.metadata().health_config.timeout_secs,
            health_config.timeout_secs
        );
        assert_eq!(
            worker.metadata().health_config.check_interval_secs,
            health_config.check_interval_secs
        );
        assert_eq!(
            worker.metadata().health_config.failure_threshold,
            health_config.failure_threshold
        );
        assert_eq!(
            worker.metadata().health_config.success_threshold,
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
        let worker = BasicWorkerBuilder::new("http://localhost:8080")
            .dp_config(2, 8)
            .build();

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

        let worker = BasicWorkerBuilder::new("http://localhost:8080")
            .dp_config(3, 16)
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
            worker.metadata().health_config.timeout_secs,
            health_config.timeout_secs
        );
        assert_eq!(
            worker.metadata().health_config.check_interval_secs,
            health_config.check_interval_secs
        );
        assert_eq!(
            worker.metadata().health_config.failure_threshold,
            health_config.failure_threshold
        );
        assert_eq!(
            worker.metadata().health_config.success_threshold,
            health_config.success_threshold
        );
    }

    #[test]
    fn test_dp_aware_worker_with_grpc() {
        let worker = BasicWorkerBuilder::new("grpc://cluster.local")
            .dp_config(1, 4)
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

    #[test]
    fn test_parse_bootstrap_host_normal_url() {
        assert_eq!(parse_bootstrap_host("http://worker1:8080"), "worker1");
        assert_eq!(parse_bootstrap_host("https://10.0.0.5:443"), "10.0.0.5");
        assert_eq!(
            parse_bootstrap_host("grpc://cluster.local"),
            "cluster.local"
        );
    }

    #[test]
    fn test_parse_bootstrap_host_dp_aware_url() {
        // DP-aware URLs use @rank suffix — must extract host, not rank
        assert_eq!(parse_bootstrap_host("http://worker1:8080@0"), "worker1");
        assert_eq!(parse_bootstrap_host("http://worker1:8080@3"), "worker1");
        assert_eq!(
            parse_bootstrap_host("grpc://prefill.local@7"),
            "prefill.local"
        );
    }

    #[test]
    fn test_parse_bootstrap_host_bare_host() {
        assert_eq!(parse_bootstrap_host("worker1:8080"), "worker1");
        assert_eq!(parse_bootstrap_host("localhost"), "localhost");
    }

    #[test]
    fn test_dp_aware_worker_bootstrap_host() {
        let worker = BasicWorkerBuilder::new("http://prefill1:8080")
            .dp_config(3, 8)
            .worker_type(WorkerType::Prefill)
            .build();

        // bootstrap_host should be "prefill1", not "3"
        assert_eq!(worker.metadata().spec.bootstrap_host, "prefill1");
    }
}
