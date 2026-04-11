use std::{
    fmt,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, LazyLock,
    },
    time::Duration,
};

use arc_swap::ArcSwap;
use async_trait::async_trait;
use openai_protocol::{
    model_card::ModelCard,
    worker::{ConnectionMode, WorkerModels, WorkerType},
};
use tokio::{sync::OnceCell, time};

use super::{
    metadata::{WorkerMetadata, WorkerRoutingKeyLoad},
    traits::{Worker, WorkerTypeExt},
    CircuitBreaker, ResolvedResilience, WorkerError, WorkerResult,
};
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

#[expect(
    clippy::expect_used,
    reason = "LazyLock static initialization — reqwest::Client::build() only fails on TLS backend misconfiguration which is unrecoverable"
)]
static WORKER_CLIENT: LazyLock<reqwest::Client> = LazyLock::new(|| {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(DEFAULT_WORKER_HTTP_TIMEOUT_SECS))
        .build()
        .expect("Failed to create worker HTTP client")
});

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
    /// Runtime-mutable models override (for lazy discovery).
    /// When not `Wildcard`, overrides metadata.models for routing decisions.
    /// Uses `ArcSwap` for lock-free reads on the hot path (`supports_model`).
    pub models_override: Arc<ArcSwap<WorkerModels>>,
    /// Per-worker HTTP client with isolated connection pool.
    pub http_client: reqwest::Client,
    /// Resolved resilience config (retry + circuit breaker settings).
    pub resilience: ResolvedResilience,
}

impl fmt::Debug for BasicWorker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BasicWorker")
            .field("metadata", &self.metadata)
            .field("healthy", &self.healthy.load(Ordering::Relaxed))
            .field("circuit_breaker", &self.circuit_breaker)
            .field("grpc_client", &"<OnceCell>")
            .finish()
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

    fn api_key(&self) -> Option<&String> {
        self.metadata.spec.api_key.as_ref()
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
                reason: format!("Health check failed (consecutive failures: {failures})"),
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

    fn resilience(&self) -> &ResolvedResilience {
        &self.resilience
    }

    fn http_client(&self) -> &reqwest::Client {
        &self.http_client
    }

    fn models(&self) -> Vec<ModelCard> {
        let overridden = self.models_override.load();
        let source = if overridden.is_wildcard() {
            self.metadata.spec.models.all()
        } else {
            overridden.all()
        };
        source.to_vec()
    }

    fn supports_model(&self, model_id: &str) -> bool {
        let overridden = self.models_override.load();
        if !overridden.is_wildcard() {
            return overridden.supports(model_id);
        }
        self.metadata.supports_model(model_id)
    }

    fn set_models(&self, models: Vec<ModelCard>) {
        tracing::debug!(
            "Setting {} models for worker {} via lazy discovery",
            models.len(),
            self.metadata.spec.url
        );
        self.models_override
            .store(Arc::new(WorkerModels::from(models)));
    }

    fn has_models_discovered(&self) -> bool {
        !self.models_override.load().is_wildcard() || !self.metadata.spec.models.is_wildcard()
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
                                    reason: format!("Failed to connect to gRPC server: {e}"),
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

#[cfg(test)]
mod tests {
    use std::{sync::Arc, thread, time::Duration};

    use openai_protocol::{
        model_card::ModelCard,
        worker::{HealthCheckConfig, WorkerModels, WorkerSpec},
    };
    use tokio::time;

    use super::*;
    use crate::worker::{
        body::WorkerLoadGuard,
        circuit_breaker::{CircuitBreakerConfig, CircuitState},
        error::WorkerError,
        metadata::{WorkerMetadata, WorkerRoutingKeyLoad},
        traits::{Worker, WorkerType},
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
        let config = HealthCheckConfig::default();
        assert_eq!(config.timeout_secs, 30);
        assert_eq!(config.check_interval_secs, 60);
        assert_eq!(config.failure_threshold, 3);
        assert_eq!(config.success_threshold, 2);
        assert!(!config.disable_health_check);
    }

    #[test]
    fn test_health_config_custom() {
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

        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .labels(labels.clone())
            .build();

        assert_eq!(worker.metadata().spec.labels, labels);
    }

    #[test]
    fn test_worker_with_health_config() {
        let custom_config = HealthCheckConfig {
            timeout_secs: 15,
            check_interval_secs: 45,
            failure_threshold: 4,
            success_threshold: 2,
            disable_health_check: false,
        };

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
        let worker = BasicWorkerBuilder::new("http://worker1:8080")
            .worker_type(WorkerType::Regular)
            .build();
        assert_eq!(worker.url(), "http://worker1:8080");
    }

    #[test]
    fn test_worker_type_getter() {
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
        let worker = Arc::new(
            BasicWorkerBuilder::new("http://test:8080")
                .worker_type(WorkerType::Regular)
                .build(),
        );

        let mut handles = vec![];

        for _ in 0..100 {
            let worker_clone = Arc::clone(&worker);
            #[expect(
                clippy::disallowed_methods,
                reason = "Test helper: short-lived tasks joined before test ends"
            )]
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
            #[expect(
                clippy::disallowed_methods,
                reason = "Test helper: short-lived tasks joined before test ends"
            )]
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
        let worker = Arc::new(
            BasicWorkerBuilder::new("http://test:8080")
                .worker_type(WorkerType::Regular)
                .build(),
        );

        let mut handles = vec![];

        for i in 0..100 {
            let worker_clone = Arc::clone(&worker);
            #[expect(
                clippy::disallowed_methods,
                reason = "Test helper: short-lived tasks joined before test ends"
            )]
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
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .build();

        // Health check should fail since there's no actual server
        let result = worker.check_health_async().await;
        assert!(result.is_err());
    }

    #[test]
    #[expect(clippy::print_stderr)]
    fn test_load_counter_performance() {
        use std::time::Instant;

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
        eprintln!("Load counter operations per second: {ops_per_sec:.0}");

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
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .build();

        assert!(worker.is_available());
        assert_eq!(worker.circuit_breaker().state(), CircuitState::Closed);

        worker.record_outcome(503);
        worker.record_outcome(503);

        assert!(worker.is_available());

        worker.record_outcome(503);
        worker.record_outcome(503);
        worker.record_outcome(503);

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

        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .circuit_breaker_config(config)
            .build();

        worker.record_outcome(503);
        assert!(worker.is_available());
        worker.record_outcome(503);
        assert!(!worker.is_available());

        thread::sleep(Duration::from_millis(150));

        assert!(worker.is_available());
        assert_eq!(worker.circuit_breaker().state(), CircuitState::HalfOpen);

        worker.record_outcome(200);
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
            dp_worker.record_outcome(503);
        }

        assert!(!dp_worker.is_available());
        assert_eq!(dp_worker.circuit_breaker().state(), CircuitState::Open);
    }

    #[tokio::test]
    async fn test_mixed_worker_types() {
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

    #[test]
    fn test_worker_metadata_empty_models_accepts_all() {
        let metadata = WorkerMetadata {
            spec: WorkerSpec::new("http://test:8080"),
            health_config: HealthCheckConfig::default(),
            health_endpoint: "/health".to_string(),
        };

        assert!(metadata.supports_model("any-model"));
        assert!(metadata.supports_model("gpt-4"));
        assert!(metadata.supports_model("llama-3.1"));
    }

    #[test]
    fn test_worker_metadata_find_model() {
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
        };

        assert!(metadata.find_model("meta-llama/Llama-3.1-8B").is_some());
        assert!(metadata.find_model("gpt-4o").is_some());
        assert!(metadata.find_model("llama-3.1-8b").is_some());
        assert!(metadata.find_model("llama3.1").is_some());
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

    #[test]
    fn test_lazy_discovered_models_override_wildcard() {
        let worker = BasicWorkerBuilder::new("http://test:8080").build();

        assert!(worker.models().is_empty());
        assert!(!worker.has_models_discovered());
        assert!(worker.supports_model("gpt-4o-mini"));

        let discovered = vec![
            ModelCard::new("gpt-4o-mini"),
            ModelCard::new("text-embedding-3-small"),
        ];
        worker.set_models(discovered);

        let ids: Vec<String> = worker.models().into_iter().map(|m| m.id).collect();
        assert_eq!(ids, vec!["gpt-4o-mini", "text-embedding-3-small"]);
        assert!(worker.supports_model("gpt-4o-mini"));
        assert!(worker.supports_model("text-embedding-3-small"));
        assert!(!worker.supports_model("non-existent-model"));
        assert!(worker.has_models_discovered());
    }
}
