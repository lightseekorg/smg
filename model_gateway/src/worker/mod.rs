//! Worker domain — identity, registry, health, resilience, monitoring, service.

pub mod basic;
pub mod body;
pub mod builder;
pub mod circuit_breaker;
pub mod error;
pub mod event;
pub(crate) mod health_checker;
pub mod http_client;
pub mod kv_event_monitor;
pub mod load;
pub mod manager;
pub mod metadata;
pub mod metrics_aggregator;
pub mod registry;
pub mod resilience;
pub mod retry;
pub mod service;
pub mod token_bucket;
pub mod traits;

// Re-export commonly used types for convenience
pub use basic::{BasicWorker, DEFAULT_BOOTSTRAP_PORT, MOONCAKE_CONNECTOR};
pub use body::{AttachedBody, WorkerLoadGuard};
pub use builder::BasicWorkerBuilder;
pub use circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
pub use error::{WorkerError, WorkerResult};
pub use http_client::build_worker_http_client;
pub use kv_event_monitor::KvEventMonitor;
pub use load::WorkerLoadManager;
pub use manager::{LoadMonitor, WorkerManager};
pub use metadata::WorkerMetadata;
// Re-export UNKNOWN_MODEL_ID from protocols
pub use openai_protocol::UNKNOWN_MODEL_ID;
pub use openai_protocol::{
    model_card::ModelCard,
    model_type::{Endpoint, ModelType},
    worker::{ProviderType, WorkerGroupKey},
};
pub use registry::{HashRing, WorkerRegistry};
pub use resilience::{resolve_resilience, ResolvedResilience, DEFAULT_RETRYABLE_STATUS_CODES};
pub use retry::{is_retryable_status, RetryExecutor};
pub use service::WorkerService;
pub use traits::{ConnectionMode, RuntimeType, Worker, WorkerType};
