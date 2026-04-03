//! Core abstractions for the SGLang router
//!
//! This module contains the fundamental types and traits used throughout the router:
//! - Worker trait and implementations
//! - Model types and endpoint definitions
//! - Error types
//! - Circuit breaker for reliability
//! - Token buckets for rate limiting
//! - Workflow steps for multi-step operations
//! - Common utilities

// Re-export UNKNOWN_MODEL_ID from protocols for use throughout core
pub use openai_protocol::UNKNOWN_MODEL_ID;

pub mod circuit_breaker;
pub mod error;
pub mod http_client;
pub mod job_queue;
pub mod kv_event_monitor;
pub mod metrics_aggregator;
pub mod resilience;
pub mod retry;
pub mod steps;
pub mod token_bucket;
pub mod worker;
pub mod worker_builder;
pub mod worker_event;
pub mod worker_manager;
pub mod worker_registry;
pub mod worker_service;

// Re-export commonly used types for convenience
pub use circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
pub use error::{WorkerError, WorkerResult};
pub use http_client::build_worker_http_client;
pub use job_queue::{Job, JobQueue, JobQueueConfig};
pub use kv_event_monitor::KvEventMonitor;
pub use openai_protocol::{
    model_card::ModelCard,
    model_type::{Endpoint, ModelType},
    worker::{ProviderType, WorkerGroupKey},
};
pub use resilience::{resolve_resilience, ResolvedResilience, DEFAULT_RETRYABLE_STATUS_CODES};
pub use retry::{is_retryable_status, RetryExecutor};
pub use worker::{
    AttachedBody, BasicWorker, ConnectionMode, RuntimeType, Worker, WorkerLoadGuard, WorkerType,
    DEFAULT_BOOTSTRAP_PORT, MOONCAKE_CONNECTOR,
};
pub use worker_builder::BasicWorkerBuilder;
pub use worker_manager::{LoadMonitor, WorkerManager};
pub use worker_registry::{HashRing, WorkerRegistry};
pub use worker_service::WorkerService;
