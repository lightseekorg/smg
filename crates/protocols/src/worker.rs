//! Canonical worker types and identity.
//!
//! This module defines the single source of truth for worker identity, type
//! enums, and core configuration. These types are shared across API
//! request/response boundaries and internal runtime state.

use std::collections::HashMap;

#[cfg(feature = "axum")]
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
#[cfg(feature = "axum")]
use serde_json::{json, Value};

use super::model_card::ModelCard;

// ── Default value constants ──────────────────────────────────────────

pub const DEFAULT_WORKER_PRIORITY: u32 = 50;
pub const DEFAULT_WORKER_COST: f32 = 1.0;

// ── Enums ────────────────────────────────────────────────────────────

/// Worker type classification.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default, schemars::JsonSchema,
)]
#[serde(rename_all = "lowercase")]
pub enum WorkerType {
    /// Regular worker for standard routing.
    #[default]
    Regular,
    /// Prefill worker for PD disaggregated mode.
    Prefill,
    /// Decode worker for PD disaggregated mode.
    Decode,
}

impl std::fmt::Display for WorkerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WorkerType::Regular => write!(f, "regular"),
            WorkerType::Prefill => write!(f, "prefill"),
            WorkerType::Decode => write!(f, "decode"),
        }
    }
}

impl std::str::FromStr for WorkerType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.eq_ignore_ascii_case("regular") {
            Ok(WorkerType::Regular)
        } else if s.eq_ignore_ascii_case("prefill") {
            Ok(WorkerType::Prefill)
        } else if s.eq_ignore_ascii_case("decode") {
            Ok(WorkerType::Decode)
        } else {
            Err(format!("Unknown worker type: {s}"))
        }
    }
}

/// Connection mode for worker communication.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default, schemars::JsonSchema,
)]
#[serde(rename_all = "lowercase")]
pub enum ConnectionMode {
    /// HTTP/REST connection.
    #[default]
    Http,
    /// gRPC connection.
    Grpc,
}

impl std::fmt::Display for ConnectionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConnectionMode::Http => write!(f, "http"),
            ConnectionMode::Grpc => write!(f, "grpc"),
        }
    }
}

/// Composite key identifying a group of workers with the same characteristics.
///
/// Groups workers by `(model_id, worker_type, connection_mode)` — the natural
/// partitioning used for metrics, load monitoring, and policy management.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WorkerGroupKey {
    pub model_id: String,
    pub worker_type: WorkerType,
    pub connection_mode: ConnectionMode,
}

impl std::fmt::Display for WorkerGroupKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}:{}:{}",
            self.model_id, self.worker_type, self.connection_mode
        )
    }
}

/// Runtime implementation type for workers.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default, schemars::JsonSchema,
)]
#[serde(rename_all = "lowercase")]
pub enum RuntimeType {
    /// No runtime type specified — the backend will be auto-detected.
    #[default]
    Unspecified,
    /// SGLang runtime.
    Sglang,
    /// vLLM runtime.
    Vllm,
    /// TensorRT-LLM runtime.
    Trtllm,
    /// External OpenAI-compatible API (not local inference).
    External,
}

impl RuntimeType {
    /// Returns `true` when the caller supplied an explicit runtime type.
    pub fn is_specified(self) -> bool {
        !matches!(self, RuntimeType::Unspecified)
    }
}

impl std::fmt::Display for RuntimeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuntimeType::Unspecified => write!(f, "unspecified"),
            RuntimeType::Sglang => write!(f, "sglang"),
            RuntimeType::Vllm => write!(f, "vllm"),
            RuntimeType::Trtllm => write!(f, "trtllm"),
            RuntimeType::External => write!(f, "external"),
        }
    }
}

impl std::str::FromStr for RuntimeType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.eq_ignore_ascii_case("unspecified") {
            Ok(RuntimeType::Unspecified)
        } else if s.eq_ignore_ascii_case("sglang") {
            Ok(RuntimeType::Sglang)
        } else if s.eq_ignore_ascii_case("vllm") {
            Ok(RuntimeType::Vllm)
        } else if s.eq_ignore_ascii_case("trtllm") || s.eq_ignore_ascii_case("tensorrt-llm") {
            Ok(RuntimeType::Trtllm)
        } else if s.eq_ignore_ascii_case("external") {
            Ok(RuntimeType::External)
        } else {
            Err(format!("Unknown runtime type: {s}"))
        }
    }
}

/// Provider type for external API transformations.
///
/// Different providers have different API formats and requirements.
/// `None` (when used as `Option<ProviderType>`) means native/passthrough —
/// no transformation needed (local SGLang backends).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum ProviderType {
    /// OpenAI API — strip SGLang-specific fields.
    #[serde(alias = "openai")]
    OpenAI,
    /// xAI/Grok — special handling for input items.
    #[serde(alias = "xai", alias = "grok")]
    #[expect(
        clippy::upper_case_acronyms,
        reason = "xAI is a proper company name; XAI matches industry convention and existing serde aliases"
    )]
    XAI,
    /// Anthropic Claude — different API format.
    #[serde(alias = "anthropic", alias = "claude")]
    Anthropic,
    /// Google Gemini — special logprobs handling.
    #[serde(alias = "gemini", alias = "google")]
    Gemini,
    /// Custom provider with string identifier.
    #[serde(untagged)]
    Custom(String),
}

impl ProviderType {
    /// Get provider name as string.
    pub fn as_str(&self) -> &str {
        match self {
            Self::OpenAI => "openai",
            Self::XAI => "xai",
            Self::Anthropic => "anthropic",
            Self::Gemini => "gemini",
            Self::Custom(s) => s.as_str(),
        }
    }

    /// Detect provider from URL host.
    /// Returns `None` for URLs that don't match known providers or can't be parsed.
    pub fn from_url(url: &str) -> Option<Self> {
        let host = url::Url::parse(url).ok()?.host_str()?.to_lowercase();

        if host.ends_with("openai.com") {
            Some(Self::OpenAI)
        } else if host.ends_with("x.ai") {
            Some(Self::XAI)
        } else if host.ends_with("anthropic.com") {
            Some(Self::Anthropic)
        } else if host.ends_with("googleapis.com") {
            Some(Self::Gemini)
        } else {
            None
        }
    }

    /// Environment variable name for per-provider admin API key (model discovery).
    /// Returns `None` for `Custom` providers since there's no known env var.
    pub fn admin_key_env_var(&self) -> Option<&'static str> {
        match self {
            Self::OpenAI => Some("OPENAI_ADMIN_KEY"),
            Self::XAI => Some("XAI_ADMIN_KEY"),
            Self::Anthropic => Some("ANTHROPIC_ADMIN_KEY"),
            Self::Gemini => Some("GEMINI_ADMIN_KEY"),
            Self::Custom(_) => None,
        }
    }

    /// Whether this provider uses `x-api-key` header instead of `Authorization: Bearer`.
    pub fn uses_x_api_key(&self) -> bool {
        matches!(self, Self::Anthropic)
    }

    /// Detect provider from model name (heuristic fallback).
    /// Returns `None` for models that don't match known external providers.
    pub fn from_model_name(model: &str) -> Option<Self> {
        let model_lower = model.to_lowercase();
        if model_lower.starts_with("grok") {
            Some(Self::XAI)
        } else if model_lower.starts_with("gemini") {
            Some(Self::Gemini)
        } else if model_lower.starts_with("claude") {
            Some(Self::Anthropic)
        } else if model_lower.starts_with("gpt")
            || model_lower.starts_with("o1")
            || model_lower.starts_with("o3")
        {
            Some(Self::OpenAI)
        } else {
            None
        }
    }
}

impl std::fmt::Display for ProviderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ── Serde default helpers ────────────────────────────────────────────

fn default_priority() -> u32 {
    DEFAULT_WORKER_PRIORITY
}

fn default_cost() -> f32 {
    DEFAULT_WORKER_COST
}

fn default_health_check_timeout() -> u64 {
    30
}

fn default_health_check_interval() -> u64 {
    60
}

fn default_health_success_threshold() -> u32 {
    2
}

fn default_health_failure_threshold() -> u32 {
    3
}

fn default_max_connection_attempts() -> u32 {
    20
}

// ── Health check config ─────────────────────────────────────────────

/// Health check configuration shared across protocol and runtime layers.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct HealthCheckConfig {
    /// Health check timeout in seconds (default: 30).
    #[serde(default = "default_health_check_timeout")]
    pub timeout_secs: u64,

    /// Health check interval in seconds (default: 60).
    #[serde(default = "default_health_check_interval")]
    pub check_interval_secs: u64,

    /// Number of successful health checks needed to mark worker as healthy (default: 2).
    #[serde(default = "default_health_success_threshold")]
    pub success_threshold: u32,

    /// Number of failed health checks before marking worker as unhealthy (default: 3).
    #[serde(default = "default_health_failure_threshold")]
    pub failure_threshold: u32,

    /// Disable periodic health checks for this worker (default: false).
    #[serde(default)]
    pub disable_health_check: bool,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            timeout_secs: default_health_check_timeout(),
            check_interval_secs: default_health_check_interval(),
            success_threshold: default_health_success_threshold(),
            failure_threshold: default_health_failure_threshold(),
            disable_health_check: false,
        }
    }
}

// ── Worker models ───────────────────────────────────────────────────

/// Models configuration for a worker.
///
/// Encodes the three real cases instead of relying on `Vec` semantics:
/// - `Wildcard` — accepts any model (empty models list on the wire)
/// - `Single` — serves exactly one model
/// - `Multi` — serves multiple distinct models (len >= 2)
#[derive(Debug, Clone, Default)]
pub enum WorkerModels {
    /// Worker accepts any model (e.g., external API without discovery).
    #[default]
    Wildcard,
    /// Worker serves exactly one model (most common for local inference).
    Single(Box<ModelCard>),
    /// Worker serves multiple distinct models (len >= 2).
    Multi(Vec<ModelCard>),
}

impl WorkerModels {
    /// Returns `true` if this is a wildcard (accepts any model).
    pub fn is_wildcard(&self) -> bool {
        matches!(self, Self::Wildcard)
    }

    /// Returns the primary model: `Single` → `Some`, `Multi` → first, `Wildcard` → `None`.
    pub fn primary(&self) -> Option<&ModelCard> {
        match self {
            Self::Wildcard => None,
            Self::Single(card) => Some(card.as_ref()),
            Self::Multi(cards) => cards.first(),
        }
    }

    /// Returns all models as a slice (empty for `Wildcard`).
    pub fn all(&self) -> &[ModelCard] {
        match self {
            Self::Wildcard => &[],
            Self::Single(card) => std::slice::from_ref(card.as_ref()),
            Self::Multi(cards) => cards,
        }
    }

    /// Find a model by ID (checks aliases via `ModelCard::matches`).
    pub fn find(&self, id: &str) -> Option<&ModelCard> {
        match self {
            Self::Wildcard => None,
            Self::Single(card) => card.matches(id).then_some(card.as_ref()),
            Self::Multi(cards) => cards.iter().find(|m| m.matches(id)),
        }
    }

    /// Returns `true` if the worker supports the given model ID.
    /// Wildcard workers always return `true`.
    pub fn supports(&self, id: &str) -> bool {
        match self {
            Self::Wildcard => true,
            _ => self.find(id).is_some(),
        }
    }

    /// Iterate over all models. Empty iterator for `Wildcard`.
    pub fn iter(&self) -> impl Iterator<Item = &ModelCard> {
        self.all().iter()
    }
}

impl From<Vec<ModelCard>> for WorkerModels {
    fn from(models: Vec<ModelCard>) -> Self {
        match models.len() {
            0 => Self::Wildcard,
            1 => {
                let Some(model) = models.into_iter().next() else {
                    return Self::Wildcard;
                };
                Self::Single(Box::new(model))
            }
            _ => Self::Multi(models),
        }
    }
}

/// Serialize as `Vec<ModelCard>` for wire compatibility.
impl Serialize for WorkerModels {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.all().serialize(serializer)
    }
}

/// Deserialize from `Vec<ModelCard>` for wire compatibility.
impl<'de> Deserialize<'de> for WorkerModels {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let models = Vec::<ModelCard>::deserialize(deserializer)?;
        Ok(Self::from(models))
    }
}

/// JsonSchema: wire format is `Vec<ModelCard>`.
impl JsonSchema for WorkerModels {
    fn schema_name() -> String {
        "WorkerModels".to_string()
    }

    fn json_schema(gen: &mut schemars::gen::SchemaGenerator) -> schemars::schema::Schema {
        Vec::<ModelCard>::json_schema(gen)
    }
}

// ── Core identity ────────────────────────────────────────────────────

/// Core worker identity and configuration.
///
/// The single canonical representation of "what is a worker". Used as the
/// shared sub-struct across API requests, API responses, and internal runtime
/// state via `#[serde(flatten)]`.
///
/// Fields use `#[serde(default)]` so the same struct works for both input
/// (partial config from user) and output (fully resolved state).
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct WorkerSpec {
    /// Worker URL.
    pub url: String,

    /// Models this worker can serve.
    #[serde(default, skip_serializing_if = "WorkerModels::is_wildcard")]
    pub models: WorkerModels,

    /// Worker type: regular, prefill, or decode.
    #[serde(default)]
    pub worker_type: WorkerType,

    /// Connection mode: http or grpc.
    #[serde(default)]
    pub connection_mode: ConnectionMode,

    /// Runtime type: sglang, vllm, trtllm, or external.
    #[serde(default, alias = "runtime")]
    pub runtime_type: RuntimeType,

    /// External provider for API transformations.
    /// `None` means native/passthrough.
    pub provider: Option<ProviderType>,

    /// Additional labels/tags.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub labels: HashMap<String, String>,

    /// Worker priority (higher = preferred).
    #[serde(default = "default_priority")]
    pub priority: u32,

    /// Worker cost factor (baseline = 1.0).
    #[serde(default = "default_cost")]
    pub cost: f32,

    /// Worker API key. Accepted on input, never included in responses.
    #[serde(default, skip_serializing)]
    pub api_key: Option<String>,

    /// Bootstrap port for prefill workers in PD disaggregated mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bootstrap_port: Option<u16>,

    /// Bootstrap hostname (derived from URL at construction time).
    #[serde(default, skip)]
    pub bootstrap_host: String,

    /// Base URL without DP rank suffix (for DP-aware workers).
    /// When set, `url` contains the rank-suffixed form (`{base}@{rank}`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dp_base_url: Option<String>,

    /// Data-parallel rank (None = not DP-aware).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dp_rank: Option<usize>,

    /// Total data-parallel group size (None = not DP-aware).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dp_size: Option<usize>,

    /// KV connector type (e.g. "MooncakeConnector", "NixlConnector").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_connector: Option<String>,

    /// KV role (e.g. "kv_producer", "kv_consumer", "kv_both").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_role: Option<String>,

    /// KV cache block size (tokens per block) for event-driven routing.
    /// When set, overrides the router-level default for this worker's model.
    /// Typically matches the backend engine's page size (e.g. 16 for SGLang).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_block_size: Option<usize>,

    /// Per-worker health check overrides (partial — only `Some` fields override router defaults).
    #[serde(default, skip_serializing_if = "HealthCheckUpdate::is_empty")]
    pub health: HealthCheckUpdate,

    /// Per-worker HTTP connection pool overrides.
    #[serde(default, skip_serializing_if = "HttpPoolConfig::is_empty")]
    pub http_pool: HttpPoolConfig,

    /// Per-worker resilience overrides (retry + circuit breaker).
    #[serde(default, skip_serializing_if = "ResilienceUpdate::is_empty")]
    pub resilience: ResilienceUpdate,

    /// Maximum connection attempts during worker registration (default: 20).
    #[serde(default = "default_max_connection_attempts")]
    pub max_connection_attempts: u32,

    /// Per-worker load monitor interval override (seconds).
    /// When set, workers in the same group use this interval for load polling.
    /// Falls back to the global `load_monitor_interval_secs` from router config.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub load_monitor_interval_secs: Option<u64>,
}

impl WorkerSpec {
    /// Create a new `WorkerSpec` with the given URL and sensible defaults.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            models: WorkerModels::Wildcard,
            worker_type: WorkerType::default(),
            connection_mode: ConnectionMode::default(),
            runtime_type: RuntimeType::default(),
            provider: None,
            labels: HashMap::new(),
            priority: DEFAULT_WORKER_PRIORITY,
            cost: DEFAULT_WORKER_COST,
            api_key: None,
            bootstrap_port: None,
            bootstrap_host: String::new(),
            dp_base_url: None,
            dp_rank: None,
            dp_size: None,
            kv_connector: None,
            kv_role: None,
            kv_block_size: None,
            health: HealthCheckUpdate::default(),
            http_pool: HttpPoolConfig::default(),
            resilience: ResilienceUpdate::default(),
            max_connection_attempts: default_max_connection_attempts(),
            load_monitor_interval_secs: None,
        }
    }
}

// ── API types ───────────────────────────────────────────────────────

/// Worker information for API responses.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct WorkerInfo {
    /// Worker unique identifier.
    pub id: String,

    /// Primary model ID for backwards compatibility.
    /// Computed from `models[0].id` (single/multi) or `null` (wildcard).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,

    /// Worker identity and configuration.
    #[serde(flatten)]
    pub spec: WorkerSpec,

    /// Whether the worker is healthy.
    pub is_healthy: bool,

    /// Current load on the worker.
    pub load: usize,

    /// Job status for async operations (if available).
    pub job_status: Option<JobStatus>,
}

impl WorkerInfo {
    /// Create a partial WorkerInfo for pending workers (not yet registered).
    pub fn pending(worker_id: &str, url: String, job_status: Option<JobStatus>) -> Self {
        Self {
            id: worker_id.to_string(),
            model_id: None,
            spec: WorkerSpec::new(url),
            is_healthy: false,
            load: 0,
            job_status,
        }
    }
}

/// Job status for async control plane operations
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct JobStatus {
    pub job_type: String,
    pub worker_url: String,
    pub status: String,
    pub message: Option<String>,
    pub timestamp: u64,
}

impl JobStatus {
    /// Create a pending job status
    pub fn pending(job_type: &str, worker_url: &str) -> Self {
        Self {
            job_type: job_type.to_string(),
            worker_url: worker_url.to_string(),
            status: "pending".to_string(),
            message: None,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    /// Create a processing job status
    pub fn processing(job_type: &str, worker_url: &str) -> Self {
        Self {
            job_type: job_type.to_string(),
            worker_url: worker_url.to_string(),
            status: "processing".to_string(),
            message: None,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    /// Create a failed job status
    pub fn failed(job_type: &str, worker_url: &str, error: String) -> Self {
        Self {
            job_type: job_type.to_string(),
            worker_url: worker_url.to_string(),
            status: "failed".to_string(),
            message: Some(error),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }
}

/// Worker list response
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct WorkerListResponse {
    pub workers: Vec<WorkerInfo>,
    pub total: usize,
    pub stats: WorkerStats,
}

/// Worker statistics
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct WorkerStats {
    pub total_workers: usize,
    pub healthy_workers: usize,
    pub total_models: usize,
    pub total_load: usize,
    pub by_type: WorkerTypeStats,
}

/// Worker statistics by type
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct WorkerTypeStats {
    pub regular: usize,
    pub prefill: usize,
    pub decode: usize,
}

// ── Update types ────────────────────────────────────────────────────

/// Partial health check config for PATCH-style updates.
///
/// Each `None` field means "keep the existing value". This avoids the problem
/// where `#[serde(default)]` on [`HealthCheckConfig`] would silently reset
/// unspecified fields to defaults.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct HealthCheckUpdate {
    pub timeout_secs: Option<u64>,
    pub check_interval_secs: Option<u64>,
    pub success_threshold: Option<u32>,
    pub failure_threshold: Option<u32>,
    pub disable_health_check: Option<bool>,
}

impl HealthCheckUpdate {
    /// Returns `true` if all fields are `None` (no overrides specified).
    pub fn is_empty(&self) -> bool {
        self.timeout_secs.is_none()
            && self.check_interval_secs.is_none()
            && self.success_threshold.is_none()
            && self.failure_threshold.is_none()
            && self.disable_health_check.is_none()
    }
}

impl HealthCheckUpdate {
    /// Merge this update into an existing [`HealthCheckConfig`], returning a new config.
    /// Only `Some` fields are applied; `None` fields keep the existing value.
    pub fn apply_to(&self, existing: &HealthCheckConfig) -> HealthCheckConfig {
        HealthCheckConfig {
            timeout_secs: self.timeout_secs.unwrap_or(existing.timeout_secs),
            check_interval_secs: self
                .check_interval_secs
                .unwrap_or(existing.check_interval_secs),
            success_threshold: self.success_threshold.unwrap_or(existing.success_threshold),
            failure_threshold: self.failure_threshold.unwrap_or(existing.failure_threshold),
            disable_health_check: self
                .disable_health_check
                .unwrap_or(existing.disable_health_check),
        }
    }
}

/// Per-worker HTTP connection pool configuration.
/// All fields optional — `None` means "use router/global default".
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct HttpPoolConfig {
    /// Max idle connections per host (default: 8).
    pub pool_max_idle_per_host: Option<usize>,
    /// Idle connection timeout in seconds (default: 50).
    pub pool_idle_timeout_secs: Option<u64>,
    /// Request timeout in seconds (default: 30).
    pub timeout_secs: Option<u64>,
    /// Connect timeout in seconds (default: 10).
    pub connect_timeout_secs: Option<u64>,
}

impl HttpPoolConfig {
    /// Returns `true` if all fields are `None` (no overrides).
    pub fn is_empty(&self) -> bool {
        self.pool_max_idle_per_host.is_none()
            && self.pool_idle_timeout_secs.is_none()
            && self.timeout_secs.is_none()
            && self.connect_timeout_secs.is_none()
    }
}

/// Per-worker resilience overrides (retry + circuit breaker).
/// All fields optional — `None` means "use router default".
/// Mirrors `HealthCheckUpdate` pattern for PATCH-style config.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ResilienceUpdate {
    // ── Retry overrides ──
    /// Max retry attempts (includes first attempt). 1 = no retries.
    pub max_retries: Option<u32>,
    /// Initial backoff delay in milliseconds.
    pub initial_backoff_ms: Option<u64>,
    /// Maximum backoff delay in milliseconds.
    pub max_backoff_ms: Option<u64>,
    /// Backoff multiplier for exponential backoff.
    pub backoff_multiplier: Option<f32>,
    /// Jitter factor (0.0–1.0) applied to backoff delay.
    pub jitter_factor: Option<f32>,
    /// Disable retries entirely for this worker.
    pub disable_retry: Option<bool>,

    // ── Circuit breaker overrides ──
    /// Consecutive failures to open the circuit.
    pub cb_failure_threshold: Option<u32>,
    /// Consecutive successes to close the circuit from half-open.
    pub cb_success_threshold: Option<u32>,
    /// Seconds to wait before attempting half-open.
    pub cb_timeout_secs: Option<u64>,
    /// Time window in seconds for failure counting.
    pub cb_window_secs: Option<u64>,
    /// Disable circuit breaker entirely for this worker.
    pub disable_circuit_breaker: Option<bool>,

    // ── Retryable status codes ──
    /// Custom retryable HTTP status codes.
    /// When set, replaces the default set (408, 429, 500, 502, 503, 504).
    pub retryable_status_codes: Option<Vec<u16>>,
}

impl ResilienceUpdate {
    /// Returns `true` if all fields are `None` (no overrides).
    pub fn is_empty(&self) -> bool {
        self.max_retries.is_none()
            && self.initial_backoff_ms.is_none()
            && self.max_backoff_ms.is_none()
            && self.backoff_multiplier.is_none()
            && self.jitter_factor.is_none()
            && self.disable_retry.is_none()
            && self.cb_failure_threshold.is_none()
            && self.cb_success_threshold.is_none()
            && self.cb_timeout_secs.is_none()
            && self.cb_window_secs.is_none()
            && self.disable_circuit_breaker.is_none()
            && self.retryable_status_codes.is_none()
    }
}

/// Worker update request
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct WorkerUpdateRequest {
    /// Update priority
    pub priority: Option<u32>,

    /// Update cost
    pub cost: Option<f32>,

    /// Update labels
    pub labels: Option<HashMap<String, String>>,

    /// Update API key (for key rotation)
    pub api_key: Option<String>,

    /// Update health check configuration (partial — only specified fields change)
    pub health: Option<HealthCheckUpdate>,
}

// ── Response types ──────────────────────────────────────────────────

/// Generic API response
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct WorkerApiResponse {
    pub success: bool,
    pub message: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker: Option<WorkerInfo>,
}

/// Error response
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct WorkerErrorResponse {
    pub error: String,
    pub code: String,
}

/// Result from flush cache operations across workers
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FlushCacheResult {
    pub successful: Vec<String>,
    pub failed: Vec<(String, String)>,
    pub total_workers: usize,
    pub http_workers: usize,
    pub message: String,
}

/// Result from getting worker loads
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WorkerLoadsResult {
    pub loads: Vec<WorkerLoadInfo>,
    pub total_workers: usize,
    pub successful: usize,
    pub failed: usize,
}

/// Per-DP-rank load snapshot from a backend.
///
/// Contains core metrics from the sglang `/v1/loads` endpoint or `GetLoads` gRPC RPC.
/// Each snapshot represents one data-parallel rank's scheduler state.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SchedulerLoadSnapshot {
    pub dp_rank: i32,
    pub num_running_reqs: i32,
    pub num_waiting_reqs: i32,
    pub num_total_reqs: i32,
    pub num_used_tokens: i32,
    pub max_total_num_tokens: i32,
    /// Token usage ratio (0.0–1.0).
    pub token_usage: f64,
    pub gen_throughput: f64,
    pub cache_hit_rate: f64,
    pub utilization: f64,
    pub max_running_requests: i32,
}

/// Full load response for a single worker across all DP ranks.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct WorkerLoadResponse {
    pub timestamp: String,
    pub dp_rank_count: i32,
    pub loads: Vec<SchedulerLoadSnapshot>,
}

impl WorkerLoadResponse {
    /// Average token usage ratio across DP ranks. Returns 0.0 if empty.
    pub fn effective_token_usage(&self) -> f64 {
        if self.loads.is_empty() {
            return 0.0;
        }
        self.loads.iter().map(|l| l.token_usage).sum::<f64>() / self.loads.len() as f64
    }

    /// Total used tokens summed across all DP ranks.
    pub fn total_used_tokens(&self) -> i64 {
        self.loads.iter().map(|l| l.num_used_tokens as i64).sum()
    }

    pub fn dp_rank_loads(&self) -> HashMap<isize, isize> {
        let mut map = HashMap::new();
        for snapshot in &self.loads {
            map.insert(snapshot.dp_rank as isize, snapshot.num_used_tokens as isize);
        }
        map
    }
}

/// Individual worker load information
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WorkerLoadInfo {
    pub worker: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker_type: Option<String>,
    pub load: isize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<WorkerLoadResponse>,
}

#[cfg(feature = "axum")]
impl IntoResponse for FlushCacheResult {
    fn into_response(self) -> Response {
        let status = if self.failed.is_empty() {
            StatusCode::OK
        } else {
            StatusCode::PARTIAL_CONTENT
        };

        let mut body = json!({
            "status": if self.failed.is_empty() { "success" } else { "partial_success" },
            "message": self.message,
            "workers_flushed": self.successful.len(),
            "total_http_workers": self.http_workers,
            "total_workers": self.total_workers
        });

        if !self.failed.is_empty() {
            body["successful"] = json!(self.successful);
            body["failed"] = json!(self
                .failed
                .into_iter()
                .map(|(url, err)| json!({"worker": url, "error": err}))
                .collect::<Vec<_>>());
        }

        (status, Json(body)).into_response()
    }
}

#[cfg(feature = "axum")]
impl IntoResponse for WorkerLoadsResult {
    fn into_response(self) -> Response {
        let loads: Vec<Value> = self
            .loads
            .iter()
            .map(|info| {
                let mut entry = json!({"worker": &info.worker, "load": info.load});
                if let Some(ref details) = info.details {
                    entry["details"] = json!(details);
                }
                entry
            })
            .collect();
        Json(json!({"workers": loads})).into_response()
    }
}
