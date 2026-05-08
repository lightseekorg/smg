use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Membership state entry used by topology planning.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
pub struct MembershipState {
    pub name: String,
    pub address: String,
    pub status: i32,
    pub version: u64,
    pub metadata: BTreeMap<String, Vec<u8>>,
}

/// Global rate limit configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct RateLimitConfig {
    pub limit_per_second: u64,
}

/// Key for global rate limit configuration in MeshKV's `config:` namespace.
pub const GLOBAL_RATE_LIMIT_KEY: &str = "global_rate_limit";

/// Key for the global rate limit counter in MeshKV's `rl:` namespace.
pub const GLOBAL_RATE_LIMIT_COUNTER_KEY: &str = "global";

/// Worker state entry synced across mesh nodes.
///
/// Contains runtime state (`health`, `load`) plus an opaque `spec` blob
/// carrying the full worker configuration. The mesh crate doesn't interpret
/// `spec`; the gateway serializes/deserializes it at the adapter boundary.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct WorkerState {
    pub worker_id: String,
    pub model_id: String,
    pub url: String,
    pub health: bool,
    pub load: f64,
    pub version: u64,
    #[serde(default)]
    pub spec: Vec<u8>,
}

impl std::hash::Hash for WorkerState {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.worker_id.hash(state);
        self.model_id.hash(state);
        self.url.hash(state);
        self.health.hash(state);
        (self.load as i64).hash(state);
        self.version.hash(state);
        self.spec.hash(state);
    }
}

impl Eq for WorkerState {}
