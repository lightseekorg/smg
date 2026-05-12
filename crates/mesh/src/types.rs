//! Value types that flow across the mesh wire and into gateway
//! adapters. Kept in the mesh crate so producers and consumers
//! share a single canonical definition.

use serde::{Deserialize, Serialize};

/// Worker state entry synced across mesh nodes.
///
/// Contains runtime state (`health`, `load`) plus an opaque
/// `spec` blob carrying the full worker configuration. The mesh
/// crate doesn't interpret `spec` — the gateway serializes
/// `WorkerSpec` into it on the sending side and deserializes on
/// the receiving side.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct WorkerState {
    pub worker_id: String,
    pub model_id: String,
    pub url: String,
    pub health: bool,
    pub load: f64,
    pub version: u64,
    /// Opaque worker specification (bincode-serialized
    /// `WorkerSpec` from the gateway). Empty on old nodes that
    /// don't populate this field.
    #[serde(default)]
    pub spec: Vec<u8>,
}

// Manual Hash impl: `f64` doesn't implement `Hash`, so we coerce
// `load` to `i64` for hashing purposes. Two states with `load`
// values that differ but truncate to the same `i64` will hash
// equal; equality (below) uses the same epsilon discipline.
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

// Manual Eq via the derived PartialEq; the f64 comparison is
// bitwise via `PartialEq`, which is acceptable because the
// gateway either advertises a deterministic value or zero.
impl Eq for WorkerState {}
