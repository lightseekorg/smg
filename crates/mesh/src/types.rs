//! Value types that flow across the mesh wire and into gateway
//! adapters. Kept in the mesh crate so producers and consumers
//! share a single canonical definition.

use serde::{Deserialize, Serialize};

/// Worker state entry synced across mesh nodes. `spec` is an
/// opaque JSON-serialized `WorkerSpec`; the mesh crate
/// doesn't interpret it.
///
/// `Eq`/`Hash` are intentionally omitted: `load: f64` can be
/// NaN, which would violate `Eq` reflexivity.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct WorkerState {
    pub worker_id: String,
    pub model_id: String,
    pub url: String,
    pub health: bool,
    pub load: f64,
    pub version: u64,
    /// Opaque worker specification (JSON-serialized `WorkerSpec`
    /// from the gateway; JSON because the type's serde-skip
    /// attributes don't round-trip positional formats). Empty on
    /// old nodes that don't populate this field.
    #[serde(default)]
    pub spec: Vec<u8>,
}
