//! Test helpers shared by the four adapter unit-test modules. Builds a fresh
//! `CrossRegionSyncService` over a per-test `MeshKV` namespace and exposes a
//! `snapshot` helper that decodes every envelope currently materialized in
//! the namespace.

#![allow(
    clippy::expect_used,
    reason = "test-only fixtures decode bytes we just wrote"
)]

use std::sync::Arc;

use smg_mesh::{MergeStrategy, MeshKV};

use crate::cross_region::{
    sync::{SignalKind, CROSS_REGION_NAMESPACE_PREFIX},
    CrossRegionSyncService, SignalEnvelope,
};

/// Default region/server identity used by adapter tests.
pub const TEST_REGION: &str = "us-ashburn-1";
pub const TEST_SERVER: &str = "smg-router-a";

/// Build a sync service rooted at `(TEST_REGION, TEST_SERVER)` over a fresh
/// in-process mesh KV instance. Each call produces an independent namespace,
/// so tests do not share state.
pub fn service() -> Arc<CrossRegionSyncService> {
    service_with_identity(TEST_REGION, TEST_SERVER)
}

/// Same as [`service`] but lets the caller stamp a custom region/server.
pub fn service_with_identity(region: &str, server: &str) -> Arc<CrossRegionSyncService> {
    let mesh_kv = Arc::new(MeshKV::new(server.to_string()));
    let namespace =
        mesh_kv.configure_crdt_prefix(CROSS_REGION_NAMESPACE_PREFIX, MergeStrategy::LastWriterWins);
    Arc::new(
        CrossRegionSyncService::new(region.to_string(), server.to_string(), namespace)
            .expect("service should construct"),
    )
}

/// Decode every live envelope currently visible in the sync service's mesh
/// namespace. Tombstones are not returned because mesh `get` reports `None`
/// for deleted keys; assert tombstone presence by checking that the original
/// key no longer reads back.
pub fn live_envelopes(svc: &CrossRegionSyncService) -> Vec<SignalEnvelope<SignalKind>> {
    let namespace = svc.namespace();
    let mut envelopes = Vec::new();
    for key in namespace.keys("") {
        if let Some(bytes) = namespace.get(&key) {
            let envelope: SignalEnvelope<SignalKind> = serde_json::from_slice(&bytes)
                .expect("namespace value must round-trip through serde_json");
            envelopes.push(envelope);
        }
    }
    envelopes.sort_by_key(|env| env.key.as_path());
    envelopes
}

/// Convenience: assert that exactly one envelope is live and return it.
pub fn single_live(svc: &CrossRegionSyncService) -> SignalEnvelope<SignalKind> {
    let mut envelopes = live_envelopes(svc);
    assert_eq!(
        envelopes.len(),
        1,
        "expected exactly one live envelope, found {}",
        envelopes.len()
    );
    envelopes.pop().expect("checked length above")
}
