//! Test helpers shared by the four adapter unit-test modules. Builds a fresh
//! `CrossRegionSyncService` over a per-test broadcast stream namespace and
//! exposes a `snapshot` helper that returns every envelope currently staged
//! in the outbox (the same envelopes mesh would ship on the next gossip
//! round).

#![allow(
    clippy::expect_used,
    reason = "test-only fixtures decode bytes we just wrote"
)]

use std::sync::Arc;

use smg_mesh::{MeshKV, StreamConfig, StreamRouting};

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
    let namespace = mesh_kv.configure_stream_prefix(
        CROSS_REGION_NAMESPACE_PREFIX,
        StreamConfig {
            max_buffer_bytes: 16 * 1024 * 1024,
            routing: StreamRouting::Broadcast,
        },
    );
    Arc::new(
        CrossRegionSyncService::new(region.to_string(), server.to_string(), namespace)
            .expect("service should construct"),
    )
}

/// Snapshot every envelope currently staged in the sync service's outbox
/// (i.e. queued for the next mesh gossip round). Tombstones do not appear —
/// `remove_signal` purges the outbox locally rather than emitting a wire
/// entry.
pub fn live_envelopes(svc: &CrossRegionSyncService) -> Vec<SignalEnvelope<SignalKind>> {
    let mut envelopes = svc.outbox_snapshot();
    envelopes.sort_by_key(|env| env.key.as_path());
    envelopes
}

/// Convenience: assert that exactly one envelope is staged and return it.
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
