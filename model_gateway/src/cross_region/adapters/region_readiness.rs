//! Region-readiness adapter — emits `smg-readiness/{region}/{server_name}`.
//!
//! v1 readiness shape is just `ready: bool`. The richer worker-keyed capacity
//! map called for by design §3a (consumer-side worker dedup) is a follow-up;
//! the v1 body change would ripple into too many consumer call sites without
//! producer-side data to populate it yet. This adapter ships the bool and
//! leaves the capacity map at body level for a later refactor.

use std::sync::Arc;

use crate::cross_region::{
    CrossRegionResult, CrossRegionSyncService, SignalKey, SignalKind, SmgReadinessSignal,
};

/// Default freshness window (design §4): readiness is recomputed every 5 s, so
/// a 30 s stale-after gives consumers ~6 refresh cycles of cushion before they
/// gate the signal out at ranking time.
pub const DEFAULT_READINESS_STALE_AFTER_MS: u32 = 30_000;

/// Region-readiness producer.
///
/// Holds the sync-service handle plus the freshness window to stamp on every
/// publish. The reconcile cadence is the caller's responsibility — start
/// either a periodic tokio task or a hook-driven flow that calls
/// [`Self::publish_ready`] on relevant state changes.
#[derive(Debug, Clone)]
pub struct RegionReadinessAdapter {
    sync: Arc<CrossRegionSyncService>,
    stale_after_ms: u32,
}

impl RegionReadinessAdapter {
    /// Build an adapter rooted at the given sync service. Uses the default
    /// stale-after window; call [`Self::with_stale_after_ms`] to override.
    pub fn new(sync: Arc<CrossRegionSyncService>) -> Self {
        Self {
            sync,
            stale_after_ms: DEFAULT_READINESS_STALE_AFTER_MS,
        }
    }

    /// Override the freshness window stamped on each published envelope.
    pub fn with_stale_after_ms(mut self, stale_after_ms: u32) -> Self {
        self.stale_after_ms = stale_after_ms;
        self
    }

    /// Publish the current readiness state. Adapters typically wire this to
    /// the gateway's readiness gate plus a periodic reconcile.
    pub fn publish_ready(&self, ready: bool) -> CrossRegionResult<()> {
        let region_id = self.sync.region_id().to_string();
        let server_name = self.sync.server_name().to_string();
        let key = SignalKey::SmgReadiness {
            region_id: region_id.clone(),
            server_name: server_name.clone(),
        };
        let body = SmgReadinessSignal {
            region_id,
            server_name,
            ready,
        };
        self.sync
            .publish_signal(key, SignalKind::SmgReadiness(body), self.stale_after_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn service() -> Arc<CrossRegionSyncService> {
        Arc::new(
            CrossRegionSyncService::new("us-ashburn-1".to_string(), "smg-router-a".to_string())
                .expect("service constructs"),
        )
    }

    #[test]
    fn publish_ready_emits_per_replica_envelope() {
        let svc = service();
        let adapter = RegionReadinessAdapter::new(svc.clone());

        adapter.publish_ready(true).expect("publish ok");
        let (entries, _) = svc.local_log_snapshot();
        assert_eq!(entries.len(), 1);
        let env = &entries[0];
        assert!(matches!(env.signal, Some(SignalKind::SmgReadiness(_))));
        match &env.key {
            SignalKey::SmgReadiness {
                region_id,
                server_name,
            } => {
                assert_eq!(region_id, "us-ashburn-1");
                assert_eq!(server_name, "smg-router-a");
            }
            _ => panic!("unexpected key kind: {:?}", env.key),
        }
        assert_eq!(env.actor, "smg-router-a");
        assert_eq!(env.stale_after_ms, DEFAULT_READINESS_STALE_AFTER_MS);
    }

    #[test]
    fn publish_ready_toggles_value() {
        let svc = service();
        let adapter = RegionReadinessAdapter::new(svc.clone());

        adapter.publish_ready(true).unwrap();
        adapter.publish_ready(false).unwrap();
        let (entries, _) = svc.local_log_snapshot();
        assert_eq!(entries.len(), 2);
        let ready_values: Vec<bool> = entries
            .iter()
            .filter_map(|e| match e.signal.as_ref()? {
                SignalKind::SmgReadiness(s) => Some(s.ready),
                _ => None,
            })
            .collect();
        assert_eq!(ready_values, vec![true, false]);
        assert!(entries[1].version > entries[0].version);
    }

    #[test]
    fn with_stale_after_ms_overrides_default() {
        let svc = service();
        let adapter = RegionReadinessAdapter::new(svc.clone()).with_stale_after_ms(5_000);
        adapter.publish_ready(true).unwrap();
        let (entries, _) = svc.local_log_snapshot();
        assert_eq!(entries[0].stale_after_ms, 5_000);
    }
}
