//! Client-latency adapter — emits `client-latency/{client_region}/{target_region}/{server_name}`.
//!
//! Each completed cross-region forward records its target + latency via
//! [`Self::record_latency`]. A periodic 10 s reconcile drains the per-target
//! observation buffer, computes p50/p95, and publishes one envelope per
//! target. v1 keeps the scalar p50/p95 wire shape; the additive-bucket
//! histogram (design §1) is a follow-up so the per-replica → consumer
//! aggregation does correct percentile math instead of averaging.

use std::{collections::HashMap, sync::Arc};

use parking_lot::Mutex;

use crate::cross_region::{
    ClientLatencySignal, CrossRegionResult, CrossRegionSyncService, SignalKey, SignalKind,
};

/// Default freshness window (design §4): 30 s with a 10 s reconcile.
pub const DEFAULT_CLIENT_LATENCY_STALE_AFTER_MS: u32 = 30_000;

/// Per-target rolling observation buffer.
#[derive(Debug, Default)]
struct TargetObservations {
    samples_ms: Vec<u64>,
}

impl TargetObservations {
    fn record(&mut self, latency_ms: u64) {
        self.samples_ms.push(latency_ms);
    }

    /// Drain samples and compute p50/p95. Returns `None` if no samples are
    /// buffered (so the adapter skips publishing a stale-only signal).
    fn drain_summary(&mut self) -> Option<(u64, u64)> {
        if self.samples_ms.is_empty() {
            return None;
        }
        let mut sorted = std::mem::take(&mut self.samples_ms);
        sorted.sort_unstable();
        let len = sorted.len();
        let p50 = sorted[len / 2];
        // Percentile-rank ceiling so p95 of a single sample = that sample.
        let p95_idx = ((len as f64) * 0.95).ceil() as usize;
        let p95 = sorted[p95_idx.saturating_sub(1).min(len - 1)];
        Some((p50, p95))
    }
}

/// Client-latency producer.
///
/// Cloning is cheap (Arc-wrapped state); the request-plane forwarder can
/// hold a clone and call [`Self::record_latency`] from any task.
#[derive(Debug, Clone)]
pub struct ClientLatencyAdapter {
    sync: Arc<CrossRegionSyncService>,
    observations: Arc<Mutex<HashMap<String, TargetObservations>>>,
    stale_after_ms: u32,
}

impl ClientLatencyAdapter {
    pub fn new(sync: Arc<CrossRegionSyncService>) -> Self {
        Self {
            sync,
            observations: Arc::new(Mutex::new(HashMap::new())),
            stale_after_ms: DEFAULT_CLIENT_LATENCY_STALE_AFTER_MS,
        }
    }

    pub fn with_stale_after_ms(mut self, stale_after_ms: u32) -> Self {
        self.stale_after_ms = stale_after_ms;
        self
    }

    /// Called by the cross-region forwarder on each completed remote forward.
    pub fn record_latency(&self, target_region: &str, latency_ms: u64) {
        let mut obs = self.observations.lock();
        obs.entry(target_region.to_string())
            .or_default()
            .record(latency_ms);
    }

    /// Drain every per-target observation buffer and publish one envelope
    /// per target. Invoked by the 10 s reconcile tick. Targets with no
    /// samples since the previous drain are skipped (their last published
    /// envelope will age out via `stale_after_ms`).
    pub fn drain_and_publish(&self) -> CrossRegionResult<()> {
        let drained = {
            let mut obs = self.observations.lock();
            obs.iter_mut()
                .filter_map(|(target, buf)| buf.drain_summary().map(|s| (target.clone(), s)))
                .collect::<Vec<_>>()
        };
        for (target_region, (p50, p95)) in drained {
            self.publish_for(&target_region, p50, p95)?;
        }
        Ok(())
    }

    /// Direct publish — exposed for tests and any path that wants to push a
    /// precomputed (p50, p95) without going through the observation buffer.
    pub fn publish_for(
        &self,
        target_region: &str,
        p50_latency_ms: u64,
        p95_latency_ms: u64,
    ) -> CrossRegionResult<()> {
        let client_region = self.sync.region_id().to_string();
        let server_name = self.sync.server_name().to_string();
        let key = SignalKey::ClientLatency {
            client_region: client_region.clone(),
            target_region: target_region.to_string(),
            server_name: server_name.clone(),
        };
        let body = ClientLatencySignal {
            client_region,
            target_region: target_region.to_string(),
            server_name,
            p50_latency_ms,
            p95_latency_ms,
        };
        self.sync
            .publish_signal(key, SignalKind::ClientLatency(body), self.stale_after_ms)
    }

    /// Drop a tombstone for one target — used when a target region exits
    /// the allowed-regions config.
    pub fn remove_for(&self, target_region: &str) -> CrossRegionResult<()> {
        self.observations.lock().remove(target_region);
        let client_region = self.sync.region_id().to_string();
        let server_name = self.sync.server_name().to_string();
        let key = SignalKey::ClientLatency {
            client_region,
            target_region: target_region.to_string(),
            server_name,
        };
        self.sync.remove_signal(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn service() -> Arc<CrossRegionSyncService> {
        Arc::new(
            CrossRegionSyncService::new("us-phoenix-1".to_string(), "smg-router-a".to_string())
                .expect("service constructs"),
        )
    }

    #[test]
    fn drain_summary_returns_none_when_empty() {
        let mut obs = TargetObservations::default();
        assert!(obs.drain_summary().is_none());
    }

    #[test]
    fn drain_summary_computes_p50_p95() {
        let mut obs = TargetObservations::default();
        for v in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] {
            obs.record(v);
        }
        let (p50, p95) = obs.drain_summary().expect("samples present");
        assert_eq!(p50, 60); // index len/2 = 5 → sorted[5] = 60
        assert_eq!(p95, 100); // ceil(10*0.95)=10 → sorted[9] = 100

        // Buffer was consumed.
        assert!(obs.drain_summary().is_none());
    }

    #[test]
    fn record_and_drain_publishes_one_envelope_per_target() {
        let svc = service();
        let adapter = ClientLatencyAdapter::new(svc.clone());
        for v in [12, 18, 25] {
            adapter.record_latency("us-chicago-1", v);
        }
        for v in [80, 90, 100, 110, 120] {
            adapter.record_latency("us-ashburn-1", v);
        }

        adapter.drain_and_publish().expect("publish ok");

        let (entries, _) = svc.local_log_snapshot();
        assert_eq!(entries.len(), 2);
        let mut targets: Vec<&str> = entries
            .iter()
            .filter_map(|e| match e.signal.as_ref()? {
                SignalKind::ClientLatency(s) => Some(s.target_region.as_str()),
                _ => None,
            })
            .collect();
        targets.sort_unstable();
        assert_eq!(targets, vec!["us-ashburn-1", "us-chicago-1"]);
    }

    #[test]
    fn publish_for_uses_local_region_as_client_region() {
        let svc = service();
        let adapter = ClientLatencyAdapter::new(svc.clone());
        adapter.publish_for("us-chicago-1", 30, 80).unwrap();

        let (entries, _) = svc.local_log_snapshot();
        assert_eq!(entries.len(), 1);
        match &entries[0].key {
            SignalKey::ClientLatency {
                client_region,
                target_region,
                server_name,
            } => {
                assert_eq!(client_region, "us-phoenix-1");
                assert_eq!(target_region, "us-chicago-1");
                assert_eq!(server_name, "smg-router-a");
            }
            _ => panic!("unexpected key kind: {:?}", entries[0].key),
        }
        match &entries[0].signal {
            Some(SignalKind::ClientLatency(s)) => {
                assert_eq!(s.p50_latency_ms, 30);
                assert_eq!(s.p95_latency_ms, 80);
            }
            other => panic!("unexpected signal: {other:?}"),
        }
    }

    #[test]
    fn drain_skips_targets_with_no_new_samples() {
        let svc = service();
        let adapter = ClientLatencyAdapter::new(svc.clone());
        adapter.record_latency("us-chicago-1", 30);
        adapter.drain_and_publish().unwrap();
        let (entries_after_first, _) = svc.local_log_snapshot();
        assert_eq!(entries_after_first.len(), 1);

        // No new samples — second drain publishes nothing.
        adapter.drain_and_publish().unwrap();
        let (entries_after_second, _) = svc.local_log_snapshot();
        assert_eq!(entries_after_second.len(), 1);
    }

    #[test]
    fn remove_for_emits_tombstone() {
        let svc = service();
        let adapter = ClientLatencyAdapter::new(svc.clone());
        adapter.publish_for("us-chicago-1", 30, 80).unwrap();
        adapter.remove_for("us-chicago-1").unwrap();

        let (entries, _) = svc.local_log_snapshot();
        assert_eq!(entries.len(), 2);
        assert!(entries[1].removed);
        assert!(entries[1].version > entries[0].version);
    }
}
