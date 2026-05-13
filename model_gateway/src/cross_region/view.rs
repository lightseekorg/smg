//! Typed projection layer over the materialized `CrossRegionState`.
//!
//! `RemoteRegionView` is the read-side entry point for cross-region consumers
//! (candidate ranking, observability projections). It applies the
//! Phase B aggregation rules that translate per-replica signals into the
//! region-/worker-/pair-shaped values callers actually want, with a
//! freshness window applied uniformly:
//!
//! - **readiness**: a region is ready when at least one fresh replica
//!   reports `ready: true`. (No quorum/all-ready variant today; if product
//!   semantics ever change, this is the only place to update.)
//! - **worker_health**: a worker is routable if any fresh replica observes
//!   it routable; otherwise the worker_status falls through to the first
//!   fresh observation in `server_name` order so callers still see one
//!   representative status.
//! - **worker_load**: per-replica loads are summed for the requested
//!   `model_id`. Each replica reports its own dispatcher's disjoint
//!   in-flight count, so the sum is the total observed load across
//!   replicas without double counting.
//! - **client_latency**: the minimum p50/p95 across fresh replicas is the
//!   best-case observed network path. Min is the v1 scalar choice; a
//!   histogram-bucketed projection can replace it without breaking callers.
//!
//! `freshness_age_ms` on each projection tracks the **oldest** contributing
//! replica we trust, so candidate ranking can bound its overall freshness.

use openai_protocol::worker::WorkerStatus;

use super::{
    state::{CrossRegionState, SignalVersion},
    ClientLatencySignal,
};

/// Aggregated read view of `CrossRegionState` against a fixed clock and
/// freshness window. Cheap to construct (holds a borrow + two scalars);
/// build a new view per request rather than caching one.
#[derive(Debug)]
pub struct RemoteRegionView<'a> {
    state: &'a CrossRegionState,
    now_ms: i64,
    freshness_window_ms: i64,
}

impl<'a> RemoteRegionView<'a> {
    /// Build a view rooted at the materialized state with an explicit clock
    /// and freshness window. Entries with `now_ms - updated_at_ms >
    /// freshness_window_ms` are filtered out before aggregation.
    pub fn new(state: &'a CrossRegionState, now_ms: i64, freshness_window_ms: i64) -> Self {
        Self {
            state,
            now_ms,
            freshness_window_ms,
        }
    }

    /// All regions observed in the materialized state (irrespective of
    /// freshness). Use [`readiness`](Self::readiness) to narrow to ones with
    /// fresh signals.
    pub fn regions(&self) -> Vec<&'a str> {
        self.state.regions()
    }

    /// Aggregated readiness for one region. `Some` when at least one fresh
    /// replica is observed; `None` when every replica is stale or the
    /// region has never been observed.
    pub fn readiness(&self, region_id: &str) -> Option<RegionReadinessProjection<'a>> {
        let mut ready = false;
        let mut age = None;
        let mut region_segment = None;
        for (signal, version) in self.state.readiness_replicas(region_id) {
            if !is_fresh(self.now_ms, version, self.freshness_window_ms) {
                continue;
            }
            let entry_age = signal_age_ms(self.now_ms, version);
            age = Some(age.unwrap_or(0).max(entry_age));
            if region_segment.is_none() {
                region_segment = Some(signal.region_id.as_str());
            }
            if signal.ready {
                ready = true;
            }
        }
        age.map(|freshness_age_ms| RegionReadinessProjection {
            region_id: region_segment.unwrap_or(""),
            ready,
            freshness_age_ms,
        })
    }

    /// True when the state has at least one readiness replica for the
    /// region, regardless of staleness. Lets callers distinguish "stale"
    /// from "absent" without re-reading the state.
    pub fn has_readiness_replica(&self, region_id: &str) -> bool {
        self.state.readiness_replicas(region_id).next().is_some()
    }

    /// Worker IDs observed (across replicas) in a region.
    pub fn worker_ids(&self, region_id: &str) -> Vec<&'a str> {
        self.state.worker_ids(region_id)
    }

    /// Aggregated worker projection for one `(region, worker)` pair.
    /// Returns `None` when the worker has never been observed; returns
    /// `Some` even when all replicas are stale so callers can read the
    /// `saw_*` discriminants on the worker projection's accessors.
    pub fn worker(
        &self,
        region_id: &'a str,
        worker_id: &'a str,
    ) -> Option<WorkerProjection<'a>> {
        let has_health = self
            .state
            .worker_health_replicas(region_id, worker_id)
            .next()
            .is_some();
        let has_load = self
            .state
            .worker_load_replicas(region_id, worker_id)
            .next()
            .is_some();
        if !has_health && !has_load {
            return None;
        }
        Some(WorkerProjection {
            state: self.state,
            now_ms: self.now_ms,
            freshness_window_ms: self.freshness_window_ms,
            region_id,
            worker_id,
        })
    }

    /// Aggregated client latency for one `(client_region, target_region)`
    /// pair. `Some` when at least one fresh replica is observed.
    pub fn client_latency(
        &self,
        client_region: &str,
        target_region: &str,
    ) -> Option<ClientLatencyProjection<'a>> {
        let mut min_p50 = None;
        let mut min_p95 = None;
        let mut age = None;
        let mut signal_owner: Option<&'a ClientLatencySignal> = None;
        for (signal, version) in self.state.client_latency_replicas(client_region, target_region) {
            if !is_fresh(self.now_ms, version, self.freshness_window_ms) {
                continue;
            }
            let entry_age = signal_age_ms(self.now_ms, version);
            age = Some(age.unwrap_or(0).max(entry_age));
            min_p50 = Some(
                min_p50
                    .map(|prev: u64| prev.min(signal.p50_latency_ms))
                    .unwrap_or(signal.p50_latency_ms),
            );
            min_p95 = Some(
                min_p95
                    .map(|prev: u64| prev.min(signal.p95_latency_ms))
                    .unwrap_or(signal.p95_latency_ms),
            );
            if signal_owner.is_none() {
                signal_owner = Some(signal);
            }
        }
        age.zip(min_p50).zip(min_p95).map(|((freshness_age_ms, p50), p95)| {
            ClientLatencyProjection {
                client_region: signal_owner.map(|s| s.client_region.as_str()).unwrap_or(""),
                target_region: signal_owner.map(|s| s.target_region.as_str()).unwrap_or(""),
                min_p50_ms: p50,
                min_p95_ms: p95,
                freshness_age_ms,
            }
        })
    }

    /// True when at least one fresh or stale client-latency replica exists
    /// for the pair.
    pub fn has_client_latency_replica(&self, client_region: &str, target_region: &str) -> bool {
        self.state
            .client_latency_replicas(client_region, target_region)
            .next()
            .is_some()
    }
}

/// Aggregated readiness for one region. `ready == true` when at least one
/// fresh replica reports ready (any-fresh-ready). `freshness_age_ms` is the
/// oldest contributing replica's age (the conservative reading).
#[derive(Debug, Clone, Copy)]
pub struct RegionReadinessProjection<'a> {
    pub region_id: &'a str,
    pub ready: bool,
    pub freshness_age_ms: i64,
}

/// Aggregated worker projection across replicas. Holds a borrow back into
/// state plus the worker's `(region, worker)` coordinates so accessors can
/// re-iterate replicas when callers ask for model-specific load.
#[derive(Debug)]
pub struct WorkerProjection<'a> {
    state: &'a CrossRegionState,
    now_ms: i64,
    freshness_window_ms: i64,
    region_id: &'a str,
    worker_id: &'a str,
}

impl<'a> WorkerProjection<'a> {
    pub fn region_id(&self) -> &'a str {
        self.region_id
    }

    pub fn worker_id(&self) -> &'a str {
        self.worker_id
    }

    /// Aggregated worker status across fresh replicas — any routable
    /// observation wins. `None` when no fresh health replica exists for the
    /// worker (caller can read `worker_health_status_discriminants` to tell
    /// "stale" from "absent").
    pub fn aggregated_status(&self) -> WorkerStatusResolution {
        let mut chosen: Option<WorkerStatus> = None;
        let mut age: Option<i64> = None;
        let mut saw_replica = false;
        let mut saw_stale = false;
        for (signal, version) in self
            .state
            .worker_health_replicas(self.region_id, self.worker_id)
        {
            saw_replica = true;
            if !is_fresh(self.now_ms, version, self.freshness_window_ms) {
                saw_stale = true;
                continue;
            }
            let entry_age = signal_age_ms(self.now_ms, version);
            age = Some(age.unwrap_or(0).max(entry_age));
            chosen = match chosen {
                Some(current) if current.is_routable() => Some(current),
                _ => Some(signal.status),
            };
        }
        WorkerStatusResolution {
            status: chosen,
            freshness_age_ms: age,
            saw_replica,
            saw_stale_only: saw_replica && chosen.is_none() && saw_stale,
        }
    }

    /// Aggregated load for the requested `model_id` across fresh replicas.
    /// Each replica's load is its own dispatcher's disjoint in-flight count,
    /// so summing is the correct "total observed load" semantics for v1.
    pub fn load_for_model(&self, model_id: &str) -> WorkerLoadResolution {
        let mut total: Option<isize> = None;
        let mut age: Option<i64> = None;
        let mut saw_replica = false;
        let mut saw_model_mismatch = false;
        let mut saw_stale_match = false;
        for (signal, version) in self
            .state
            .worker_load_replicas(self.region_id, self.worker_id)
        {
            saw_replica = true;
            match signal.load.model_id.as_deref() {
                Some(remote) if remote == model_id => {}
                Some(_) => {
                    saw_model_mismatch = true;
                    continue;
                }
                None => continue,
            }
            if !is_fresh(self.now_ms, version, self.freshness_window_ms) {
                saw_stale_match = true;
                continue;
            }
            let entry_age = signal_age_ms(self.now_ms, version);
            age = Some(age.unwrap_or(0).max(entry_age));
            total = Some(total.unwrap_or(0).saturating_add(signal.load.load));
        }
        WorkerLoadResolution {
            total,
            freshness_age_ms: age,
            saw_replica,
            saw_model_mismatch,
            saw_stale_match,
        }
    }
}

/// Result of resolving an aggregated worker status. Callers consult the
/// discriminants to distinguish "no replica" from "all stale" — important
/// for translating into rejection reasons on the candidate-ranking path.
#[derive(Debug, Clone, Copy)]
pub struct WorkerStatusResolution {
    /// Aggregated status when at least one fresh replica was observed.
    pub status: Option<WorkerStatus>,
    /// Oldest contributing replica's age (worst-case freshness).
    pub freshness_age_ms: Option<i64>,
    /// True when at least one replica (fresh or stale) exists for the worker.
    pub saw_replica: bool,
    /// True when every observed replica was stale.
    pub saw_stale_only: bool,
}

/// Result of resolving aggregated load for one `(worker, model_id)` pair.
/// Discriminants let callers translate to `ModelNotAllowed` vs
/// `StaleRemoteSignal` vs `MissingRemoteSignal` without re-reading state.
#[derive(Debug, Clone, Copy)]
pub struct WorkerLoadResolution {
    /// Sum of fresh matching-model loads, or `None` when no fresh match.
    pub total: Option<isize>,
    /// Oldest contributing replica's age (worst-case freshness).
    pub freshness_age_ms: Option<i64>,
    /// True when at least one load replica (any model) was observed.
    pub saw_replica: bool,
    /// True when at least one replica reported a non-matching `model_id`.
    pub saw_model_mismatch: bool,
    /// True when at least one matching-model replica existed but was stale.
    pub saw_stale_match: bool,
}

/// Aggregated client latency for a `(client_region, target_region)` pair.
/// `min_p50_ms` / `min_p95_ms` are the minimum across fresh replicas — the
/// best-case observed path. The strategy is documented here so a histogram
/// projection can replace it cleanly.
#[derive(Debug, Clone, Copy)]
pub struct ClientLatencyProjection<'a> {
    pub client_region: &'a str,
    pub target_region: &'a str,
    pub min_p50_ms: u64,
    pub min_p95_ms: u64,
    pub freshness_age_ms: i64,
}

fn is_fresh(now_ms: i64, version: &SignalVersion, max_age_ms: i64) -> bool {
    let age = signal_age_ms(now_ms, version);
    version.updated_at_ms >= 0 && age <= max_age_ms
}

fn signal_age_ms(now_ms: i64, version: &SignalVersion) -> i64 {
    if version.updated_at_ms < 0 {
        0
    } else {
        now_ms.saturating_sub(version.updated_at_ms).max(0)
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::worker::{WorkerLoadInfo, WorkerStatus};

    use super::*;
    use crate::cross_region::{
        SignalVersion, SmgReadinessSignal, WorkerHealthSignal, WorkerLoadSignal,
    };

    const NOW_MS: i64 = 1_000_000;
    const WINDOW_MS: i64 = 30_000;

    fn version(version: u64, actor: &str, updated_at_ms: i64) -> SignalVersion {
        SignalVersion {
            version,
            actor: actor.to_string(),
            updated_at_ms,
        }
    }

    fn fresh_version(actor: &str) -> SignalVersion {
        version(1, actor, NOW_MS - 1_000)
    }

    fn stale_version(actor: &str) -> SignalVersion {
        version(1, actor, NOW_MS - WINDOW_MS - 1_000)
    }

    fn upsert_readiness(state: &mut CrossRegionState, region: &str, server: &str, ready: bool, v: SignalVersion) {
        state.upsert_readiness(
            SmgReadinessSignal {
                region_id: region.to_string(),
                server_name: server.to_string(),
                ready,
            },
            v,
        );
    }

    fn upsert_worker_health(
        state: &mut CrossRegionState,
        region: &str,
        worker: &str,
        server: &str,
        status: WorkerStatus,
        v: SignalVersion,
    ) {
        state.upsert_worker_health(
            WorkerHealthSignal {
                region_id: region.to_string(),
                worker_id: worker.to_string(),
                server_name: server.to_string(),
                status,
            },
            v,
        );
    }

    fn upsert_worker_load(
        state: &mut CrossRegionState,
        region: &str,
        worker: &str,
        server: &str,
        model_id: Option<&str>,
        load: isize,
        v: SignalVersion,
    ) {
        state.upsert_worker_load(
            WorkerLoadSignal {
                region_id: region.to_string(),
                worker_id: worker.to_string(),
                server_name: server.to_string(),
                load: WorkerLoadInfo {
                    worker: worker.to_string(),
                    worker_type: None,
                    load,
                    details: None,
                    region_id: Some(region.to_string()),
                    worker_id: Some(worker.to_string()),
                    model_id: model_id.map(str::to_string),
                    status: Some(WorkerStatus::Ready),
                    generated_at_ms: Some(v.updated_at_ms),
                    version: Some(1),
                    source: None,
                    remote_workers: None,
                },
            },
            v,
        );
    }

    fn upsert_client_latency(
        state: &mut CrossRegionState,
        client: &str,
        target: &str,
        server: &str,
        p50: u64,
        p95: u64,
        v: SignalVersion,
    ) {
        state.upsert_client_latency(
            ClientLatencySignal {
                client_region: client.to_string(),
                target_region: target.to_string(),
                server_name: server.to_string(),
                p50_latency_ms: p50,
                p95_latency_ms: p95,
            },
            v,
        );
    }

    #[test]
    fn readiness_aggregates_any_fresh_ready() {
        let mut state = CrossRegionState::new();
        upsert_readiness(&mut state, "r1", "smg-a", true, fresh_version("smg-a"));
        upsert_readiness(&mut state, "r1", "smg-b", false, fresh_version("smg-b"));

        let view = RemoteRegionView::new(&state, NOW_MS, WINDOW_MS);
        let readiness = view.readiness("r1").expect("fresh replica present");
        assert!(readiness.ready, "any-fresh-ready wins");
        assert!(readiness.freshness_age_ms >= 1_000);
    }

    #[test]
    fn readiness_filters_stale_replicas() {
        let mut state = CrossRegionState::new();
        upsert_readiness(&mut state, "r1", "smg-a", true, stale_version("smg-a"));

        let view = RemoteRegionView::new(&state, NOW_MS, WINDOW_MS);
        assert!(view.readiness("r1").is_none(), "stale replica must not project");
        assert!(
            view.has_readiness_replica("r1"),
            "discriminant distinguishes stale from absent"
        );
    }

    #[test]
    fn readiness_returns_none_when_no_replica_observed() {
        let state = CrossRegionState::new();
        let view = RemoteRegionView::new(&state, NOW_MS, WINDOW_MS);
        assert!(view.readiness("r1").is_none());
        assert!(!view.has_readiness_replica("r1"));
    }

    #[test]
    fn readiness_all_replicas_not_ready_projects_false() {
        let mut state = CrossRegionState::new();
        upsert_readiness(&mut state, "r1", "smg-a", false, fresh_version("smg-a"));
        upsert_readiness(&mut state, "r1", "smg-b", false, fresh_version("smg-b"));

        let view = RemoteRegionView::new(&state, NOW_MS, WINDOW_MS);
        let readiness = view.readiness("r1").expect("fresh replicas exist");
        assert!(!readiness.ready);
    }

    #[test]
    fn worker_status_any_fresh_routable_wins() {
        let mut state = CrossRegionState::new();
        upsert_worker_health(
            &mut state,
            "r1",
            "w1",
            "smg-a",
            WorkerStatus::Pending,
            fresh_version("smg-a"),
        );
        upsert_worker_health(
            &mut state,
            "r1",
            "w1",
            "smg-b",
            WorkerStatus::Ready,
            fresh_version("smg-b"),
        );

        let view = RemoteRegionView::new(&state, NOW_MS, WINDOW_MS);
        let projection = view.worker("r1", "w1").expect("worker observed");
        let status = projection.aggregated_status();
        assert_eq!(status.status, Some(WorkerStatus::Ready));
        assert!(status.saw_replica);
        assert!(!status.saw_stale_only);
    }

    #[test]
    fn worker_load_sums_fresh_matching_model_replicas() {
        let mut state = CrossRegionState::new();
        upsert_worker_load(
            &mut state,
            "r1",
            "w1",
            "smg-a",
            Some("model-x"),
            3,
            fresh_version("smg-a"),
        );
        upsert_worker_load(
            &mut state,
            "r1",
            "w1",
            "smg-b",
            Some("model-x"),
            4,
            fresh_version("smg-b"),
        );

        let view = RemoteRegionView::new(&state, NOW_MS, WINDOW_MS);
        let projection = view.worker("r1", "w1").expect("worker observed");
        let load = projection.load_for_model("model-x");
        assert_eq!(load.total, Some(7));
        assert!(load.saw_replica);
        assert!(!load.saw_model_mismatch);
        assert!(!load.saw_stale_match);
    }

    #[test]
    fn worker_load_ignores_other_models_but_flags_mismatch() {
        let mut state = CrossRegionState::new();
        upsert_worker_load(
            &mut state,
            "r1",
            "w1",
            "smg-a",
            Some("model-y"),
            3,
            fresh_version("smg-a"),
        );

        let view = RemoteRegionView::new(&state, NOW_MS, WINDOW_MS);
        let projection = view.worker("r1", "w1").expect("worker observed");
        let load = projection.load_for_model("model-x");
        assert!(load.total.is_none());
        assert!(load.saw_model_mismatch);
        assert!(!load.saw_stale_match);
    }

    #[test]
    fn worker_load_stale_match_does_not_count_toward_total() {
        let mut state = CrossRegionState::new();
        upsert_worker_load(
            &mut state,
            "r1",
            "w1",
            "smg-a",
            Some("model-x"),
            3,
            stale_version("smg-a"),
        );

        let view = RemoteRegionView::new(&state, NOW_MS, WINDOW_MS);
        let projection = view.worker("r1", "w1").expect("worker observed");
        let load = projection.load_for_model("model-x");
        assert!(load.total.is_none());
        assert!(load.saw_stale_match);
    }

    #[test]
    fn client_latency_min_across_fresh_replicas() {
        let mut state = CrossRegionState::new();
        upsert_client_latency(
            &mut state,
            "c1",
            "r1",
            "smg-a",
            120,
            300,
            fresh_version("smg-a"),
        );
        upsert_client_latency(
            &mut state,
            "c1",
            "r1",
            "smg-b",
            80,
            250,
            fresh_version("smg-b"),
        );

        let view = RemoteRegionView::new(&state, NOW_MS, WINDOW_MS);
        let projection = view.client_latency("c1", "r1").expect("fresh replicas");
        assert_eq!(projection.min_p50_ms, 80);
        assert_eq!(projection.min_p95_ms, 250);
    }

    #[test]
    fn client_latency_returns_none_when_only_stale_observations() {
        let mut state = CrossRegionState::new();
        upsert_client_latency(
            &mut state,
            "c1",
            "r1",
            "smg-a",
            120,
            300,
            stale_version("smg-a"),
        );

        let view = RemoteRegionView::new(&state, NOW_MS, WINDOW_MS);
        assert!(view.client_latency("c1", "r1").is_none());
        assert!(view.has_client_latency_replica("c1", "r1"));
    }
}
