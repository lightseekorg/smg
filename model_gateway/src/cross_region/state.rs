use std::collections::{BTreeSet, HashMap};

use serde::{Deserialize, Serialize};

use super::{
    ClientLatencySignal, SignalKey, SmgReadinessSignal, WorkerHealthSignal, WorkerLoadSignal,
};

/// Version and freshness metadata for a materialized signal.
///
/// `(version, actor)` is the apply-ordering key: higher `version` wins, and
/// equal `version` is broken by lexicographic `actor` to keep concurrent
/// writers deterministic.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignalVersion {
    pub version: u64,
    pub actor: String,
    pub updated_at_ms: i64,
}

/// Composite key for per-replica readiness storage: `(region_id, server_name)`.
type ReadinessKey = (String, String);

/// Composite key for per-replica worker-scoped storage:
/// `(region_id, worker_id, server_name)`.
type WorkerKey = (String, String, String);

/// Composite key for per-replica client-latency storage:
/// `(client_region, target_region, server_name)`.
type LatencyKey = (String, String, String);

/// In-memory materialized view for remote cross-region signals.
///
/// All four signal kinds are stored **per replica**: the writing replica's
/// `server_name` is part of the key, so two replicas in the same region (or
/// reporting on the same worker) do not overwrite each other and a tombstone
/// for one replica leaves sibling replicas intact.
///
/// Aggregation across replicas is the consumer's responsibility — use the
/// `*_replicas` iterators or the small set of explicit aggregate helpers.
/// A typed projection layer (`RemoteRegionView`) lands in Phase F and is the
/// preferred entry point for ranking/observability consumers.
#[derive(Debug, Clone, Default)]
pub struct CrossRegionState {
    readiness: HashMap<ReadinessKey, (SmgReadinessSignal, SignalVersion)>,
    worker_health: HashMap<WorkerKey, (WorkerHealthSignal, SignalVersion)>,
    worker_load: HashMap<WorkerKey, (WorkerLoadSignal, SignalVersion)>,
    client_latency: HashMap<LatencyKey, (ClientLatencySignal, SignalVersion)>,
}

impl CrossRegionState {
    /// Create an empty materialized signal state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return true when the materialized view has no remote signals.
    pub fn is_empty(&self) -> bool {
        self.readiness.is_empty()
            && self.worker_health.is_empty()
            && self.worker_load.is_empty()
            && self.client_latency.is_empty()
    }

    // ---------------------------------------------------------------------
    // Per-replica direct lookup
    // ---------------------------------------------------------------------

    /// Return one replica's readiness signal when present.
    pub fn readiness_replica(
        &self,
        region_id: &str,
        server_name: &str,
    ) -> Option<&SmgReadinessSignal> {
        self.readiness
            .get(&(region_id.to_string(), server_name.to_string()))
            .map(|(signal, _)| signal)
    }

    /// Return one replica's readiness signal with its freshness version.
    pub fn readiness_replica_with_version(
        &self,
        region_id: &str,
        server_name: &str,
    ) -> Option<(&SmgReadinessSignal, &SignalVersion)> {
        self.readiness
            .get(&(region_id.to_string(), server_name.to_string()))
            .map(|(signal, version)| (signal, version))
    }

    /// Return one replica's worker health signal when present.
    pub fn worker_health_replica(
        &self,
        region_id: &str,
        worker_id: &str,
        server_name: &str,
    ) -> Option<&WorkerHealthSignal> {
        self.worker_health
            .get(&(
                region_id.to_string(),
                worker_id.to_string(),
                server_name.to_string(),
            ))
            .map(|(signal, _)| signal)
    }

    /// Return one replica's worker health signal with its freshness version.
    pub fn worker_health_replica_with_version(
        &self,
        region_id: &str,
        worker_id: &str,
        server_name: &str,
    ) -> Option<(&WorkerHealthSignal, &SignalVersion)> {
        self.worker_health
            .get(&(
                region_id.to_string(),
                worker_id.to_string(),
                server_name.to_string(),
            ))
            .map(|(signal, version)| (signal, version))
    }

    /// Return one replica's worker load signal when present.
    pub fn worker_load_replica(
        &self,
        region_id: &str,
        worker_id: &str,
        server_name: &str,
    ) -> Option<&WorkerLoadSignal> {
        self.worker_load
            .get(&(
                region_id.to_string(),
                worker_id.to_string(),
                server_name.to_string(),
            ))
            .map(|(signal, _)| signal)
    }

    /// Return one replica's worker load signal with its freshness version.
    pub fn worker_load_replica_with_version(
        &self,
        region_id: &str,
        worker_id: &str,
        server_name: &str,
    ) -> Option<(&WorkerLoadSignal, &SignalVersion)> {
        self.worker_load
            .get(&(
                region_id.to_string(),
                worker_id.to_string(),
                server_name.to_string(),
            ))
            .map(|(signal, version)| (signal, version))
    }

    /// Return one replica's client latency signal when present.
    pub fn client_latency_replica(
        &self,
        client_region: &str,
        target_region: &str,
        server_name: &str,
    ) -> Option<&ClientLatencySignal> {
        self.client_latency
            .get(&(
                client_region.to_string(),
                target_region.to_string(),
                server_name.to_string(),
            ))
            .map(|(signal, _)| signal)
    }

    /// Return one replica's client latency signal with its freshness version.
    pub fn client_latency_replica_with_version(
        &self,
        client_region: &str,
        target_region: &str,
        server_name: &str,
    ) -> Option<(&ClientLatencySignal, &SignalVersion)> {
        self.client_latency
            .get(&(
                client_region.to_string(),
                target_region.to_string(),
                server_name.to_string(),
            ))
            .map(|(signal, version)| (signal, version))
    }

    // ---------------------------------------------------------------------
    // All-replicas iterators (deterministic order by `server_name`)
    // ---------------------------------------------------------------------

    /// Iterate every replica's readiness signal for a region.
    pub fn readiness_replicas(
        &self,
        region_id: &str,
    ) -> impl Iterator<Item = (&SmgReadinessSignal, &SignalVersion)> {
        replicas_in_region(&self.readiness, region_id, |(region, _)| region.as_str())
    }

    /// Iterate every replica's worker health signal for a `(region, worker)` pair.
    pub fn worker_health_replicas(
        &self,
        region_id: &str,
        worker_id: &str,
    ) -> impl Iterator<Item = (&WorkerHealthSignal, &SignalVersion)> {
        replicas_in_worker(&self.worker_health, region_id, worker_id)
    }

    /// Iterate every replica's worker load signal for a `(region, worker)` pair.
    pub fn worker_load_replicas(
        &self,
        region_id: &str,
        worker_id: &str,
    ) -> impl Iterator<Item = (&WorkerLoadSignal, &SignalVersion)> {
        replicas_in_worker(&self.worker_load, region_id, worker_id)
    }

    /// Iterate every replica's client latency signal for a `(client, target)` pair.
    pub fn client_latency_replicas(
        &self,
        client_region: &str,
        target_region: &str,
    ) -> impl Iterator<Item = (&ClientLatencySignal, &SignalVersion)> {
        let mut entries: Vec<_> = self
            .client_latency
            .iter()
            .filter(|((client, target, _), _)| client == client_region && target == target_region)
            .collect();
        entries.sort_by(|a, b| a.0 .2.cmp(&b.0 .2));
        entries
            .into_iter()
            .map(|(_, (signal, version))| (signal, version))
    }

    // ---------------------------------------------------------------------
    // Enumeration helpers
    // ---------------------------------------------------------------------

    /// Return all regions represented by materialized remote signals in stable order.
    pub fn regions(&self) -> Vec<&str> {
        let mut regions = BTreeSet::new();
        regions.extend(self.readiness.keys().map(|(region, _)| region.as_str()));
        regions.extend(
            self.worker_health
                .keys()
                .map(|(region, _, _)| region.as_str()),
        );
        regions.extend(
            self.worker_load
                .keys()
                .map(|(region, _, _)| region.as_str()),
        );
        regions.extend(
            self.client_latency
                .keys()
                .map(|(_, target, _)| target.as_str()),
        );
        regions.into_iter().collect()
    }

    /// Return all worker ids represented by health or load signals for one region.
    pub fn worker_ids(&self, region_id: &str) -> Vec<&str> {
        let mut worker_ids = BTreeSet::new();
        worker_ids.extend(
            self.worker_health
                .keys()
                .filter_map(|(region, worker, _)| (region == region_id).then_some(worker.as_str())),
        );
        worker_ids.extend(
            self.worker_load
                .keys()
                .filter_map(|(region, worker, _)| (region == region_id).then_some(worker.as_str())),
        );
        worker_ids.into_iter().collect()
    }

    /// Return all replica `server_name`s observed for a region's readiness.
    pub fn readiness_replica_names(&self, region_id: &str) -> Vec<&str> {
        let mut names = BTreeSet::new();
        names.extend(
            self.readiness
                .keys()
                .filter_map(|(region, server)| (region == region_id).then_some(server.as_str())),
        );
        names.into_iter().collect()
    }

    // ---------------------------------------------------------------------
    // Apply / upsert
    // ---------------------------------------------------------------------

    /// Insert or replace a readiness signal in the materialized view.
    ///
    /// Keyed by `(region_id, server_name)` from the signal body, so two
    /// replicas in the same region coexist.
    pub fn upsert_readiness(&mut self, signal: SmgReadinessSignal, version: SignalVersion) {
        let key = (signal.region_id.clone(), signal.server_name.clone());
        if should_replace(self.readiness.get(&key).map(|(_, v)| v), &version) {
            self.readiness.insert(key, (signal, version));
        }
    }

    /// Insert or replace a worker health signal in the materialized view.
    pub fn upsert_worker_health(&mut self, signal: WorkerHealthSignal, version: SignalVersion) {
        let key = (
            signal.region_id.clone(),
            signal.worker_id.clone(),
            signal.server_name.clone(),
        );
        if should_replace(self.worker_health.get(&key).map(|(_, v)| v), &version) {
            self.worker_health.insert(key, (signal, version));
        }
    }

    /// Insert or replace a worker load signal in the materialized view.
    pub fn upsert_worker_load(&mut self, signal: WorkerLoadSignal, version: SignalVersion) {
        let key = (
            signal.region_id.clone(),
            signal.worker_id.clone(),
            signal.server_name.clone(),
        );
        if should_replace(self.worker_load.get(&key).map(|(_, v)| v), &version) {
            self.worker_load.insert(key, (signal, version));
        }
    }

    /// Insert or replace a client latency signal in the materialized view.
    pub fn upsert_client_latency(&mut self, signal: ClientLatencySignal, version: SignalVersion) {
        let key = (
            signal.client_region.clone(),
            signal.target_region.clone(),
            signal.server_name.clone(),
        );
        if should_replace(self.client_latency.get(&key).map(|(_, v)| v), &version) {
            self.client_latency.insert(key, (signal, version));
        }
    }

    /// Drop every materialized entry whose `updated_at_ms` is older than
    /// `now_ms - max_age_ms`. Returns the total number of entries evicted.
    ///
    /// Stream-based sync delivers no tombstones — when a producer stops
    /// re-emitting (worker deregistered, latency target removed, etc.) the
    /// entry would otherwise live forever in the materialized state, leak
    /// memory, and clutter diagnostic enumerations like
    /// [`Self::worker_ids`]. A periodic GC sweep against the consumer's
    /// freshness window is the bound that compensates.
    pub fn gc_stale(&mut self, now_ms: i64, max_age_ms: i64) -> usize {
        let cutoff = now_ms.saturating_sub(max_age_ms);
        let mut dropped = 0;
        let before = self.readiness.len();
        self.readiness.retain(|_, (_, v)| v.updated_at_ms >= cutoff);
        dropped += before - self.readiness.len();
        let before = self.worker_health.len();
        self.worker_health
            .retain(|_, (_, v)| v.updated_at_ms >= cutoff);
        dropped += before - self.worker_health.len();
        let before = self.worker_load.len();
        self.worker_load
            .retain(|_, (_, v)| v.updated_at_ms >= cutoff);
        dropped += before - self.worker_load.len();
        let before = self.client_latency.len();
        self.client_latency
            .retain(|_, (_, v)| v.updated_at_ms >= cutoff);
        dropped += before - self.client_latency.len();
        dropped
    }

    /// Remove the materialized value addressed by a tombstone key.
    ///
    /// Removes **only** the exact per-replica entry — sibling replicas keep
    /// their values.
    pub fn remove_key(&mut self, key: &SignalKey) {
        match key {
            SignalKey::SmgReadiness {
                region_id,
                server_name,
            } => {
                self.readiness
                    .remove(&(region_id.clone(), server_name.clone()));
            }
            SignalKey::WorkerHealth {
                region_id,
                worker_id,
                server_name,
            } => {
                self.worker_health.remove(&(
                    region_id.clone(),
                    worker_id.clone(),
                    server_name.clone(),
                ));
            }
            SignalKey::WorkerLoad {
                region_id,
                worker_id,
                server_name,
            } => {
                self.worker_load.remove(&(
                    region_id.clone(),
                    worker_id.clone(),
                    server_name.clone(),
                ));
            }
            SignalKey::ClientLatency {
                client_region,
                target_region,
                server_name,
            } => {
                self.client_latency.remove(&(
                    client_region.clone(),
                    target_region.clone(),
                    server_name.clone(),
                ));
            }
        }
    }

    /// Remove the materialized value addressed by a tombstone key when the
    /// tombstone's `(version, actor)` outranks the current live entry.
    ///
    /// This preserves apply ordering for delete-after-update races: an older
    /// tombstone arriving late cannot erase a newer live signal.
    pub fn remove_key_with_version(&mut self, key: &SignalKey, version: &SignalVersion) {
        match key {
            SignalKey::SmgReadiness {
                region_id,
                server_name,
            } => {
                let storage_key = (region_id.clone(), server_name.clone());
                if should_replace(self.readiness.get(&storage_key).map(|(_, v)| v), version) {
                    self.readiness.remove(&storage_key);
                }
            }
            SignalKey::WorkerHealth {
                region_id,
                worker_id,
                server_name,
            } => {
                let storage_key = (region_id.clone(), worker_id.clone(), server_name.clone());
                if should_replace(
                    self.worker_health.get(&storage_key).map(|(_, v)| v),
                    version,
                ) {
                    self.worker_health.remove(&storage_key);
                }
            }
            SignalKey::WorkerLoad {
                region_id,
                worker_id,
                server_name,
            } => {
                let storage_key = (region_id.clone(), worker_id.clone(), server_name.clone());
                if should_replace(self.worker_load.get(&storage_key).map(|(_, v)| v), version) {
                    self.worker_load.remove(&storage_key);
                }
            }
            SignalKey::ClientLatency {
                client_region,
                target_region,
                server_name,
            } => {
                let storage_key = (
                    client_region.clone(),
                    target_region.clone(),
                    server_name.clone(),
                );
                if should_replace(
                    self.client_latency.get(&storage_key).map(|(_, v)| v),
                    version,
                ) {
                    self.client_latency.remove(&storage_key);
                }
            }
        }
    }
}

fn replicas_in_region<'a, S>(
    map: &'a HashMap<ReadinessKey, (S, SignalVersion)>,
    region_id: &str,
    region_of: impl Fn(&ReadinessKey) -> &str,
) -> impl Iterator<Item = (&'a S, &'a SignalVersion)> {
    let mut entries: Vec<_> = map
        .iter()
        .filter(|(key, _)| region_of(key) == region_id)
        .collect();
    entries.sort_by(|a, b| a.0 .1.cmp(&b.0 .1));
    entries
        .into_iter()
        .map(|(_, (signal, version))| (signal, version))
}

fn replicas_in_worker<'a, S>(
    map: &'a HashMap<WorkerKey, (S, SignalVersion)>,
    region_id: &str,
    worker_id: &str,
) -> impl Iterator<Item = (&'a S, &'a SignalVersion)> {
    let mut entries: Vec<_> = map
        .iter()
        .filter(|((region, worker, _), _)| region == region_id && worker == worker_id)
        .collect();
    entries.sort_by(|a, b| a.0 .2.cmp(&b.0 .2));
    entries
        .into_iter()
        .map(|(_, (signal, version))| (signal, version))
}

/// `(version, actor)` lexicographic apply rule. Higher `version` wins; equal
/// version is broken by lexicographic `actor`. Equal `(version, actor)` is
/// rejected so apply is idempotent for the same writer.
fn should_replace(current: Option<&SignalVersion>, incoming: &SignalVersion) -> bool {
    match current {
        None => true,
        Some(cur) => {
            (incoming.version, incoming.actor.as_str()) > (cur.version, cur.actor.as_str())
        }
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::worker::{WorkerLoadInfo, WorkerStatus};

    use super::*;

    fn readiness_signal(region_id: &str, server_name: &str, ready: bool) -> SmgReadinessSignal {
        SmgReadinessSignal {
            region_id: region_id.to_string(),
            server_name: server_name.to_string(),
            ready,
        }
    }

    fn worker_health_signal(
        region_id: &str,
        worker_id: &str,
        server_name: &str,
        status: WorkerStatus,
    ) -> WorkerHealthSignal {
        WorkerHealthSignal {
            region_id: region_id.to_string(),
            worker_id: worker_id.to_string(),
            server_name: server_name.to_string(),
            status,
        }
    }

    fn worker_load_signal(
        region_id: &str,
        worker_id: &str,
        server_name: &str,
        load: isize,
    ) -> WorkerLoadSignal {
        WorkerLoadSignal {
            region_id: region_id.to_string(),
            worker_id: worker_id.to_string(),
            server_name: server_name.to_string(),
            load: WorkerLoadInfo {
                worker: worker_id.to_string(),
                worker_type: None,
                load,
                details: None,
                region_id: Some(region_id.to_string()),
                worker_id: Some(worker_id.to_string()),
                model_id: None,
                status: Some(WorkerStatus::Ready),
                generated_at_ms: Some(0),
                version: Some(1),
                source: None,
                remote_workers: None,
            },
        }
    }

    fn version(version: u64, actor: &str, updated_at_ms: i64) -> SignalVersion {
        SignalVersion {
            version,
            actor: actor.to_string(),
            updated_at_ms,
        }
    }

    #[test]
    fn new_state_is_empty() {
        let state = CrossRegionState::new();

        assert!(state.is_empty());
        assert!(state.readiness_replica("us-chicago-1", "smg-a").is_none());
    }

    #[test]
    fn upsert_readiness_ignores_stale_or_equal_versions() {
        let mut state = CrossRegionState::new();
        let region_id = "us-ashburn-1";
        let server_name = "smg-router-a";

        state.upsert_readiness(
            readiness_signal(region_id, server_name, true),
            version(10, "actor-a", 100),
        );
        state.upsert_readiness(
            readiness_signal(region_id, server_name, false),
            version(9, "actor-a", 90),
        );
        state.upsert_readiness(
            readiness_signal(region_id, server_name, false),
            version(10, "actor-a", 110),
        );

        assert!(
            state
                .readiness_replica(region_id, server_name)
                .expect("readiness should exist")
                .ready,
            "stale and equal (version, actor) must not overwrite current state",
        );

        state.upsert_readiness(
            readiness_signal(region_id, server_name, false),
            version(11, "actor-a", 120),
        );

        assert!(
            !state
                .readiness_replica(region_id, server_name)
                .expect("readiness should exist")
                .ready,
            "newer versions should replace current state",
        );
    }

    #[test]
    fn older_version_is_rejected() {
        let mut state = CrossRegionState::new();
        state.upsert_readiness(
            readiness_signal("r1", "smg-a", true),
            version(5, "actor-a", 50),
        );
        state.upsert_readiness(
            readiness_signal("r1", "smg-a", false),
            version(4, "actor-a", 60),
        );

        assert!(
            state
                .readiness_replica("r1", "smg-a")
                .expect("present")
                .ready
        );
    }

    #[test]
    fn equal_version_higher_actor_wins() {
        let mut state = CrossRegionState::new();
        state.upsert_readiness(
            readiness_signal("r1", "smg-a", true),
            version(5, "actor-a", 50),
        );
        state.upsert_readiness(
            readiness_signal("r1", "smg-a", false),
            version(5, "actor-b", 50),
        );

        assert!(
            !state
                .readiness_replica("r1", "smg-a")
                .expect("present")
                .ready,
            "equal version with lex-greater actor should win",
        );
    }

    #[test]
    fn equal_version_lower_actor_is_rejected() {
        let mut state = CrossRegionState::new();
        state.upsert_readiness(
            readiness_signal("r1", "smg-a", true),
            version(5, "actor-b", 50),
        );
        state.upsert_readiness(
            readiness_signal("r1", "smg-a", false),
            version(5, "actor-a", 50),
        );

        assert!(
            state
                .readiness_replica("r1", "smg-a")
                .expect("present")
                .ready,
            "equal version with lex-smaller actor must not overwrite",
        );
    }

    #[test]
    fn idempotent_apply_same_envelope() {
        let mut state = CrossRegionState::new();
        let v = version(5, "actor-a", 50);
        state.upsert_readiness(readiness_signal("r1", "smg-a", true), v.clone());
        state.upsert_readiness(readiness_signal("r1", "smg-a", false), v.clone());

        assert!(
            state
                .readiness_replica("r1", "smg-a")
                .expect("present")
                .ready,
            "second apply with identical (version, actor) must be a no-op",
        );
    }

    #[test]
    fn actor_carries_through_into_accessor() {
        let mut state = CrossRegionState::new();
        state.upsert_readiness(
            readiness_signal("r1", "smg-a", true),
            version(5, "actor-a", 50),
        );

        let (_, returned) = state
            .readiness_replica_with_version("r1", "smg-a")
            .expect("readiness should exist");
        assert_eq!(returned.version, 5);
        assert_eq!(returned.actor, "actor-a");
        assert_eq!(returned.updated_at_ms, 50);
    }

    #[test]
    fn two_readiness_replicas_in_one_region_do_not_overwrite() {
        let mut state = CrossRegionState::new();
        state.upsert_readiness(
            readiness_signal("r1", "smg-a", true),
            version(1, "smg-a", 10),
        );
        state.upsert_readiness(
            readiness_signal("r1", "smg-b", false),
            version(1, "smg-b", 11),
        );

        assert!(state.readiness_replica("r1", "smg-a").expect("a").ready);
        assert!(!state.readiness_replica("r1", "smg-b").expect("b").ready);

        let replicas: Vec<_> = state
            .readiness_replicas("r1")
            .map(|(signal, _)| (signal.server_name.as_str(), signal.ready))
            .collect();
        assert_eq!(replicas, vec![("smg-a", true), ("smg-b", false)]);
    }

    #[test]
    fn two_worker_load_replicas_for_same_worker_do_not_overwrite() {
        let mut state = CrossRegionState::new();
        state.upsert_worker_load(
            worker_load_signal("r1", "w1", "smg-a", 2),
            version(1, "smg-a", 10),
        );
        state.upsert_worker_load(
            worker_load_signal("r1", "w1", "smg-b", 3),
            version(1, "smg-b", 11),
        );

        let loads: Vec<isize> = state
            .worker_load_replicas("r1", "w1")
            .map(|(signal, _)| signal.load.load)
            .collect();
        assert_eq!(loads, vec![2, 3]);
    }

    #[test]
    fn per_replica_tombstone_does_not_remove_sibling_replica() {
        let mut state = CrossRegionState::new();
        state.upsert_worker_health(
            worker_health_signal("r1", "w1", "smg-a", WorkerStatus::Ready),
            version(1, "smg-a", 10),
        );
        state.upsert_worker_health(
            worker_health_signal("r1", "w1", "smg-b", WorkerStatus::Ready),
            version(1, "smg-b", 11),
        );

        state.remove_key(&SignalKey::WorkerHealth {
            region_id: "r1".to_string(),
            worker_id: "w1".to_string(),
            server_name: "smg-a".to_string(),
        });

        assert!(state.worker_health_replica("r1", "w1", "smg-a").is_none());
        assert!(state.worker_health_replica("r1", "w1", "smg-b").is_some());
    }

    #[test]
    fn gc_stale_evicts_entries_older_than_cutoff() {
        let mut state = CrossRegionState::new();
        state.upsert_readiness(
            readiness_signal("r1", "smg-a", true),
            version(1, "smg-a", 100),
        );
        state.upsert_readiness(
            readiness_signal("r1", "smg-b", true),
            version(1, "smg-b", 1_000),
        );
        state.upsert_worker_health(
            worker_health_signal("r1", "w1", "smg-a", WorkerStatus::Ready),
            version(1, "smg-a", 200),
        );
        state.upsert_client_latency(
            ClientLatencySignal {
                client_region: "r2".to_string(),
                target_region: "r1".to_string(),
                server_name: "smg-c".to_string(),
                p50_latency_ms: 10,
                p95_latency_ms: 20,
            },
            version(1, "smg-c", 2_000),
        );

        // Cutoff at 1_500: keep updated_at_ms >= 1_500. Two entries are
        // older (100, 200, 1_000); one newer (2_000).
        let dropped = state.gc_stale(2_000, 500);
        assert_eq!(dropped, 3);
        assert!(state.readiness_replica("r1", "smg-a").is_none());
        assert!(state.readiness_replica("r1", "smg-b").is_none());
        assert!(state.worker_health_replica("r1", "w1", "smg-a").is_none());
        assert!(state.client_latency_replica("r2", "r1", "smg-c").is_some());
    }

    #[test]
    fn gc_stale_is_noop_when_nothing_expired() {
        let mut state = CrossRegionState::new();
        state.upsert_readiness(
            readiness_signal("r1", "smg-a", true),
            version(1, "smg-a", 1_000),
        );
        let dropped = state.gc_stale(1_010, 100);
        assert_eq!(dropped, 0);
        assert!(state.readiness_replica("r1", "smg-a").is_some());
    }

    #[test]
    fn per_replica_version_rejection_holds_for_each_replica_independently() {
        let mut state = CrossRegionState::new();
        state.upsert_readiness(
            readiness_signal("r1", "smg-a", true),
            version(5, "smg-a", 50),
        );
        state.upsert_readiness(
            readiness_signal("r1", "smg-b", true),
            version(5, "smg-b", 50),
        );

        // Older version for smg-a only — should not change either replica.
        state.upsert_readiness(
            readiness_signal("r1", "smg-a", false),
            version(4, "smg-a", 60),
        );
        assert!(state.readiness_replica("r1", "smg-a").expect("a").ready);

        // Newer version for smg-b only — only smg-b flips.
        state.upsert_readiness(
            readiness_signal("r1", "smg-b", false),
            version(6, "smg-b", 70),
        );
        assert!(state.readiness_replica("r1", "smg-a").expect("a").ready);
        assert!(!state.readiness_replica("r1", "smg-b").expect("b").ready);
    }
}
