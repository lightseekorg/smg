use std::{sync::Arc, time::Duration};

use dashmap::DashMap;
use tokio::sync::broadcast;

use crate::{bus::EventBus, types::WorkerSnapshot};

pub struct MetricsStore {
    cache: DashMap<String, Arc<WorkerSnapshot>>,
    bus: Arc<EventBus>,
    staleness_threshold: Duration,
}

impl MetricsStore {
    pub fn new(bus: Arc<EventBus>, staleness_threshold: Duration) -> Self {
        Self {
            cache: DashMap::new(),
            bus,
            staleness_threshold,
        }
    }

    pub fn update(&self, mut new_snapshot: WorkerSnapshot) {
        let url = new_snapshot.url.clone();

        // Ensure timestamp is current
        new_snapshot.timestamp = std::time::SystemTime::now();

        let should_update = if let Some(mut entry) = self.cache.get_mut(&url) {
            let current = entry.value();

            // 1. Monotonic sequence number check for the same source.
            // seq_no = 0 means "unset" — the store auto-assigns; skip the check.
            // For explicit seq numbers (> 0), reject anything that isn't strictly newer.
            if new_snapshot.source == current.source
                && new_snapshot.seq_no > 0
                && new_snapshot.seq_no <= current.seq_no
            {
                return; // Ignore stale explicit-seq updates from the same source
            }

            // 2. Source priority check
            // Treat Err (clock skew, future timestamp) as "already stale" so a
            // future timestamp never permanently blocks lower-priority updates.
            let age = match current.timestamp.elapsed() {
                Ok(d) => d,
                Err(_) => self.staleness_threshold + Duration::from_secs(1),
            };
            if new_snapshot.source < current.source && age < self.staleness_threshold {
                // Reject lower priority update if the higher priority one is still fresh
                return;
            }

            // Merge custom metrics so lower-priority scrapes aren't fully lost
            let mut merged_custom = current.custom_metrics.clone();
            merged_custom.extend(new_snapshot.custom_metrics.clone());
            new_snapshot.custom_metrics = merged_custom;

            // Increment sequence number locally for subscribers to see a monotonic progression
            new_snapshot.seq_no = current.seq_no + 1;

            let arc_new = Arc::new(new_snapshot);
            *entry.value_mut() = arc_new.clone();
            Some(arc_new)
        } else {
            // First time seeing this worker
            if new_snapshot.seq_no == 0 {
                new_snapshot.seq_no = 1;
            }
            let arc_new = Arc::new(new_snapshot);
            self.cache.insert(url, arc_new.clone());
            Some(arc_new)
        };

        if let Some(snapshot) = should_update {
            self.bus.publish(snapshot);
        }
    }

    pub fn get(&self, url: &str) -> Option<Arc<WorkerSnapshot>> {
        self.cache
            .get(url)
            .map(|ref_multi| Arc::clone(ref_multi.value()))
    }

    pub fn get_all(&self) -> Vec<Arc<WorkerSnapshot>> {
        self.cache.iter().map(|kv| Arc::clone(kv.value())).collect()
    }

    /// Snapshot-then-subscribe API to eliminate subscriber cold starts.
    ///
    /// Returns `(receiver, initial_snapshot)`. The initial snapshot is captured
    /// *before* the receiver is created, so a concurrent update will be missed
    /// rather than duplicated. Callers should treat the initial snapshot as
    /// slightly stale and rely on the receiver for subsequent updates.
    pub fn subscribe(
        &self,
    ) -> (
        broadcast::Receiver<Arc<WorkerSnapshot>>,
        Vec<Arc<WorkerSnapshot>>,
    ) {
        // Take the snapshot first to avoid delivering the same update twice:
        // a concurrent update between subscribe() and get_all() would appear
        // in both the initial list and as a broadcast event if the order were
        // reversed.
        let initial_state = self.get_all();
        let rx = self.bus.subscribe();
        (rx, initial_state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{bus::EventBus, types::MetricSource};

    fn make_store() -> MetricsStore {
        let bus = Arc::new(EventBus::new(64));
        MetricsStore::new(bus, Duration::from_secs(60))
    }

    fn snap(url: &str, source: MetricSource, in_flight: isize) -> WorkerSnapshot {
        let mut s = WorkerSnapshot::new(url.to_string(), source);
        s.in_flight_requests = in_flight;
        s
    }

    #[test]
    fn test_basic_update_and_get() {
        let store = make_store();
        store.update(snap("http://w1", MetricSource::DirectScrape, 5));
        let got = store.get("http://w1").unwrap();
        assert_eq!(got.in_flight_requests, 5);
    }

    #[test]
    fn test_monotonic_seq_rejection() {
        let store = make_store();

        // First update — gets seq_no 1
        store.update(snap("http://w1", MetricSource::Piggyback, 3));

        // Craft a stale snapshot: seq_no = 1 equals the current stored seq_no (1),
        // so it satisfies `> 0 && <= current.seq_no` and gets rejected.
        let mut old = snap("http://w1", MetricSource::Piggyback, 99);
        old.seq_no = 1; // explicit seq_no = 1 == current (1) → rejected
        store.update(old);

        // Value should still be 3
        let got = store.get("http://w1").unwrap();
        // Note: seq_no 0 is treated as "unset" by the store; it still gets
        // stored on first insert. After that, the existing seq is 1.
        // The second update has seq_no 0 < 1 so it is rejected.
        assert_eq!(got.in_flight_requests, 3);
    }

    #[test]
    fn test_source_priority_rejection() {
        let store = make_store();

        // High priority (Piggyback = 100) stored first
        store.update(snap("http://w1", MetricSource::Piggyback, 10));

        // Low priority (Prometheus = 25) update — should be rejected while Piggyback is fresh
        store.update(snap("http://w1", MetricSource::Prometheus, 99));

        let got = store.get("http://w1").unwrap();
        assert_eq!(
            got.in_flight_requests, 10,
            "Prometheus should not overwrite fresh Piggyback"
        );
    }

    #[test]
    fn test_custom_metrics_merge() {
        let store = make_store();

        let mut s1 = WorkerSnapshot::new("http://w1".to_string(), MetricSource::Piggyback);
        s1.custom_metrics.insert("gpu_util".to_string(), 0.8);
        store.update(s1);

        let mut s2 = WorkerSnapshot::new("http://w1".to_string(), MetricSource::Piggyback);
        s2.custom_metrics.insert("cache_hit_rate".to_string(), 0.65);
        store.update(s2);

        let got = store.get("http://w1").unwrap();
        // Both keys survive after merge
        assert!(
            got.custom_metrics.contains_key("gpu_util"),
            "gpu_util should be preserved after merge"
        );
        assert!(
            got.custom_metrics.contains_key("cache_hit_rate"),
            "cache_hit_rate should be added by second update"
        );
    }

    #[tokio::test]
    async fn test_subscribe_returns_initial_state() {
        let store = make_store();
        store.update(snap("http://w1", MetricSource::DirectScrape, 7));
        store.update(snap("http://w2", MetricSource::DirectScrape, 3));

        let (_rx, initial) = store.subscribe();
        assert_eq!(initial.len(), 2);
        let urls: Vec<_> = initial.iter().map(|s| s.url.as_str()).collect();
        assert!(urls.contains(&"http://w1"));
        assert!(urls.contains(&"http://w2"));
    }

    #[tokio::test]
    async fn test_subscribe_receives_event_bus_update() {
        let store = make_store();
        let (mut rx, _initial) = store.subscribe();

        store.update(snap("http://w3", MetricSource::Piggyback, 42));

        let received = rx.recv().await.expect("should receive update on bus");
        assert_eq!(received.url, "http://w3");
        assert_eq!(received.in_flight_requests, 42);
    }

    #[test]
    fn test_get_all() {
        let store = make_store();
        store.update(snap("http://a", MetricSource::Piggyback, 1));
        store.update(snap("http://b", MetricSource::Piggyback, 2));
        let all = store.get_all();
        assert_eq!(all.len(), 2);
    }
}
