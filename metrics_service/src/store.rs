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

            // 1. Monotonic sequence number check for the same source
            if new_snapshot.source == current.source
                && new_snapshot.seq_no > 0
                && new_snapshot.seq_no <= current.seq_no
            {
                return; // Ignore older updates from the same source
            }

            // 2. Source priority check
            let age = current.timestamp.elapsed().unwrap_or_default();
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

    /// Snapshot-then-subscribe API to eliminate subscriber cold starts
    pub fn subscribe(
        &self,
    ) -> (
        broadcast::Receiver<Arc<WorkerSnapshot>>,
        Vec<Arc<WorkerSnapshot>>,
    ) {
        let rx = self.bus.subscribe();
        let initial_state = self.get_all();
        (rx, initial_state)
    }
}
