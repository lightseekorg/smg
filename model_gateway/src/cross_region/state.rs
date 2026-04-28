use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::{ClientLatencySignal, SmgReadinessSignal, WorkerHealthSignal, WorkerLoadSignal};

/// Version and freshness metadata for a materialized signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignalVersion {
    pub version: u64,
    pub updated_at_ms: i64,
}

/// In-memory materialized view for remote cross-region signals.
#[derive(Debug, Clone, Default)]
pub struct CrossRegionState {
    readiness: HashMap<String, (SmgReadinessSignal, SignalVersion)>,
    worker_health: HashMap<(String, String), (WorkerHealthSignal, SignalVersion)>,
    worker_load: HashMap<(String, String), (WorkerLoadSignal, SignalVersion)>,
    client_latency: HashMap<(String, String), (ClientLatencySignal, SignalVersion)>,
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

    /// Return the readiness signal for a region when present.
    pub fn readiness(&self, region_id: &str) -> Option<&SmgReadinessSignal> {
        self.readiness.get(region_id).map(|(signal, _)| signal)
    }

    /// Return the worker health signal for a region/worker when present.
    pub fn worker_health(&self, region_id: &str, worker_id: &str) -> Option<&WorkerHealthSignal> {
        self.worker_health
            .get(&(region_id.to_string(), worker_id.to_string()))
            .map(|(signal, _)| signal)
    }

    /// Return the worker load signal for a region/worker when present.
    pub fn worker_load(&self, region_id: &str, worker_id: &str) -> Option<&WorkerLoadSignal> {
        self.worker_load
            .get(&(region_id.to_string(), worker_id.to_string()))
            .map(|(signal, _)| signal)
    }

    /// Return the client latency signal for a client/target region pair when present.
    pub fn client_latency(
        &self,
        client_region: &str,
        target_region: &str,
    ) -> Option<&ClientLatencySignal> {
        self.client_latency
            .get(&(client_region.to_string(), target_region.to_string()))
            .map(|(signal, _)| signal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_state_is_empty() {
        let state = CrossRegionState::new();

        assert!(state.is_empty());
        assert!(state.readiness("us-chicago-1").is_none());
    }
}
