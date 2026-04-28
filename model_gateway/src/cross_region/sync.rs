use super::{
    ClientLatencySignal, CrossRegionResult, SmgReadinessSignal, WorkerHealthSignal,
    WorkerLoadSignal,
};

/// Signal events exchanged by the future sync plane.
#[derive(Debug, Clone)]
pub enum SyncEvent {
    SmgReadiness(SmgReadinessSignal),
    WorkerHealth(WorkerHealthSignal),
    WorkerLoad(WorkerLoadSignal),
    ClientLatency(ClientLatencySignal),
}

/// No-op sync service boundary for future signal publication/subscription.
#[derive(Debug, Clone, Default)]
pub struct CrossRegionSyncService {
    enabled: bool,
}

impl CrossRegionSyncService {
    /// Create a no-op sync service.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return no local signal events until sync collection is implemented.
    pub fn collect_local_signals(&self) -> Vec<SyncEvent> {
        if !self.enabled {
            return Vec::new();
        }
        Vec::new()
    }

    /// Accept a sync event without mutating state until idempotent apply is implemented.
    pub fn apply_event(&self, event: SyncEvent) -> CrossRegionResult<()> {
        if !self.enabled {
            return Ok(());
        }
        match event {
            SyncEvent::SmgReadiness(signal) if signal.region_id.trim().is_empty() => {
                return Err(super::CrossRegionError::InvalidConfig {
                    reason: "readiness signal region_id must not be empty".to_string(),
                });
            }
            SyncEvent::WorkerHealth(signal) if signal.region_id.trim().is_empty() => {
                return Err(super::CrossRegionError::InvalidConfig {
                    reason: "worker health signal region_id must not be empty".to_string(),
                });
            }
            SyncEvent::WorkerLoad(signal) if signal.region_id.trim().is_empty() => {
                return Err(super::CrossRegionError::InvalidConfig {
                    reason: "worker load signal region_id must not be empty".to_string(),
                });
            }
            SyncEvent::ClientLatency(signal) if signal.target_region.trim().is_empty() => {
                return Err(super::CrossRegionError::InvalidConfig {
                    reason: "client latency target_region must not be empty".to_string(),
                });
            }
            _ => {}
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_op_sync_collects_no_signals() {
        let service = CrossRegionSyncService::new();

        assert!(service.collect_local_signals().is_empty());
    }
}
