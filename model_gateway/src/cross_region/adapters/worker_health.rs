//! Worker-health adapter — emits `worker-health/{region}/{worker_id}/{server_name}`.
//!
//! Event-driven on `WorkerRegistry::subscribe_events()`. Each lifecycle
//! transition (`Registered`, `Replaced`, `StatusChanged`) republishes the
//! signal with the current status; `Removed` writes a tombstone via
//! `CrossRegionSyncService::remove_signal`.
//!
//! v1 doesn't enforce the `xr.origin = local | mesh` label discipline yet
//! (design §3b) — that filter is a follow-up. As long as
//! `WorkerSyncAdapter` and `WorkerHealthAdapter` are wired in the same
//! process and the gateway doesn't mesh-import workers, this v1 is correct.
//! The label filter becomes load-bearing only once cross-region mesh import
//! ships.

use std::sync::Arc;

use openai_protocol::worker::WorkerStatus;

use crate::{
    cross_region::{
        CrossRegionResult, CrossRegionSyncService, SignalKey, SignalKind, WorkerHealthSignal,
    },
    worker::{event::WorkerEvent, WorkerRegistry},
};

/// Default freshness window (design §4): worker-health uses 60 s with a 20 s
/// periodic reconcile, so a healthy worker that emits no status events still
/// stays fresh for two reconciles before consumers gate it out.
pub const DEFAULT_WORKER_HEALTH_STALE_AFTER_MS: u32 = 60_000;

/// Worker-health producer.
#[derive(Debug, Clone)]
pub struct WorkerHealthAdapter {
    sync: Arc<CrossRegionSyncService>,
    stale_after_ms: u32,
}

impl WorkerHealthAdapter {
    pub fn new(sync: Arc<CrossRegionSyncService>) -> Self {
        Self {
            sync,
            stale_after_ms: DEFAULT_WORKER_HEALTH_STALE_AFTER_MS,
        }
    }

    pub fn with_stale_after_ms(mut self, stale_after_ms: u32) -> Self {
        self.stale_after_ms = stale_after_ms;
        self
    }

    /// Publish health for one worker.
    pub fn publish_for(&self, worker_id: &str, status: WorkerStatus) -> CrossRegionResult<()> {
        let region_id = self.sync.region_id().to_string();
        let server_name = self.sync.server_name().to_string();
        let key = SignalKey::WorkerHealth {
            region_id: region_id.clone(),
            worker_id: worker_id.to_string(),
            server_name: server_name.clone(),
        };
        let body = WorkerHealthSignal {
            region_id,
            worker_id: worker_id.to_string(),
            server_name,
            status,
        };
        self.sync
            .publish_signal(key, SignalKind::WorkerHealth(body), self.stale_after_ms)
    }

    /// Emit a tombstone for one worker.
    pub fn remove_for(&self, worker_id: &str) -> CrossRegionResult<()> {
        let region_id = self.sync.region_id().to_string();
        let server_name = self.sync.server_name().to_string();
        let key = SignalKey::WorkerHealth {
            region_id,
            worker_id: worker_id.to_string(),
            server_name,
        };
        self.sync.remove_signal(key)
    }

    /// Dispatch a single registry event into the right publish/remove call.
    pub fn handle_event(&self, event: &WorkerEvent) -> CrossRegionResult<()> {
        match event {
            WorkerEvent::Registered { worker_id, worker }
            | WorkerEvent::StatusChanged {
                worker_id, worker, ..
            } => self.publish_for(worker_id.as_str(), worker.status()),
            WorkerEvent::Replaced { worker_id, new, .. } => {
                self.publish_for(worker_id.as_str(), new.status())
            }
            WorkerEvent::Removed { worker_id, .. } => self.remove_for(worker_id.as_str()),
        }
    }

    /// Republish every known worker's current health. Caller invokes this
    /// on a 20 s reconcile tick so stable healthy workers don't go stale
    /// past `stale_after_ms` between event-driven updates.
    pub fn reconcile(&self, registry: &WorkerRegistry) {
        for (worker_id, worker) in registry.get_all_with_ids() {
            if let Err(err) = self.publish_for(worker_id.as_str(), worker.status()) {
                tracing::warn!(
                    worker_id = worker_id.as_str(),
                    error = %err,
                    "worker-health reconcile publish failed"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use openai_protocol::model_card::ModelCard;

    use super::*;
    use crate::{
        cross_region::adapters::test_support::{live_envelopes, service, single_live},
        worker::{registry::WorkerId, BasicWorkerBuilder},
    };

    fn make_registry_with_worker(url: &str, status: WorkerStatus) -> (Arc<WorkerRegistry>, String) {
        let registry = Arc::new(WorkerRegistry::new());
        let worker = Arc::new(
            BasicWorkerBuilder::new(url)
                .model(ModelCard::new("cohere.command-r-plus"))
                .status(status)
                .build(),
        );
        let id = registry
            .register(worker)
            .expect("register")
            .as_str()
            .to_string();
        (registry, id)
    }

    #[test]
    fn publish_for_emits_correct_key_shape() {
        let svc = service();
        let adapter = WorkerHealthAdapter::new(svc.clone());

        adapter
            .publish_for("worker-uuid-1", WorkerStatus::Ready)
            .expect("publish ok");

        let env = single_live(&svc);
        match &env.key {
            SignalKey::WorkerHealth {
                region_id,
                worker_id,
                server_name,
            } => {
                assert_eq!(region_id, "us-ashburn-1");
                assert_eq!(worker_id, "worker-uuid-1");
                assert_eq!(server_name, "smg-router-a");
            }
            _ => panic!("unexpected key shape: {:?}", env.key),
        }
        assert_eq!(env.stale_after_ms, DEFAULT_WORKER_HEALTH_STALE_AFTER_MS);
        assert!(!env.removed);
    }

    #[test]
    fn handle_event_registered_publishes_status() {
        let svc = service();
        let adapter = WorkerHealthAdapter::new(svc.clone());
        let (registry, worker_id) =
            make_registry_with_worker("http://w1:8000", WorkerStatus::Ready);
        let worker = registry
            .get(&WorkerId::from_string(worker_id.clone()))
            .expect("registered");

        let event = WorkerEvent::Registered {
            worker_id: WorkerId::from_string(worker_id.clone()),
            worker,
        };
        adapter.handle_event(&event).expect("handle ok");

        let env = single_live(&svc);
        match env.signal {
            Some(SignalKind::WorkerHealth(s)) => {
                assert_eq!(s.status, WorkerStatus::Ready);
                assert_eq!(s.worker_id, worker_id);
            }
            other => panic!("unexpected signal: {other:?}"),
        }
    }

    #[test]
    fn handle_event_removed_emits_tombstone() {
        let svc = service();
        let adapter = WorkerHealthAdapter::new(svc.clone());
        let (registry, worker_id) =
            make_registry_with_worker("http://w1:8000", WorkerStatus::Ready);
        let worker = registry
            .get(&WorkerId::from_string(worker_id.clone()))
            .expect("registered");

        // Seed a live envelope so the tombstone has something to delete.
        adapter
            .publish_for(&worker_id, WorkerStatus::Ready)
            .expect("publish ok");
        assert_eq!(live_envelopes(&svc).len(), 1);

        let event = WorkerEvent::Removed {
            worker_id: WorkerId::from_string(worker_id.clone()),
            worker,
        };
        adapter.handle_event(&event).expect("handle ok");

        // Removal purges the outbox locally — peers age out via the freshness
        // window once the producer stops re-emitting — and drops the entry
        // from materialized state immediately.
        assert!(live_envelopes(&svc).is_empty());
        let key = SignalKey::WorkerHealth {
            region_id: "us-ashburn-1".to_string(),
            worker_id: worker_id.clone(),
            server_name: "smg-router-a".to_string(),
        };
        assert!(svc
            .outbox_snapshot()
            .iter()
            .find(|env| env.key == key)
            .is_none());
    }

    #[test]
    fn handle_event_status_changed_publishes_new_status() {
        let svc = service();
        let adapter = WorkerHealthAdapter::new(svc.clone());
        let (registry, worker_id) =
            make_registry_with_worker("http://w1:8000", WorkerStatus::Pending);
        let worker = registry
            .get(&WorkerId::from_string(worker_id.clone()))
            .expect("registered");

        // Mutate the worker's status to simulate the registry's transition.
        worker.set_status(WorkerStatus::Ready);
        let event = WorkerEvent::StatusChanged {
            worker_id: WorkerId::from_string(worker_id.clone()),
            worker,
            old_status: WorkerStatus::Pending,
            new_status: WorkerStatus::Ready,
        };
        adapter.handle_event(&event).unwrap();

        let env = single_live(&svc);
        match env.signal {
            Some(SignalKind::WorkerHealth(s)) => assert_eq!(s.status, WorkerStatus::Ready),
            other => panic!("unexpected signal: {other:?}"),
        }
    }

    #[test]
    fn reconcile_publishes_for_every_registered_worker() {
        let svc = service();
        let adapter = WorkerHealthAdapter::new(svc.clone());
        let registry = Arc::new(WorkerRegistry::new());
        for i in 0..3 {
            let worker = Arc::new(
                BasicWorkerBuilder::new(format!("http://w{i}:8000"))
                    .model(ModelCard::new("cohere.command-r-plus"))
                    .status(WorkerStatus::Ready)
                    .build(),
            );
            registry.register(worker).unwrap();
        }

        adapter.reconcile(&registry);

        let envelopes = live_envelopes(&svc);
        assert_eq!(envelopes.len(), 3);
        assert!(envelopes.iter().all(|e| !e.removed));
    }
}
