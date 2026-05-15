//! Worker-load adapter — emits `worker-load/{region}/{worker_id}/{server_name}`.
//!
//! Periodic reconcile (no event-driven path — load changes continuously). On
//! each tick the adapter scans `WorkerRegistry::get_all_with_ids()` and
//! publishes one `WorkerLoadSignal` per worker. v1 uses the existing
//! `Worker::load()` atomic counter as the in-flight metric; the
//! `WorkerLoadInfo.load` field carries it for compatibility with `/get_loads`.

use std::sync::Arc;

use openai_protocol::worker::WorkerLoadInfo;

use crate::{
    cross_region::{
        CrossRegionResult, CrossRegionSyncService, SignalKey, SignalKind, WorkerLoadSignal,
    },
    worker::{registry::WorkerId, Worker, WorkerRegistry},
};

/// Default freshness window (design §4): worker-load uses 15 s with a 5 s
/// reconcile tick, so a quiescent worker stays fresh for two refresh cycles.
pub const DEFAULT_WORKER_LOAD_STALE_AFTER_MS: u32 = 15_000;

/// Worker-load producer.
#[derive(Debug, Clone)]
pub struct WorkerLoadAdapter {
    sync: Arc<CrossRegionSyncService>,
    stale_after_ms: u32,
}

impl WorkerLoadAdapter {
    pub fn new(sync: Arc<CrossRegionSyncService>) -> Self {
        Self {
            sync,
            stale_after_ms: DEFAULT_WORKER_LOAD_STALE_AFTER_MS,
        }
    }

    pub fn with_stale_after_ms(mut self, stale_after_ms: u32) -> Self {
        self.stale_after_ms = stale_after_ms;
        self
    }

    /// Publish current load for one worker. Public for adapters that want to
    /// drive single-worker updates (e.g., threshold-triggered emits).
    pub fn publish_for(
        &self,
        worker_id: &WorkerId,
        worker: &Arc<dyn Worker>,
    ) -> CrossRegionResult<()> {
        let region_id = self.sync.region_id().to_string();
        let server_name = self.sync.server_name().to_string();
        let load_isize = isize::try_from(worker.load()).unwrap_or(isize::MAX);
        let load_info = WorkerLoadInfo {
            worker: worker.url().to_string(),
            worker_type: None,
            load: load_isize,
            details: None,
            region_id: Some(region_id.clone()),
            worker_id: Some(worker_id.as_str().to_string()),
            model_id: Some(worker.model_id().to_string()),
            status: Some(worker.status()),
            generated_at_ms: None,
            version: None,
            source: None,
            remote_workers: None,
        };
        let key = SignalKey::WorkerLoad {
            region_id: region_id.clone(),
            worker_id: worker_id.as_str().to_string(),
            server_name: server_name.clone(),
        };
        let body = WorkerLoadSignal {
            region_id,
            worker_id: worker_id.as_str().to_string(),
            server_name,
            load: load_info,
        };
        self.sync.publish_signal(
            key,
            SignalKind::WorkerLoad(Box::new(body)),
            self.stale_after_ms,
        )
    }

    /// Republish load for every registered worker. Invoked by the gateway's
    /// 5 s reconcile tick.
    pub fn reconcile(&self, registry: &WorkerRegistry) {
        for (worker_id, worker) in registry.get_all_with_ids() {
            if let Err(err) = self.publish_for(&worker_id, &worker) {
                tracing::warn!(
                    worker_id = worker_id.as_str(),
                    error = %err,
                    "worker-load reconcile publish failed"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::{model_card::ModelCard, worker::WorkerStatus};

    use super::*;
    use crate::{
        cross_region::adapters::test_support::{live_envelopes, service, single_live},
        worker::BasicWorkerBuilder,
    };

    fn registry_with_load(url: &str, load: usize) -> (Arc<WorkerRegistry>, WorkerId) {
        let registry = Arc::new(WorkerRegistry::new());
        let worker = Arc::new(
            BasicWorkerBuilder::new(url)
                .model(ModelCard::new("cohere.command-r-plus"))
                .status(WorkerStatus::Ready)
                .build(),
        );
        for _ in 0..load {
            worker.increment_load();
        }
        let id = registry.register(worker).expect("register");
        (registry, id)
    }

    #[test]
    fn publish_for_emits_load_signal_with_per_replica_key() {
        let svc = service();
        let adapter = WorkerLoadAdapter::new(svc.clone());
        let (registry, worker_id) = registry_with_load("http://w1:8000", 7);
        let worker = registry.get(&worker_id).expect("registered");

        adapter
            .publish_for(&worker_id, &worker)
            .expect("publish ok");

        let env = single_live(&svc);
        match &env.key {
            SignalKey::WorkerLoad {
                region_id,
                worker_id: key_wid,
                server_name,
            } => {
                assert_eq!(region_id, "us-ashburn-1");
                assert_eq!(key_wid, worker_id.as_str());
                assert_eq!(server_name, "smg-router-a");
            }
            _ => panic!("unexpected key: {:?}", env.key),
        }
        match env.signal {
            Some(SignalKind::WorkerLoad(s)) => {
                assert_eq!(s.load.load, 7);
                assert_eq!(s.load.worker_id.as_deref(), Some(worker_id.as_str()));
            }
            other => panic!("unexpected signal: {other:?}"),
        }
    }

    #[test]
    fn reconcile_publishes_one_envelope_per_worker() {
        let svc = service();
        let adapter = WorkerLoadAdapter::new(svc.clone());
        let registry = Arc::new(WorkerRegistry::new());
        for i in 0..3 {
            let worker = Arc::new(
                BasicWorkerBuilder::new(format!("http://w{i}:8000"))
                    .model(ModelCard::new("cohere.command-r-plus"))
                    .status(WorkerStatus::Ready)
                    .build(),
            );
            for _ in 0..i {
                worker.increment_load();
            }
            registry.register(worker).unwrap();
        }

        adapter.reconcile(&registry);

        let envelopes = live_envelopes(&svc);
        assert_eq!(envelopes.len(), 3);
        let mut loads: Vec<isize> = envelopes
            .iter()
            .filter_map(|e| match e.signal.as_ref()? {
                SignalKind::WorkerLoad(s) => Some(s.load.load),
                _ => None,
            })
            .collect();
        loads.sort_unstable();
        assert_eq!(loads, vec![0, 1, 2]);
    }
}
