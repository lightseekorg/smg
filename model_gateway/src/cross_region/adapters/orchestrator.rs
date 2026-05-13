//! Cross-region producer orchestrator — owns the sync service and the four
//! local-producer adapters, plus the tokio tasks that drive event-driven and
//! periodic publication.
//!
//! Constructed once at gateway startup when `cross_region.enabled` is true;
//! call [`CrossRegionProducers::start`] to spawn the long-running tasks.
//! Drop the returned [`ProducerHandles`] (or call [`ProducerHandles::abort`])
//! to stop publication.

use std::{sync::Arc, time::Duration};

use tokio::task::JoinHandle;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;

use super::{ClientLatencyAdapter, RegionReadinessAdapter, WorkerHealthAdapter, WorkerLoadAdapter};
use crate::{
    cross_region::{CrossRegionResult, CrossRegionSyncService},
    worker::WorkerRegistry,
};

/// Reconcile cadences. Each value matches the defaults documented in design §4.
#[derive(Debug, Clone, Copy)]
pub struct ProducerCadences {
    pub readiness_reconcile_interval: Duration,
    pub worker_health_reconcile_interval: Duration,
    pub worker_load_refresh_interval: Duration,
    pub client_latency_publish_interval: Duration,
}

impl Default for ProducerCadences {
    fn default() -> Self {
        Self {
            readiness_reconcile_interval: Duration::from_secs(5),
            worker_health_reconcile_interval: Duration::from_secs(20),
            worker_load_refresh_interval: Duration::from_secs(5),
            client_latency_publish_interval: Duration::from_secs(10),
        }
    }
}

/// Bundle of the four producer adapters plus the sync service that backs
/// them. Cheap to clone (everything is `Arc`-wrapped); the gateway holds one
/// instance and adapters hand out their own clones to call sites.
#[derive(Debug, Clone)]
pub struct CrossRegionProducers {
    pub sync: Arc<CrossRegionSyncService>,
    pub region_readiness: RegionReadinessAdapter,
    pub worker_health: WorkerHealthAdapter,
    pub worker_load: WorkerLoadAdapter,
    pub client_latency: ClientLatencyAdapter,
}

impl CrossRegionProducers {
    /// Build adapters wrapping a fresh sync service. The sync service is
    /// constructed inside this call so the gateway only ever passes
    /// `region_id` / `server_name` in.
    pub fn new(region_id: String, server_name: String) -> CrossRegionResult<Self> {
        let sync = Arc::new(CrossRegionSyncService::new(region_id, server_name)?);
        Ok(Self::from_sync(sync))
    }

    /// Variant that takes a pre-built sync service. Useful for tests that
    /// want custom retention windows.
    pub fn from_sync(sync: Arc<CrossRegionSyncService>) -> Self {
        Self {
            region_readiness: RegionReadinessAdapter::new(sync.clone()),
            worker_health: WorkerHealthAdapter::new(sync.clone()),
            worker_load: WorkerLoadAdapter::new(sync.clone()),
            client_latency: ClientLatencyAdapter::new(sync.clone()),
            sync,
        }
    }

    /// Spawn the tokio tasks that drive event-driven (worker-health) and
    /// periodic (readiness, load, latency) publication. Returns handles the
    /// caller can drop to stop publication.
    ///
    /// Readiness is a placeholder `true` until the real readiness gate is
    /// plumbed; the gateway should call [`RegionReadinessAdapter::publish_ready`]
    /// directly from its readiness path once that exists. The periodic task
    /// keeps the signal fresh in the meantime.
    pub fn start(
        &self,
        worker_registry: Arc<WorkerRegistry>,
        cadences: ProducerCadences,
    ) -> ProducerHandles {
        let readiness_handle = spawn_readiness_loop(
            self.region_readiness.clone(),
            cadences.readiness_reconcile_interval,
        );
        let health_event_handle =
            spawn_worker_health_event_loop(self.worker_health.clone(), worker_registry.clone());
        let health_reconcile_handle = spawn_worker_health_reconcile_loop(
            self.worker_health.clone(),
            worker_registry.clone(),
            cadences.worker_health_reconcile_interval,
        );
        let load_handle = spawn_worker_load_loop(
            self.worker_load.clone(),
            worker_registry,
            cadences.worker_load_refresh_interval,
        );
        let latency_handle = spawn_client_latency_loop(
            self.client_latency.clone(),
            cadences.client_latency_publish_interval,
        );

        ProducerHandles {
            tasks: vec![
                readiness_handle,
                health_event_handle,
                health_reconcile_handle,
                load_handle,
                latency_handle,
            ],
        }
    }
}

/// Handles for the spawned producer tasks. Aborts on drop unless explicitly
/// detached via [`Self::detach`].
#[derive(Debug)]
pub struct ProducerHandles {
    tasks: Vec<JoinHandle<()>>,
}

impl ProducerHandles {
    /// Abort every producer task. Idempotent.
    pub fn abort(&self) {
        for handle in &self.tasks {
            handle.abort();
        }
    }

    /// Take ownership of the underlying join handles, e.g. to attach them to
    /// the gateway's graceful-shutdown machinery.
    pub fn detach(mut self) -> Vec<JoinHandle<()>> {
        std::mem::take(&mut self.tasks)
    }
}

impl Drop for ProducerHandles {
    fn drop(&mut self) {
        for handle in &self.tasks {
            handle.abort();
        }
    }
}

#[expect(
    clippy::disallowed_methods,
    reason = "task is bounded by ProducerHandles which aborts on drop"
)]
fn spawn_readiness_loop(adapter: RegionReadinessAdapter, interval: Duration) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            ticker.tick().await;
            if let Err(err) = adapter.publish_ready(true) {
                tracing::warn!(error = %err, "readiness reconcile publish failed");
            }
        }
    })
}

#[expect(
    clippy::disallowed_methods,
    reason = "task is bounded by ProducerHandles which aborts on drop"
)]
fn spawn_worker_health_event_loop(
    adapter: WorkerHealthAdapter,
    registry: Arc<WorkerRegistry>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut events = BroadcastStream::new(registry.subscribe_events());
        while let Some(item) = events.next().await {
            match item {
                Ok(event) => {
                    if let Err(err) = adapter.handle_event(&event) {
                        tracing::warn!(error = %err, "worker-health event publish failed");
                    }
                }
                Err(err) => {
                    // Broadcast lagged — log and continue. The reconcile loop
                    // will re-emit the current state.
                    tracing::warn!(error = %err, "worker-health event stream lagged");
                }
            }
        }
    })
}

#[expect(
    clippy::disallowed_methods,
    reason = "task is bounded by ProducerHandles which aborts on drop"
)]
fn spawn_worker_health_reconcile_loop(
    adapter: WorkerHealthAdapter,
    registry: Arc<WorkerRegistry>,
    interval: Duration,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            ticker.tick().await;
            adapter.reconcile(&registry);
        }
    })
}

#[expect(
    clippy::disallowed_methods,
    reason = "task is bounded by ProducerHandles which aborts on drop"
)]
fn spawn_worker_load_loop(
    adapter: WorkerLoadAdapter,
    registry: Arc<WorkerRegistry>,
    interval: Duration,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            ticker.tick().await;
            adapter.reconcile(&registry);
        }
    })
}

#[expect(
    clippy::disallowed_methods,
    reason = "task is bounded by ProducerHandles which aborts on drop"
)]
fn spawn_client_latency_loop(adapter: ClientLatencyAdapter, interval: Duration) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            ticker.tick().await;
            if let Err(err) = adapter.drain_and_publish() {
                tracing::warn!(error = %err, "client-latency drain publish failed");
            }
        }
    })
}
