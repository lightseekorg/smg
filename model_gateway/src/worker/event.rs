//! Events emitted by [`WorkerRegistry`] on state mutations.

use std::sync::Arc;

use openai_protocol::worker::WorkerStatus;

use super::{registry::WorkerId, worker::Worker};

/// Events broadcast when worker state changes.
///
/// Subscribers (WorkerManager, WorkerMonitor, WorkerSyncAdapter) use these for
/// incremental updates. Events carry `Arc<dyn Worker>` so subscribers can access
/// any worker data they need without re-querying the registry.
///
/// For `Removed`, the worker Arc is a pre-removal snapshot — the worker is
/// already gone from the registry when the event fires.
#[derive(Debug, Clone)]
pub enum WorkerEvent {
    /// A worker was added to the registry.
    Registered {
        worker_id: WorkerId,
        worker: Arc<dyn Worker>,
    },

    /// A worker was removed from the registry.
    /// The worker Arc is a pre-removal snapshot.
    Removed {
        worker_id: WorkerId,
        worker: Arc<dyn Worker>,
    },

    /// A worker was replaced (same URL, new worker object — e.g. property update).
    Replaced {
        worker_id: WorkerId,
        old: Arc<dyn Worker>,
        new: Arc<dyn Worker>,
    },

    /// A worker's lifecycle status changed (Pending→Ready, Ready→NotReady, etc.)
    StatusChanged {
        worker_id: WorkerId,
        worker: Arc<dyn Worker>,
        old_status: WorkerStatus,
        new_status: WorkerStatus,
    },
}
