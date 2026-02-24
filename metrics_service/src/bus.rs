use std::sync::Arc;

use tokio::sync::broadcast;

use crate::types::WorkerSnapshot;

pub struct EventBus {
    tx: broadcast::Sender<Arc<WorkerSnapshot>>,
}

impl EventBus {
    pub fn new(capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(capacity);
        Self { tx }
    }

    pub fn publish(&self, snapshot: Arc<WorkerSnapshot>) {
        // Ignore SendError which happens if there are no active receivers
        let _ = self.tx.send(snapshot);
    }

    pub fn subscribe(&self) -> broadcast::Receiver<Arc<WorkerSnapshot>> {
        self.tx.subscribe()
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new(1024)
    }
}
