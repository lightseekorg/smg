use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use tokio::sync::broadcast;
use tracing::warn;

use crate::types::WorkerSnapshot;

pub struct EventBus {
    tx: broadcast::Sender<Arc<WorkerSnapshot>>,
    /// Count of messages dropped because a slow subscriber's buffer was full.
    lagged_count: Arc<AtomicU64>,
}

impl EventBus {
    pub fn new(capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(capacity);
        Self {
            tx,
            lagged_count: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn publish(&self, snapshot: Arc<WorkerSnapshot>) {
        // Ignore SendError which happens if there are no active receivers
        let _ = self.tx.send(snapshot);
    }

    pub fn subscribe(&self) -> broadcast::Receiver<Arc<WorkerSnapshot>> {
        self.tx.subscribe()
    }

    /// Wrap a receiver so lag events are counted and logged automatically.
    ///
    /// Use this instead of the raw `Receiver` whenever you want lag visibility.
    /// The returned closure advances the receiver, skips over missed messages,
    /// and increments the internal `lagged_count` counter.
    pub fn subscribe_with_lag_tracking(
        &self,
    ) -> (broadcast::Receiver<Arc<WorkerSnapshot>>, Arc<AtomicU64>) {
        (self.tx.subscribe(), Arc::clone(&self.lagged_count))
    }

    /// Total number of messages that have been dropped due to slow subscribers.
    pub fn lagged_count(&self) -> u64 {
        self.lagged_count.load(Ordering::Relaxed)
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new(1024)
    }
}

/// Helper: receive from a broadcast channel, automatically handling `Lagged`
/// by skipping the missed messages and bumping the counter.
///
/// Returns `None` when the channel is closed (sender dropped).
pub async fn recv_or_skip(
    rx: &mut broadcast::Receiver<Arc<WorkerSnapshot>>,
    lagged_count: &AtomicU64,
) -> Option<Arc<WorkerSnapshot>> {
    loop {
        match rx.recv().await {
            Ok(msg) => return Some(msg),
            Err(broadcast::error::RecvError::Lagged(n)) => {
                lagged_count.fetch_add(n, Ordering::Relaxed);
                warn!(
                    missed = n,
                    total_lagged = lagged_count.load(Ordering::Relaxed),
                    "EventBus subscriber lagged; {} snapshot(s) dropped",
                    n
                );
                // continue — the receiver has been advanced past the missed messages
            }
            Err(broadcast::error::RecvError::Closed) => return None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::MetricSource;

    fn make_snap(url: &str) -> Arc<WorkerSnapshot> {
        Arc::new(WorkerSnapshot::new(
            url.to_string(),
            MetricSource::Piggyback,
        ))
    }

    #[tokio::test]
    async fn test_publish_and_receive() {
        let bus = EventBus::new(16);
        let mut rx = bus.subscribe();
        bus.publish(make_snap("http://w1"));
        let got = rx.recv().await.unwrap();
        assert_eq!(got.url, "http://w1");
    }

    #[tokio::test]
    async fn test_no_receiver_publish_is_silent() {
        // Should not panic when there are no subscribers
        let bus = EventBus::new(4);
        bus.publish(make_snap("http://w1"));
    }

    #[tokio::test]
    async fn test_lagged_count_increments() {
        let bus = EventBus::new(2); // very small buffer
        let (mut rx, lag_counter) = bus.subscribe_with_lag_tracking();

        // Overflow the buffer — capacity=2, so publishing 5 will lag the receiver by 3
        for i in 0..5u8 {
            bus.publish(make_snap(&format!("http://w{i}")));
        }

        // recv_or_skip should detect the lag and update the counter
        let _ = recv_or_skip(&mut rx, &lag_counter).await;
        assert!(
            lag_counter.load(Ordering::Relaxed) > 0,
            "lagged_count should be incremented after buffer overflow"
        );
    }

    #[test]
    fn test_default_capacity() {
        let bus = EventBus::default();
        assert_eq!(bus.lagged_count(), 0);
    }
}
