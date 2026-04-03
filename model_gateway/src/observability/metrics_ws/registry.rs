//! Per-topic watch channel registry.
//!
//! Collectors write to senders; WS handlers clone receivers.
//! Channels are initialized with `None` — WS handlers should skip sending
//! initial snapshots until collectors have published real data.

use serde_json::Value;
use tokio::sync::watch;

use super::types::Topic;

/// Holds watch channel senders for each topic.
/// Channels hold `Option<Value>`: `None` means no data has been published yet.
pub struct WatchRegistry {
    workers: watch::Sender<Option<Value>>,
    loads: watch::Sender<Option<Value>>,
    cluster: watch::Sender<Option<Value>>,
    mesh: watch::Sender<Option<Value>>,
    rate_limits: watch::Sender<Option<Value>>,
    models: watch::Sender<Option<Value>>,
    metrics: watch::Sender<Option<Value>>,
}

impl Default for WatchRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl WatchRegistry {
    pub fn new() -> Self {
        Self {
            workers: watch::Sender::new(None),
            loads: watch::Sender::new(None),
            cluster: watch::Sender::new(None),
            mesh: watch::Sender::new(None),
            rate_limits: watch::Sender::new(None),
            models: watch::Sender::new(None),
            metrics: watch::Sender::new(None),
        }
    }

    /// Get the sender for a topic (used internally by collectors in the same crate).
    pub(crate) fn sender(&self, topic: Topic) -> &watch::Sender<Option<Value>> {
        match topic {
            Topic::Workers => &self.workers,
            Topic::Loads => &self.loads,
            Topic::Cluster => &self.cluster,
            Topic::Mesh => &self.mesh,
            Topic::RateLimits => &self.rate_limits,
            Topic::Models => &self.models,
            Topic::Metrics => &self.metrics,
        }
    }

    /// Publish latest state for a topic.
    /// Uses `send_replace` so state is retained even with zero active receivers.
    pub fn publish(&self, topic: Topic, value: Value) {
        self.sender(topic).send_replace(Some(value));
    }

    /// Get a receiver for a topic (used by WS handlers).
    pub fn subscribe(&self, topic: Topic) -> watch::Receiver<Option<Value>> {
        self.sender(topic).subscribe()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_value_is_none() {
        let registry = WatchRegistry::new();
        let rx = registry.subscribe(Topic::Workers);
        assert!(rx.borrow().is_none());
    }

    #[test]
    fn subscribe_receives_updates() {
        let registry = WatchRegistry::new();
        let mut rx = registry.subscribe(Topic::Workers);

        let data = serde_json::json!({"workers": [{"id": "w1"}], "total": 1});
        registry.publish(Topic::Workers, data.clone());

        assert!(rx.has_changed().unwrap());
        let value = rx.borrow_and_update();
        assert_eq!(value.as_ref().unwrap(), &data);
    }

    #[test]
    fn separate_topics_are_independent() {
        let registry = WatchRegistry::new();
        let rx_workers = registry.subscribe(Topic::Workers);
        let rx_loads = registry.subscribe(Topic::Loads);

        registry.publish(Topic::Workers, serde_json::json!({"updated": true}));

        assert!(rx_workers.has_changed().unwrap());
        assert!(!rx_loads.has_changed().unwrap());
    }

    #[test]
    fn multiple_subscribers_receive_same_data() {
        let registry = WatchRegistry::new();
        let mut rx1 = registry.subscribe(Topic::Metrics);
        let mut rx2 = registry.subscribe(Topic::Metrics);

        let data = serde_json::json!({"raw": "test_counter 42"});
        registry.publish(Topic::Metrics, data.clone());

        assert_eq!(rx1.borrow_and_update().as_ref().unwrap(), &data);
        assert_eq!(rx2.borrow_and_update().as_ref().unwrap(), &data);
    }

    #[test]
    fn late_subscriber_sees_latest_value() {
        let registry = WatchRegistry::new();
        let data = serde_json::json!({"workers": [{"id": "w1"}], "total": 1});
        registry.publish(Topic::Workers, data.clone());

        // Subscribe after publish — should immediately see the latest value
        let rx = registry.subscribe(Topic::Workers);
        assert_eq!(rx.borrow().as_ref().unwrap(), &data);
    }

    #[test]
    fn all_topics_have_senders() {
        let registry = WatchRegistry::new();
        for topic in Topic::ALL {
            let _ = registry.sender(*topic);
            let _ = registry.subscribe(*topic);
        }
    }
}
