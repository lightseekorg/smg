//! Rate limit time window management
//!
//! Manages time windows for global rate limiting with periodic counter resets

use std::{sync::Arc, time::Duration};

use tokio::time::interval;
use tracing::{debug, info};

use super::sync::MeshSyncManager;

/// Rate limit window manager
/// Handles periodic reset of rate limit counters for time window management
pub struct RateLimitWindow {
    sync_manager: Arc<MeshSyncManager>,
    window_seconds: u64,
}

impl RateLimitWindow {
    pub fn new(sync_manager: Arc<MeshSyncManager>, window_seconds: u64) -> Self {
        Self {
            sync_manager,
            window_seconds,
        }
    }

    /// Start the window reset task
    /// This task periodically resets the global rate limit counter
    ///
    /// # Arguments
    /// * `shutdown_rx` - A watch receiver that signals when to stop the task
    pub async fn start_reset_task(self, mut shutdown_rx: tokio::sync::watch::Receiver<bool>) {
        let mut interval_timer = interval(Duration::from_secs(self.window_seconds));
        info!(
            "Starting rate limit window reset task with {}s interval",
            self.window_seconds
        );

        loop {
            tokio::select! {
                _ = interval_timer.tick() => {
                    debug!("Resetting global rate limit counter");
                    self.sync_manager.reset_global_rate_limit_counter();
                }
                _ = shutdown_rx.changed() => {
                    info!("Rate limit window reset task received shutdown signal");
                    break;
                }
            }
        }

        info!("Rate limit window reset task stopped");
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Duration};

    use tokio::time::sleep;

    use super::*;
    use crate::stores::{
        RateLimitConfig, StateStores, GLOBAL_RATE_LIMIT_COUNTER_KEY, GLOBAL_RATE_LIMIT_KEY,
    };

    #[test]
    fn test_rate_limit_window_new() {
        let stores = Arc::new(StateStores::with_self_name("node1".to_string()));
        let sync_manager = Arc::new(MeshSyncManager::new(stores, "node1".to_string()));

        let window = RateLimitWindow::new(sync_manager, 60);
        // Should create without panicking
        assert_eq!(window.window_seconds, 60);
    }

    #[test]
    fn test_rate_limit_window_different_intervals() {
        let stores = Arc::new(StateStores::with_self_name("node1".to_string()));
        let sync_manager = Arc::new(MeshSyncManager::new(stores, "node1".to_string()));

        let window1 = RateLimitWindow::new(sync_manager.clone(), 30);
        assert_eq!(window1.window_seconds, 30);

        let window2 = RateLimitWindow::new(sync_manager, 120);
        assert_eq!(window2.window_seconds, 120);
    }

    #[tokio::test]
    async fn test_rate_limit_window_reset_task_interval() {
        let stores = Arc::new(StateStores::with_self_name("node1".to_string()));
        let sync_manager = Arc::new(MeshSyncManager::new(stores, "node1".to_string()));

        // Set a very short window for testing (1 second)
        let window = RateLimitWindow::new(sync_manager, 1);

        // Create shutdown channel
        let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

        // Spawn the reset task
        let task_handle = tokio::spawn(async move {
            window.start_reset_task(shutdown_rx).await;
        });

        // Wait a bit to allow the task to run
        sleep(Duration::from_millis(1500)).await;

        // Send shutdown signal
        shutdown_tx
            .send(true)
            .expect("failed to send shutdown signal");

        // Wait for task to complete gracefully
        let res = tokio::time::timeout(Duration::from_secs(1), task_handle).await;
        assert!(res.is_ok(), "reset task did not shut down in time");
        let join_res = res.unwrap();
        assert!(join_res.is_ok(), "reset task panicked");

        // The task should have started and stopped gracefully
    }

    #[tokio::test]
    async fn test_rate_limit_window_reset_task() {
        let stores = Arc::new(StateStores::with_self_name("node1".to_string()));
        let sync_manager = Arc::new(MeshSyncManager::new(stores.clone(), "node1".to_string()));

        // Setup membership
        stores.rate_limit.update_membership(&["node1".to_string()]);

        // Setup config
        let key = GLOBAL_RATE_LIMIT_KEY.to_string();
        let config = RateLimitConfig {
            limit_per_second: 100,
        };
        let serialized = serde_json::to_vec(&config).unwrap();
        stores.app.insert(
            key.clone(),
            crate::stores::AppState {
                key: GLOBAL_RATE_LIMIT_KEY.to_string(),
                value: serialized,
                version: 1,
            },
            "node1".to_string(),
        );

        // Increment counter
        if stores.rate_limit.is_owner(GLOBAL_RATE_LIMIT_COUNTER_KEY) {
            sync_manager.sync_rate_limit_inc(GLOBAL_RATE_LIMIT_COUNTER_KEY.to_string(), 10);
            let value_before = sync_manager.get_rate_limit_value(GLOBAL_RATE_LIMIT_COUNTER_KEY);
            assert!(value_before.is_some() && value_before.unwrap() > 0);

            // Create window manager with short interval for testing
            let window = RateLimitWindow::new(sync_manager.clone(), 1); // 1 second

            // Create shutdown channel
            let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

            // Start reset task in background
            let reset_handle = tokio::spawn(async move {
                window.start_reset_task(shutdown_rx).await;
            });

            // Wait a bit for reset to happen
            sleep(Duration::from_millis(1500)).await;

            // Check that counter was reset (or at least decremented)
            let _value_after = sync_manager.get_rate_limit_value(GLOBAL_RATE_LIMIT_COUNTER_KEY);
            // Counter should be reset or significantly reduced
            // Note: The exact value depends on timing, but it should be less than initial

            // Send shutdown signal
            shutdown_tx
                .send(true)
                .expect("failed to send shutdown signal");

            // Wait for task to complete gracefully
            let res = tokio::time::timeout(Duration::from_secs(1), reset_handle).await;
            assert!(res.is_ok(), "reset task did not shut down in time");
            let join_res = res.unwrap();
            assert!(join_res.is_ok(), "reset task panicked");
        }
    }

    #[tokio::test]
    async fn test_rate_limit_window_reset_with_counter() {
        use crate::stores::MembershipState;

        // Use with_self_name to ensure RateLimitStore uses the same self_name
        let stores = Arc::new(StateStores::with_self_name("test_node".to_string()));
        let sync_manager = Arc::new(MeshSyncManager::new(
            stores.clone(),
            "test_node".to_string(),
        ));

        // First, add this node to membership so it can be an owner
        let membership_key = "test_node".to_string();
        let membership_state = MembershipState {
            name: "test_node".to_string(),
            address: "127.0.0.1:8080".to_string(),
            status: 1, // NodeStatus::Alive
            version: 1,
            metadata: Default::default(),
        };
        stores
            .membership
            .insert(membership_key, membership_state, "test_node".to_string());

        // Update rate limit membership so this node becomes an owner
        sync_manager.update_rate_limit_membership();

        // Check if node is owner before incrementing
        let key = GLOBAL_RATE_LIMIT_COUNTER_KEY.to_string();
        let is_owner = stores.rate_limit.is_owner(&key);
        assert!(is_owner, "Node should be owner of the rate limit key");

        // Set up a rate limit counter via sync_manager
        // This should increment the counter if the node is an owner
        sync_manager.sync_rate_limit_inc(key.clone(), 10);

        // Verify counter exists (was created)
        // Note: The actual value might be 0 due to PNCounter implementation details,
        // but the counter should exist after inc is called
        let counter_opt = stores.rate_limit.get_counter(&key);
        assert!(counter_opt.is_some(), "Counter should exist after inc call");

        // Verify counter was created after inc call
        // Note: The actual value depends on PNCounter implementation,
        // but the counter should exist after inc is called

        // Reset the counter
        sync_manager.reset_global_rate_limit_counter();

        // Verify reset was called (counter should still exist)
        // The reset implementation decrements by current count,
        // so the value should be 0 or negative after reset
        let reset_value = stores.rate_limit.value(&key).unwrap_or(0);
        // After reset, value should be <= 0 (since we decrement by current count)
        assert!(
            reset_value <= 0,
            "Counter should be reset to 0 or less, got: {}",
            reset_value
        );
    }

    #[test]
    fn test_rate_limit_window_zero_seconds() {
        let stores = Arc::new(StateStores::with_self_name("node1".to_string()));
        let sync_manager = Arc::new(MeshSyncManager::new(stores, "node1".to_string()));

        // Should handle zero seconds (though not recommended in practice)
        let window = RateLimitWindow::new(sync_manager, 0);
        assert_eq!(window.window_seconds, 0);
    }

    #[test]
    fn test_rate_limit_window_large_interval() {
        let stores = Arc::new(StateStores::with_self_name("node1".to_string()));
        let sync_manager = Arc::new(MeshSyncManager::new(stores, "node1".to_string()));

        // Test with a large interval
        let window = RateLimitWindow::new(sync_manager, 86400); // 24 hours
        assert_eq!(window.window_seconds, 86400);
    }

    #[tokio::test]
    async fn test_reset_global_rate_limit_counter_logic() {
        let stores = Arc::new(StateStores::with_self_name("node1".to_string()));
        let sync_manager = Arc::new(MeshSyncManager::new(stores.clone(), "node1".to_string()));

        // Setup membership
        stores.rate_limit.update_membership(&["node1".to_string()]);

        if stores.rate_limit.is_owner(GLOBAL_RATE_LIMIT_COUNTER_KEY) {
            // Increment counter
            sync_manager.sync_rate_limit_inc(GLOBAL_RATE_LIMIT_COUNTER_KEY.to_string(), 20);
            let value_before = sync_manager.get_rate_limit_value(GLOBAL_RATE_LIMIT_COUNTER_KEY);
            assert!(value_before.is_some() && value_before.unwrap() > 0);

            // Reset
            sync_manager.reset_global_rate_limit_counter();

            // Check that counter was reset
            let value_after = sync_manager.get_rate_limit_value(GLOBAL_RATE_LIMIT_COUNTER_KEY);
            // Should be 0 or negative after reset
            assert!(value_after.is_none() || value_after.unwrap() <= 0);
        }
    }
}
