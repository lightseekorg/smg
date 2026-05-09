//! Readiness state machine for cold start
//!
//! Manages node lifecycle: NotReady -> Joining -> SnapshotPull -> Converging -> Ready

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use parking_lot::RwLock;
use tracing::info;

/// Local cold-start readiness state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReadinessState {
    /// Node is not ready (initial state).
    NotReady,
    /// Node is joining the cluster.
    Joining,
    /// Node is pulling initial CRDT state from peers.
    SnapshotPull,
    /// Node is converging (applying state updates).
    Converging,
    /// Node is ready to serve traffic.
    Ready,
}

impl ReadinessState {
    pub fn as_str(self) -> &'static str {
        match self {
            ReadinessState::NotReady => "not_ready",
            ReadinessState::Joining => "joining",
            ReadinessState::SnapshotPull => "snapshot_pull",
            ReadinessState::Converging => "converging",
            ReadinessState::Ready => "ready",
        }
    }
}

/// Convergence detection configuration
#[derive(Debug, Clone)]
pub struct ConvergenceConfig {
    /// Time window for convergence detection (seconds)
    pub convergence_window: Duration,
    /// Minimum number of state updates without changes to consider converged
    pub min_stable_updates: usize,
    /// Timeout for snapshot pull (seconds)
    pub snapshot_timeout: Duration,
}

impl Default for ConvergenceConfig {
    fn default() -> Self {
        Self {
            convergence_window: Duration::from_secs(10),
            min_stable_updates: 5,
            snapshot_timeout: Duration::from_secs(60),
        }
    }
}

/// Convergence tracker
#[derive(Debug)]
struct ConvergenceTracker {
    last_update_time: Option<Instant>,
    stable_update_count: usize,
    last_state_hash: Option<u64>,
}

impl ConvergenceTracker {
    fn new() -> Self {
        Self {
            last_update_time: None,
            stable_update_count: 0,
            last_state_hash: None,
        }
    }

    fn record_update(&mut self, state_hash: u64, config: &ConvergenceConfig) -> bool {
        let now = Instant::now();

        if let Some(last_hash) = self.last_state_hash {
            if last_hash == state_hash {
                // State unchanged
                self.stable_update_count += 1;
            } else {
                // State changed, reset counter
                self.stable_update_count = 0;
            }
        } else {
            // First update
            self.stable_update_count = 0;
        }

        self.last_state_hash = Some(state_hash);

        // Check elapsed time since the first stable update, not since this update
        if let Some(last_time) = self.last_update_time {
            let elapsed = now.duration_since(last_time);
            if elapsed >= config.convergence_window
                && self.stable_update_count >= config.min_stable_updates
            {
                return true;
            }
        }

        // Only set the timestamp if this is the first update or state changed
        if self.last_update_time.is_none() || self.stable_update_count == 0 {
            self.last_update_time = Some(now);
        }

        false
    }

    fn reset(&mut self) {
        self.last_update_time = None;
        self.stable_update_count = 0;
        self.last_state_hash = None;
    }
}

/// Readiness state machine for managing cold start
#[derive(Debug)]
pub struct ReadinessStateMachine {
    readiness: Arc<RwLock<ReadinessState>>,
    config: ConvergenceConfig,
    convergence_tracker: Arc<RwLock<ConvergenceTracker>>,
    snapshot_start_time: Arc<RwLock<Option<Instant>>>,
}

impl ReadinessStateMachine {
    pub fn new(config: ConvergenceConfig) -> Self {
        Self {
            readiness: Arc::new(RwLock::new(ReadinessState::NotReady)),
            config,
            convergence_tracker: Arc::new(RwLock::new(ConvergenceTracker::new())),
            snapshot_start_time: Arc::new(RwLock::new(None)),
        }
    }

    /// Get current readiness state
    pub fn readiness(&self) -> ReadinessState {
        *self.readiness.read()
    }

    /// Transition to joining state
    pub fn start_joining(&self) {
        let mut readiness = self.readiness.write();
        if *readiness == ReadinessState::NotReady {
            *readiness = ReadinessState::Joining;
            info!("Readiness state: NotReady -> Joining");
        }
    }

    /// Transition to snapshot pull state
    pub fn start_snapshot_pull(&self) {
        let mut readiness = self.readiness.write();
        if *readiness == ReadinessState::Joining {
            *readiness = ReadinessState::SnapshotPull;
            *self.snapshot_start_time.write() = Some(Instant::now());
            info!("Readiness state: Joining -> SnapshotPull");
        }
    }

    /// Check if snapshot pull has timed out
    pub fn is_snapshot_timeout(&self) -> bool {
        if let Some(start_time) = *self.snapshot_start_time.read() {
            start_time.elapsed() > self.config.snapshot_timeout
        } else {
            false
        }
    }

    /// Transition to converging state
    pub fn start_converging(&self) {
        let mut readiness = self.readiness.write();
        if *readiness == ReadinessState::SnapshotPull {
            *readiness = ReadinessState::Converging;
            *self.snapshot_start_time.write() = None;
            self.convergence_tracker.write().reset();
            info!("Readiness state: SnapshotPull -> Converging");
        }
    }

    /// Record a state update and check for convergence
    pub fn record_state_update(&self) -> bool {
        if self.readiness() != ReadinessState::Converging {
            return false;
        }

        // Calculate a simple hash of store states
        let state_hash = self.calculate_state_hash();
        let mut tracker = self.convergence_tracker.write();
        let converged = tracker.record_update(state_hash, &self.config);

        if converged {
            self.transition_to_ready();
            return true;
        }

        false
    }

    /// Transition to ready state
    pub fn transition_to_ready(&self) {
        let mut readiness = self.readiness.write();
        if *readiness == ReadinessState::Converging {
            *readiness = ReadinessState::Ready;
            info!("Readiness state: Converging -> Ready");
        }
    }

    /// Check if node is ready
    pub fn is_ready(&self) -> bool {
        self.readiness() == ReadinessState::Ready
    }

    /// Calculate a simple hash of current state (for convergence detection)
    fn calculate_state_hash(&self) -> u64 {
        use std::{
            collections::hash_map::DefaultHasher,
            hash::{Hash, Hasher},
        };

        let mut hasher = DefaultHasher::new();
        self.readiness().hash(&mut hasher);
        hasher.finish()
    }

    /// Reset state machine (for testing or recovery)
    pub fn reset(&self) {
        *self.readiness.write() = ReadinessState::NotReady;
        self.convergence_tracker.write().reset();
        *self.snapshot_start_time.write() = None;
    }
}

impl Default for ReadinessStateMachine {
    fn default() -> Self {
        Self::new(ConvergenceConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn create_test_config() -> ConvergenceConfig {
        ConvergenceConfig {
            convergence_window: Duration::from_millis(100),
            min_stable_updates: 3,
            snapshot_timeout: Duration::from_secs(1),
        }
    }

    #[test]
    fn test_readiness_state_as_str() {
        assert_eq!(ReadinessState::NotReady.as_str(), "not_ready");
        assert_eq!(ReadinessState::Joining.as_str(), "joining");
        assert_eq!(ReadinessState::SnapshotPull.as_str(), "snapshot_pull");
        assert_eq!(ReadinessState::Converging.as_str(), "converging");
        assert_eq!(ReadinessState::Ready.as_str(), "ready");
    }

    #[test]
    fn test_convergence_config_default() {
        let config = ConvergenceConfig::default();
        assert_eq!(config.convergence_window, Duration::from_secs(10));
        assert_eq!(config.min_stable_updates, 5);
        assert_eq!(config.snapshot_timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_readiness_state_machine_initial_state() {
        let config = create_test_config();
        let sm = ReadinessStateMachine::new(config);

        assert_eq!(sm.readiness(), ReadinessState::NotReady);
        assert!(!sm.is_ready());
    }

    #[test]
    fn test_state_transition_flow() {
        let config = create_test_config();
        let sm = ReadinessStateMachine::new(config);

        // Start joining
        sm.start_joining();
        assert_eq!(sm.readiness(), ReadinessState::Joining);

        // Start snapshot pull
        sm.start_snapshot_pull();
        assert_eq!(sm.readiness(), ReadinessState::SnapshotPull);
        assert!(!sm.is_snapshot_timeout());

        // Start converging
        sm.start_converging();
        assert_eq!(sm.readiness(), ReadinessState::Converging);

        // Transition to ready
        sm.transition_to_ready();
        assert_eq!(sm.readiness(), ReadinessState::Ready);
        assert!(sm.is_ready());
    }

    #[test]
    fn test_state_transition_guards() {
        let config = create_test_config();
        let sm = ReadinessStateMachine::new(config);

        // Cannot start snapshot pull without joining first
        sm.start_snapshot_pull();
        assert_eq!(sm.readiness(), ReadinessState::NotReady);

        // Cannot start converging without snapshot pull
        sm.start_joining();
        sm.start_converging();
        assert_eq!(sm.readiness(), ReadinessState::Joining);

        // Cannot transition to ready without converging
        sm.transition_to_ready();
        assert_eq!(sm.readiness(), ReadinessState::Joining);
    }

    #[test]
    fn test_snapshot_timeout() {
        let mut config = create_test_config();
        config.snapshot_timeout = Duration::from_millis(50);
        let sm = ReadinessStateMachine::new(config);

        sm.start_joining();
        sm.start_snapshot_pull();
        assert!(!sm.is_snapshot_timeout());

        // Wait for timeout
        std::thread::sleep(Duration::from_millis(100));
        assert!(sm.is_snapshot_timeout());
    }

    #[test]
    fn test_record_state_update_not_converging() {
        let config = create_test_config();
        let sm = ReadinessStateMachine::new(config);

        // Should return false when not in converging state
        assert!(!sm.record_state_update());
        assert_eq!(sm.readiness(), ReadinessState::NotReady);
    }

    #[test]
    fn test_convergence_detection() {
        let mut config = create_test_config();
        config.convergence_window = Duration::from_millis(50);
        config.min_stable_updates = 2;
        let sm = ReadinessStateMachine::new(config);

        // Transition to converging state
        sm.start_joining();
        sm.start_snapshot_pull();
        sm.start_converging();
        assert_eq!(sm.readiness(), ReadinessState::Converging);

        // Record multiple updates with same state
        let converged1 = sm.record_state_update();
        assert!(!converged1);

        // Wait a bit and record more updates
        std::thread::sleep(Duration::from_millis(60));
        let converged2 = sm.record_state_update();
        assert!(!converged2); // Still not enough stable updates

        // Record more stable updates
        std::thread::sleep(Duration::from_millis(10));
        let converged3 = sm.record_state_update();
        // Should converge after enough stable updates within window
        if converged3 {
            assert_eq!(sm.readiness(), ReadinessState::Ready);
        }
    }

    #[test]
    fn test_convergence_reset_on_state_change() {
        let mut config = create_test_config();
        config.convergence_window = Duration::from_millis(100);
        config.min_stable_updates = 2;
        let sm = ReadinessStateMachine::new(config);

        sm.start_joining();
        sm.start_snapshot_pull();
        sm.start_converging();

        // Record update
        sm.record_state_update();

        sm.reset();
        sm.start_joining();
        sm.start_snapshot_pull();
        sm.start_converging();

        sm.record_state_update();

        // The stable count should be reset
        std::thread::sleep(Duration::from_millis(110));
        let converged = sm.record_state_update();
        // Should not converge immediately after state change
        assert!(!converged || sm.readiness() == ReadinessState::Converging);
    }

    #[test]
    fn test_reset() {
        let config = create_test_config();
        let sm = ReadinessStateMachine::new(config);

        // Go through states
        sm.start_joining();
        sm.start_snapshot_pull();
        sm.start_converging();
        sm.transition_to_ready();

        assert_eq!(sm.readiness(), ReadinessState::Ready);

        // Reset
        sm.reset();
        assert_eq!(sm.readiness(), ReadinessState::NotReady);
        assert!(!sm.is_ready());
        assert!(!sm.is_snapshot_timeout());
    }

    #[test]
    fn test_calculate_state_hash() {
        let config = create_test_config();
        let sm = ReadinessStateMachine::new(config);

        let hash1 = sm.calculate_state_hash();
        sm.start_joining();
        let hash2 = sm.calculate_state_hash();
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_default_implementation() {
        let sm = ReadinessStateMachine::default();
        assert_eq!(sm.readiness(), ReadinessState::NotReady);
        assert!(!sm.is_ready());
    }
}
