//! Flow control for mesh cluster communication
//!
//! Provides:
//! - Message size limit constant for gRPC encode/decode caps
//! - Exponential backoff for peer reconnection

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use parking_lot::RwLock;

/// Maximum message size in bytes (default: 10MB)
pub const MAX_MESSAGE_SIZE: usize = 10 * 1024 * 1024;

/// Exponential backoff calculator for reconnection
#[derive(Debug, Clone)]
pub struct ExponentialBackoff {
    initial_delay: Duration,
    max_delay: Duration,
    multiplier: f64,
}

impl ExponentialBackoff {
    pub fn new(initial_delay: Duration, max_delay: Duration, multiplier: f64) -> Self {
        Self {
            initial_delay,
            max_delay,
            multiplier,
        }
    }

    /// Calculate delay for attempt number (0-indexed)
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let max_delay_secs = self.max_delay.as_secs_f64();
        let delay_secs = self.initial_delay.as_secs_f64()
            * self.multiplier.powi(attempt.min(i32::MAX as u32) as i32);
        // Guard against f64 overflow to infinity (e.g., 2.0^1024)
        // which would panic in Duration::from_secs_f64.
        let capped = if delay_secs.is_finite() && delay_secs >= 0.0 {
            delay_secs.min(max_delay_secs)
        } else {
            max_delay_secs
        };
        Duration::from_secs_f64(capped)
    }
}

impl Default for ExponentialBackoff {
    fn default() -> Self {
        Self::new(Duration::from_secs(1), Duration::from_secs(60), 2.0)
    }
}

/// Connection retry manager with exponential backoff
#[derive(Debug)]
pub struct RetryManager {
    backoff: ExponentialBackoff,
    last_attempt: Arc<RwLock<Option<Instant>>>,
    attempt_count: Arc<RwLock<u32>>,
}

impl RetryManager {
    pub fn new(backoff: ExponentialBackoff) -> Self {
        Self {
            backoff,
            last_attempt: Arc::new(RwLock::new(None)),
            attempt_count: Arc::new(RwLock::new(0)),
        }
    }

    /// Check if we should retry now (based on backoff delay)
    pub fn should_retry(&self) -> bool {
        let last = self.last_attempt.read();
        if let Some(last_attempt) = *last {
            let attempt = *self.attempt_count.read();
            let delay = self.backoff.delay_for_attempt(attempt);
            last_attempt.elapsed() >= delay
        } else {
            true // First attempt
        }
    }

    /// Record a retry attempt
    pub fn record_attempt(&self) {
        *self.last_attempt.write() = Some(Instant::now());
        let mut count = self.attempt_count.write();
        *count = count.saturating_add(1);
    }

    /// Reset retry state (on successful connection)
    pub fn reset(&self) {
        *self.last_attempt.write() = None;
        *self.attempt_count.write() = 0;
    }

    /// Get current attempt count
    pub fn attempt_count(&self) -> u32 {
        *self.attempt_count.read()
    }

    /// Get next retry delay
    pub fn next_delay(&self) -> Duration {
        let attempt = *self.attempt_count.read();
        self.backoff.delay_for_attempt(attempt)
    }
}

impl Default for RetryManager {
    fn default() -> Self {
        Self::new(ExponentialBackoff::default())
    }
}
