//! In-flight request bookkeeping for the priority scheduler.

use std::sync::atomic::{AtomicU64, Ordering};

/// Tracks a single admitted request from admission until its response body
/// finishes draining (or the scheduler preempts it).
///
/// The central race resolver is [`first_byte_at`]: it encodes three states
/// in one `AtomicU64`, using a sentinel for "preempted" so that the
/// admission side and the body-wrapping side can resolve their race with a
/// single compare-and-swap each.
///
/// State encoding for `first_byte_at`:
///
/// | Value         | Meaning                                                   |
/// |---------------|-----------------------------------------------------------|
/// | `0`           | No first-byte emitted yet and not preempted (initial)     |
/// | `u64::MAX`    | Scheduler preempted this request before TTFT (sentinel)   |
/// | anything else | Millis-since-admit of the first response byte (TTFT)      |
///
/// `try_mark_first_byte` clamps `now_ms` to `u64::MAX - 1` so a TTFT
/// measurement can never collide with the preempt sentinel.
pub struct InflightHandle {
    first_byte_at: AtomicU64,
}

impl InflightHandle {
    /// Sentinel stored in `first_byte_at` once the scheduler has selected
    /// this request for preemption.
    const PREEMPTED_SENTINEL: u64 = u64::MAX;

    pub fn new() -> Self {
        Self {
            first_byte_at: AtomicU64::new(0),
        }
    }

    /// Attempt to mark this request as preempted. Returns `true` on the
    /// first successful CAS; subsequent calls (or any call after TTFT was
    /// marked) return `false`.
    ///
    /// The caller fires the cancellation token only after a `true` return,
    /// so preemption never races with a response that already started
    /// streaming bytes to the client.
    pub fn try_mark_preempted(&self) -> bool {
        self.first_byte_at
            .compare_exchange(
                0,
                Self::PREEMPTED_SENTINEL,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    /// Attempt to mark the first response byte at `now_ms` millis-since-
    /// admit. Returns `true` on the first successful CAS; subsequent calls
    /// (or any call after preemption was marked) return `false`.
    ///
    /// `now_ms` is clamped to `u64::MAX - 1` so a TTFT measurement can
    /// never collide with the preempt sentinel.
    pub fn try_mark_first_byte(&self, now_ms: u64) -> bool {
        let value = now_ms.min(Self::PREEMPTED_SENTINEL - 1);
        self.first_byte_at
            .compare_exchange(0, value, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    /// Test-only raw read of the underlying atomic. Production callers
    /// should compare against the sentinel via the type's API rather than
    /// reading the raw value.
    #[cfg(test)]
    fn first_byte_at_raw(&self, ord: Ordering) -> u64 {
        self.first_byte_at.load(ord)
    }
}

impl Default for InflightHandle {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use super::*;

    #[test]
    fn test_try_mark_preempted_succeeds_then_fails() {
        let handle = InflightHandle::new();
        assert!(
            handle.try_mark_preempted(),
            "first preempt CAS should succeed"
        );
        assert!(
            !handle.try_mark_preempted(),
            "second preempt CAS should fail (slot already taken)"
        );
    }

    #[test]
    fn test_try_mark_preempted_stores_sentinel() {
        let handle = InflightHandle::new();
        handle.try_mark_preempted();
        assert_eq!(
            handle.first_byte_at_raw(Ordering::Acquire),
            u64::MAX,
            "preempt sentinel is u64::MAX"
        );
    }

    #[test]
    fn test_try_mark_first_byte_succeeds_then_fails() {
        let handle = InflightHandle::new();
        assert!(
            handle.try_mark_first_byte(42),
            "first TTFT CAS should succeed"
        );
        assert!(
            !handle.try_mark_first_byte(99),
            "second TTFT CAS should fail (slot already taken)"
        );
    }

    #[test]
    fn test_ttft_loses_race_against_preempt() {
        let handle = InflightHandle::new();
        assert!(handle.try_mark_preempted());
        assert!(
            !handle.try_mark_first_byte(5),
            "TTFT must lose if preempt already won"
        );
    }

    #[test]
    fn test_preempt_loses_race_against_ttft() {
        let handle = InflightHandle::new();
        assert!(handle.try_mark_first_byte(5));
        assert!(
            !handle.try_mark_preempted(),
            "preempt must lose if TTFT already won"
        );
    }

    #[test]
    fn test_try_mark_first_byte_clamps_to_avoid_sentinel_collision() {
        // u64::MAX is the preempt sentinel; a TTFT measurement that happens
        // to equal that value would falsely appear as "preempted" to any
        // reader. Clamp to u64::MAX - 1.
        let handle = InflightHandle::new();
        assert!(handle.try_mark_first_byte(u64::MAX));
        assert_eq!(handle.first_byte_at_raw(Ordering::Acquire), u64::MAX - 1);
    }

    #[test]
    fn test_initial_state_is_zero() {
        let handle = InflightHandle::new();
        assert_eq!(handle.first_byte_at_raw(Ordering::Acquire), 0);
    }
}
