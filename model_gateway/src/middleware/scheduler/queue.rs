//! Per-class waiter queue used by [`super::scheduler`] to hold admission
//! candidates while their class is at capacity.

use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use tokio_util::sync::CancellationToken;

use super::Class;

/// A request waiting for admission in a class queue.
///
/// Carries only what the queue and dispatcher need: which class to admit
/// under, when the wait started (for starvation tracking), and the
/// cancellation token the client owns (so we can GC walkaway clients
/// without admitting them). Additional fields (tenant key, request id,
/// permit channel) are attached at higher layers as they're needed.
#[derive(Debug)]
pub struct Waiter {
    pub class: Class,
    pub queued_at: Instant,
    pub cancel: CancellationToken,
}

impl Waiter {
    pub fn new(class: Class, cancel: CancellationToken) -> Self {
        Self {
            class,
            queued_at: Instant::now(),
            cancel,
        }
    }
}

/// Per-class waiter queue.
///
/// Implemented as a trait so the v1 FIFO impl can be swapped for a
/// per-tenant fair-queue implementation without touching the scheduler.
/// All methods are sync; the queue is a contention point on slot release
/// and admission, so it uses `parking_lot::Mutex` rather than an async
/// mutex.
pub trait ClassQueue: Send + Sync {
    /// Append a waiter. Returns `Err(waiter)` when the queue is at its
    /// configured limit so the caller can convert the rejection into a
    /// 429 response.
    fn try_enqueue(&self, waiter: Waiter) -> Result<(), Waiter>;

    /// Pop the next waiter, or `None` when empty.
    fn pop_eligible(&self) -> Option<Waiter>;

    /// Wall-clock age of the head waiter (used by the dispatcher's
    /// starvation override). `None` when empty.
    fn head_age(&self) -> Option<Duration>;

    /// Current depth, including waiters whose cancel token has fired.
    fn depth(&self) -> usize;

    /// Remove the head waiter if its cancel token has fired (client
    /// walked away while queued). Single-step; the caller may invoke
    /// repeatedly to GC a run of cancelled heads.
    fn drop_cancelled_head(&self);
}

/// First-in, first-out per-class queue with a fixed maximum depth.
pub struct FifoClassQueue {
    waiters: Mutex<VecDeque<Waiter>>,
    max: usize,
}

impl FifoClassQueue {
    pub fn new(max: usize) -> Self {
        Self {
            waiters: Mutex::new(VecDeque::with_capacity(max.min(64))),
            max,
        }
    }
}

impl ClassQueue for FifoClassQueue {
    fn try_enqueue(&self, waiter: Waiter) -> Result<(), Waiter> {
        let mut guard = self.waiters.lock();
        if guard.len() >= self.max {
            return Err(waiter);
        }
        guard.push_back(waiter);
        Ok(())
    }

    fn pop_eligible(&self) -> Option<Waiter> {
        self.waiters.lock().pop_front()
    }

    fn head_age(&self) -> Option<Duration> {
        self.waiters.lock().front().map(|w| w.queued_at.elapsed())
    }

    fn depth(&self) -> usize {
        self.waiters.lock().len()
    }

    fn drop_cancelled_head(&self) {
        let mut guard = self.waiters.lock();
        if guard.front().is_some_and(|w| w.cancel.is_cancelled()) {
            guard.pop_front();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::middleware::scheduler::Class;

    fn waiter(class: Class) -> Waiter {
        Waiter::new(class, CancellationToken::new())
    }

    #[test]
    fn test_try_enqueue_fills_to_capacity_then_rejects() {
        let q = FifoClassQueue::new(4);
        for _ in 0..4 {
            assert!(q.try_enqueue(waiter(Class::Default)).is_ok());
        }
        let rejected = q
            .try_enqueue(waiter(Class::Default))
            .expect_err("queue full");
        assert_eq!(rejected.class, Class::Default);
    }

    #[test]
    fn test_pop_eligible_is_fifo() {
        // Distinguish waiters by class to verify insertion order is preserved.
        let q = FifoClassQueue::new(4);
        q.try_enqueue(waiter(Class::Bulk)).unwrap();
        q.try_enqueue(waiter(Class::Default)).unwrap();
        q.try_enqueue(waiter(Class::Interactive)).unwrap();
        assert_eq!(q.pop_eligible().unwrap().class, Class::Bulk);
        assert_eq!(q.pop_eligible().unwrap().class, Class::Default);
        assert_eq!(q.pop_eligible().unwrap().class, Class::Interactive);
        assert!(q.pop_eligible().is_none());
    }

    #[test]
    fn test_pop_eligible_returns_none_when_empty() {
        let q = FifoClassQueue::new(4);
        assert!(q.pop_eligible().is_none());
    }

    #[test]
    fn test_head_age_is_some_after_enqueue() {
        let q = FifoClassQueue::new(4);
        assert!(q.head_age().is_none());
        q.try_enqueue(waiter(Class::Default)).unwrap();
        std::thread::sleep(Duration::from_millis(5));
        let age = q.head_age().expect("head present");
        assert!(age >= Duration::from_millis(5));
    }

    #[test]
    fn test_depth_reflects_enqueue_and_pop() {
        let q = FifoClassQueue::new(4);
        assert_eq!(q.depth(), 0);
        q.try_enqueue(waiter(Class::Default)).unwrap();
        q.try_enqueue(waiter(Class::Default)).unwrap();
        assert_eq!(q.depth(), 2);
        q.pop_eligible();
        assert_eq!(q.depth(), 1);
    }

    #[test]
    fn test_drop_cancelled_head_removes_cancelled_head_only() {
        let q = FifoClassQueue::new(4);
        let cancelled = CancellationToken::new();
        cancelled.cancel();
        q.try_enqueue(Waiter::new(Class::Default, cancelled))
            .unwrap();
        q.try_enqueue(waiter(Class::Interactive)).unwrap();
        assert_eq!(q.depth(), 2);

        q.drop_cancelled_head();
        assert_eq!(q.depth(), 1, "cancelled head dropped");
        assert_eq!(
            q.pop_eligible().unwrap().class,
            Class::Interactive,
            "live waiter exposed as new head"
        );
    }

    #[test]
    fn test_drop_cancelled_head_leaves_live_head_alone() {
        let q = FifoClassQueue::new(4);
        q.try_enqueue(waiter(Class::Default)).unwrap();
        q.drop_cancelled_head();
        assert_eq!(q.depth(), 1, "non-cancelled head retained");
    }

    #[test]
    fn test_drop_cancelled_head_noop_on_empty_queue() {
        let q = FifoClassQueue::new(4);
        q.drop_cancelled_head();
        assert_eq!(q.depth(), 0);
    }
}
