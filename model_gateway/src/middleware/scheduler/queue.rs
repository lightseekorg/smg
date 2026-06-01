//! Per-class waiter queue used by [`super::engine`] to hold admission
//! candidates while their class is at capacity.

use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use smg_auth::RequestId;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

use super::{engine::SchedulerPermit, Class};

/// A request waiting for admission in a class queue.
///
/// Carries the class lane to admit under, the wait-start instant (used
/// by the dispatcher's starvation override), the request id (carried
/// so admit can build the inflight handle after dequeue), the
/// cancellation token the client owns (lets the queue GC walkaway
/// clients without admitting them), and the oneshot the dispatcher
/// uses to hand back the [`SchedulerPermit`] when this waiter is
/// admitted. Rejections (queue full, queue timeout, client cancelled)
/// are returned synchronously from `admit` and never go through the
/// channel, so this carries only the success case.
#[derive(Debug)]
pub struct Waiter {
    pub class: Class,
    pub queued_at: Instant,
    pub cancel: CancellationToken,
    pub request_id: RequestId,
    pub permit_tx: oneshot::Sender<SchedulerPermit>,
}

impl Waiter {
    pub fn new(
        class: Class,
        cancel: CancellationToken,
        request_id: RequestId,
        permit_tx: oneshot::Sender<SchedulerPermit>,
    ) -> Self {
        Self {
            class,
            queued_at: Instant::now(),
            cancel,
            request_id,
            permit_tx,
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

    /// Configured maximum depth (the queue-size limit), for metrics.
    fn capacity(&self) -> usize;

    /// Drain any leading run of waiters whose cancel token has fired
    /// (clients that walked away while queued). One call removes every
    /// consecutive cancelled head in a single lock acquisition; the
    /// first non-cancelled or empty position stops the scan.
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

    fn capacity(&self) -> usize {
        self.max
    }

    fn drop_cancelled_head(&self) {
        let mut guard = self.waiters.lock();
        while guard.front().is_some_and(|w| w.cancel.is_cancelled()) {
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
        let (tx, _rx) = oneshot::channel();
        Waiter::new(class, CancellationToken::new(), RequestId("t".into()), tx)
    }

    fn waiter_with_cancel(class: Class, cancel: CancellationToken) -> Waiter {
        let (tx, _rx) = oneshot::channel();
        Waiter::new(class, cancel, RequestId("t".into()), tx)
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
        q.try_enqueue(waiter_with_cancel(Class::Default, cancelled))
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

    #[test]
    fn test_drop_cancelled_head_drains_run_in_single_call() {
        // When several clients in a row have walked away, a single call
        // should clear the whole run rather than requiring the caller to
        // loop. This keeps the dispatcher's GC pass to one lock cycle
        // instead of N.
        let q = FifoClassQueue::new(8);
        for _ in 0..3 {
            let cancelled = CancellationToken::new();
            cancelled.cancel();
            q.try_enqueue(waiter_with_cancel(Class::Default, cancelled))
                .unwrap();
        }
        q.try_enqueue(waiter(Class::Interactive)).unwrap();
        assert_eq!(q.depth(), 4);

        q.drop_cancelled_head();
        assert_eq!(
            q.depth(),
            1,
            "all three cancelled heads dropped in one call"
        );
        assert_eq!(
            q.pop_eligible().unwrap().class,
            Class::Interactive,
            "first live waiter is now the head"
        );
    }
}
