//! Priority-aware admission scheduler engine.
//!
//! Owns the [`SlotPool`], per-class [`ClassQueue`]s, and the inflight
//! registry. Construction validates that the configured reservations
//! fit under the live backend capacity; runtime admission lives in
//! follow-on commits.

use std::{collections::HashMap, sync::Arc};

use parking_lot::RwLock;
use smg_auth::RequestId;
use thiserror::Error;
use tokio::sync::{oneshot, watch, Notify};
use tokio_util::sync::CancellationToken;
use tracing::warn;

use super::{
    inflight::InflightHandle,
    queue::{ClassQueue, FifoClassQueue, Waiter},
    slots::SlotPool,
    Class, ClassRuntimeConfig, SchedulerSettings,
};

/// Construction-time failures for [`PriorityScheduler::new`].
///
/// Capacity-vs-reserved is the only invariant the scheduler can check —
/// per-field validation already happened in
/// [`SchedulerSettings::from_cli_and_yaml`].
#[derive(Debug, Error, PartialEq)]
pub enum SchedulerInitError {
    #[error("sum of class reservations ({reserved}) exceeds capacity ({capacity})")]
    ReservationsExceedCapacity { reserved: u32, capacity: u16 },
}

/// Outcome of an [`PriorityScheduler::admit`] call.
pub enum AdmitOutcome {
    Admitted(SchedulerPermit),
    Rejected(RejectionReason),
}

/// Why an admission was rejected. Maps directly to the HTTP response
/// status in the admission middleware.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectionReason {
    /// Per-class queue is at its configured limit. → 429.
    QueueFull,
    /// Queued waiter aged past `queue_timeout`. → 408.
    QueueTimeout,
    /// Scheduler cancelled this inflight to admit a higher-priority
    /// waiter. → 503 + Retry-After. Lands in a later commit.
    Preempted,
    /// The caller's cancellation token fired before admission completed
    /// (typically the HTTP client disconnected). Never serialized — the
    /// client is already gone.
    ClientCancelled,
}

/// Priority-aware admission scheduler.
///
/// Construction wires up the slot pool, per-class queues, and an empty
/// inflight registry. Admission / dispatch / capacity-watch land in
/// follow-on commits; this commit exposes the read-only constructor
/// plus an internal `acquire_inflight` so [`SchedulerPermit`] can be
/// exercised against a real slot pool from tests.
pub struct PriorityScheduler {
    slot_pool: SlotPool,
    class_queues: [Arc<dyn ClassQueue>; 4],
    inflight_registry: RwLock<HashMap<RequestId, Arc<InflightHandle>>>,
    /// Arc so the dispatcher task can await on it without holding a strong
    /// reference to the scheduler. On scheduler `Drop` we fire `notify_one`
    /// so the dispatcher wakes, observes the failed `Weak::upgrade`, and
    /// exits cleanly.
    pub(super) release_notify: Arc<Notify>,
    class_config: [ClassRuntimeConfig; 4],
}

impl Drop for PriorityScheduler {
    fn drop(&mut self) {
        // Kick the dispatcher one last time so it can observe the Weak
        // upgrade failure and exit instead of awaiting forever.
        self.release_notify.notify_one();
    }
}

impl PriorityScheduler {
    /// Build a scheduler against the given settings and live backend
    /// capacity. Refuses if the configured reservations sum to more than
    /// the available capacity (the scheduler would otherwise lock itself
    /// out of its own non-reserved classes).
    pub fn new(
        settings: &SchedulerSettings,
        capacity: u16,
    ) -> Result<Arc<Self>, SchedulerInitError> {
        let total_reserved: u32 = Class::ALL
            .iter()
            .map(|c| u32::from(settings.class_config(*c).reserved))
            .sum();
        if total_reserved > u32::from(capacity) {
            return Err(SchedulerInitError::ReservationsExceedCapacity {
                reserved: total_reserved,
                capacity,
            });
        }

        let reserved_arr = Class::ALL.map(|c| settings.class_config(c).reserved);
        let class_queues: [Arc<dyn ClassQueue>; 4] =
            Class::ALL.map(|c| queue_for(settings, c, capacity));
        let class_config: [ClassRuntimeConfig; 4] =
            Class::ALL.map(|c| ClassRuntimeConfig::from_class_config(settings.class_config(c)));

        Ok(Arc::new(Self {
            slot_pool: SlotPool::new(capacity, reserved_arr),
            class_queues,
            inflight_registry: RwLock::new(HashMap::new()),
            release_notify: Arc::new(Notify::new()),
            class_config,
        }))
    }

    /// Try to acquire a slot under `class` for the given request id.
    /// Returns `Some(permit)` on success and `None` if the slot pool
    /// refuses (e.g. capacity exhausted or reservation guard blocks the
    /// class). The admission middleware's fast path calls this directly;
    /// the queue / preempt paths layer on top in follow-on commits.
    pub fn acquire_inflight(
        self: &Arc<Self>,
        class: Class,
        request_id: RequestId,
    ) -> Option<SchedulerPermit> {
        if !self.slot_pool.try_acquire(class) {
            return None;
        }
        Some(self.register_inflight(class, request_id))
    }

    /// Register a handle in the registry and wrap it in a permit.
    /// Caller already acquired a slot via the pool.
    fn register_inflight(self: &Arc<Self>, class: Class, request_id: RequestId) -> SchedulerPermit {
        let handle = Arc::new(InflightHandle::new(class, request_id));
        self.inflight_registry
            .write()
            .insert(handle.request_id().clone(), Arc::clone(&handle));
        SchedulerPermit {
            scheduler: Arc::clone(self),
            handle,
        }
    }

    /// Admit a request under `class`. Tries the fast path first; if the
    /// slot pool refuses, enqueues a waiter and awaits the dispatcher
    /// (or a queue timeout, or a client-side cancel).
    ///
    /// The `cancel` token is monitored for the duration of the wait —
    /// fires it to short-circuit a queued admission when the client
    /// disconnects.
    pub async fn admit(
        self: &Arc<Self>,
        class: Class,
        request_id: RequestId,
        cancel: CancellationToken,
    ) -> AdmitOutcome {
        // Fast path: slot available, admit synchronously.
        if let Some(permit) = self.acquire_inflight(class, request_id.clone()) {
            return AdmitOutcome::Admitted(permit);
        }

        // Slow path: enqueue and wait. The waiter holds a child cancel
        // token of the caller's `cancel` so the queue's
        // `drop_cancelled_head` GC sees the cancellation regardless of
        // which select arm fires below — client cancel propagates via
        // the parent, queue timeout fires the child explicitly.
        let (tx, rx) = oneshot::channel::<SchedulerPermit>();
        let waiter_cancel = cancel.child_token();
        let waiter = Waiter::new(class, waiter_cancel.clone(), request_id, tx);
        if self.class_queues[class as usize]
            .try_enqueue(waiter)
            .is_err()
        {
            return AdmitOutcome::Rejected(RejectionReason::QueueFull);
        }

        let timeout = self.class_config[class as usize].queue_timeout;
        let outcome = tokio::select! {
            result = rx => match result {
                Ok(permit) => AdmitOutcome::Admitted(permit),
                // The dispatcher dropped our sender without admitting us.
                // Treat as a cancellation rather than a timeout — the
                // dispatcher only drops if it knows we no longer need a slot.
                Err(_) => AdmitOutcome::Rejected(RejectionReason::ClientCancelled),
            },
            () = tokio::time::sleep(timeout) => AdmitOutcome::Rejected(RejectionReason::QueueTimeout),
            () = cancel.cancelled() => AdmitOutcome::Rejected(RejectionReason::ClientCancelled),
        };

        // Mark the waiter cancelled on any exit path so the queue's
        // `drop_cancelled_head` reaps it. Harmless if the waiter was
        // already popped (Admitted path) — the token has no other readers.
        waiter_cancel.cancel();
        outcome
    }

    /// Remove a handle from the registry, release its slot, and notify
    /// the dispatcher. Called from [`SchedulerPermit`]'s `Drop`.
    fn release_inflight(&self, handle: &InflightHandle) {
        self.inflight_registry.write().remove(handle.request_id());
        self.slot_pool.release(handle.class());
        self.release_notify.notify_one();
    }

    /// Try to admit one queued waiter. Returns `true` if a waiter was
    /// successfully admitted (caller should call again to drain), `false`
    /// if nothing was admittable this pass.
    ///
    /// Honors two policies in order:
    /// 1. **Starvation override** — scans Bulk → Default → Interactive for
    ///    a head waiter that has aged past its class's
    ///    `starvation_threshold`. The first such waiter is admitted via
    ///    `try_acquire_ignoring_reservations`, bypassing the reservation
    ///    guard so a starved low-class waiter can take a slot that a
    ///    higher class has reserved-but-not-used.
    /// 2. **Normal priority** — System → Interactive → Default → Bulk.
    ///    Each class's queue is drained one waiter at a time so the
    ///    caller's outer loop can interleave drains across classes.
    pub fn wake_next_waiter(self: &Arc<Self>) -> bool {
        // Starvation override — lowest priority first (the ones most at
        // risk of starving).
        for class in [Class::Bulk, Class::Default, Class::Interactive] {
            let idx = class as usize;
            self.class_queues[idx].drop_cancelled_head();
            let Some(head_age) = self.class_queues[idx].head_age() else {
                continue;
            };
            if head_age <= self.class_config[idx].starvation_threshold {
                continue;
            }
            if self.slot_pool.try_acquire_ignoring_reservations(class) {
                if self.send_to_head(class) {
                    return true;
                }
                // Acquired a slot but the head was gone (racy cancel).
                // Release and fall through to normal priority.
                self.slot_pool.release(class);
            }
        }

        // Normal priority — highest class first.
        for class in [
            Class::System,
            Class::Interactive,
            Class::Default,
            Class::Bulk,
        ] {
            let idx = class as usize;
            self.class_queues[idx].drop_cancelled_head();
            if self.class_queues[idx].depth() == 0 {
                continue;
            }
            if self.slot_pool.try_acquire(class) {
                if self.send_to_head(class) {
                    return true;
                }
                self.slot_pool.release(class);
            }
        }

        false
    }

    /// Pop the head waiter for `class` and deliver a permit through its
    /// oneshot. Returns `false` if the queue was empty (race with a
    /// cancellation that drained the head between the depth check and
    /// the pop).
    ///
    /// The caller must have already acquired a slot under `class`.
    fn send_to_head(self: &Arc<Self>, class: Class) -> bool {
        let Some(Waiter {
            request_id,
            permit_tx,
            ..
        }) = self.class_queues[class as usize].pop_eligible()
        else {
            return false;
        };
        let permit = self.register_inflight(class, request_id);
        // If the receiver was dropped, the permit goes out of scope here
        // and its Drop releases the slot back. The dispatcher's outer
        // loop will try another waiter on the next iteration.
        let _ = permit_tx.send(permit);
        true
    }

    /// Spawn the dispatcher background task. `select!`s over two
    /// signals:
    ///
    /// - `release_notify` — slot was released (or a waiter enqueued
    ///   via Drop-fire on scheduler shutdown). Drain queued waiters
    ///   until none can be admitted.
    /// - `capacity_watch.changed()` — backend capacity changed. Apply
    ///   the new value (scaling reservations down if they no longer
    ///   fit) and kick the drain.
    ///
    /// Holds only a `Weak<Self>` so the task does not keep the
    /// scheduler alive past its last external strong reference. Drop
    /// on the scheduler fires `release_notify`, which lets the
    /// dispatcher observe the failed upgrade and exit.
    pub fn spawn_dispatcher(self: &Arc<Self>, capacity_watch: watch::Receiver<u16>) {
        let weak = Arc::downgrade(self);
        let notify = Arc::clone(&self.release_notify);
        #[expect(
            clippy::disallowed_methods,
            reason = "dispatcher loop holds only a Weak<Self> and exits when the scheduler is dropped (Drop fires release_notify)"
        )]
        tokio::spawn(async move {
            // `Option<watch::Receiver>` so we can drop the receiver once the
            // upstream sender is closed. A bare `Receiver` whose sender is
            // gone makes `changed()` resolve `Err` immediately on every
            // poll — the `select!` would otherwise pick that arm forever
            // and burn CPU.
            let mut capacity_watch = Some(capacity_watch);
            loop {
                let new_capacity = match capacity_watch.as_mut() {
                    Some(rx) => tokio::select! {
                        () = notify.notified() => None,
                        result = rx.changed() => match result {
                            Ok(()) => Some(*rx.borrow()),
                            Err(_) => {
                                // Upstream sender dropped. Stop watching;
                                // the dispatcher keeps serving release
                                // events via the other arm.
                                capacity_watch = None;
                                continue;
                            }
                        },
                    },
                    None => {
                        notify.notified().await;
                        None
                    }
                };
                let Some(scheduler) = weak.upgrade() else {
                    break;
                };
                match new_capacity {
                    Some(new_cap) => scheduler.apply_new_capacity(new_cap),
                    None => while scheduler.wake_next_waiter() {},
                }
            }
        });
    }

    /// Apply a new backend capacity from the WorkerCapacity watch.
    ///
    /// On shrink past `Σ reserved`, scales reservations down
    /// proportionally so the new capacity remains usable. On grow,
    /// fires `release_notify` so the dispatcher re-evaluates queued
    /// waiters under the larger ceiling.
    fn apply_new_capacity(&self, new_capacity: u16) {
        let old = self.slot_pool.set_capacity(new_capacity);
        if new_capacity == old {
            return;
        }

        let total_reserved: u32 = Class::ALL
            .iter()
            .map(|c| u32::from(self.slot_pool.reserved(*c)))
            .sum();
        if total_reserved > u32::from(new_capacity) && total_reserved > 0 {
            let scale = f64::from(new_capacity) / f64::from(total_reserved as u16);
            for class in Class::ALL {
                let r = self.slot_pool.reserved(class);
                let scaled = (f64::from(r) * scale).floor() as u16;
                self.slot_pool.set_reserved(class, scaled);
            }
            warn!(
                old_total_reserved = total_reserved,
                new_capacity,
                scale = scale,
                "scheduler: reserved slots scaled down after capacity shrink"
            );
        }

        if new_capacity > old {
            // Capacity grew — wake the dispatcher to drain.
            self.release_notify.notify_one();
        }
    }
}

/// Compute the effective per-class queue limit for a given backend
/// capacity: `max(queue_size, ceil(queue_size_per_slot * capacity))`.
fn queue_for(settings: &SchedulerSettings, class: Class, capacity: u16) -> Arc<dyn ClassQueue> {
    let cfg = settings.class_config(class);
    let multiplier = (cfg.queue_size_per_slot * f32::from(capacity)).ceil() as u32;
    let limit = cfg.queue_size.max(multiplier) as usize;
    Arc::new(FifoClassQueue::new(limit))
}

/// RAII handle on one admitted request. Holding a permit keeps the slot
/// reserved; dropping it returns the slot, removes the handle from the
/// inflight registry, and notifies the dispatcher.
pub struct SchedulerPermit {
    scheduler: Arc<PriorityScheduler>,
    handle: Arc<InflightHandle>,
}

impl SchedulerPermit {
    /// Borrow the underlying inflight handle (for TTFT marking and
    /// preemption coordination in follow-on commits).
    pub fn handle(&self) -> &Arc<InflightHandle> {
        &self.handle
    }
}

impl std::fmt::Debug for SchedulerPermit {
    // Hand-rolled to avoid recursing into Arc<PriorityScheduler>, which
    // doesn't itself derive Debug (it contains atomics and a Notify).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SchedulerPermit")
            .field("class", &self.handle.class())
            .field("request_id", self.handle.request_id())
            .finish()
    }
}

impl Drop for SchedulerPermit {
    fn drop(&mut self) {
        self.scheduler.release_inflight(&self.handle);
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Duration};

    use super::*;
    use crate::middleware::scheduler::{ClassConfig, PrioritySchedulerYaml};

    fn default_settings() -> SchedulerSettings {
        SchedulerSettings::from_cli_and_yaml(true, Class::Default, 32, None).unwrap()
    }

    fn rid(s: &str) -> RequestId {
        RequestId(s.to_string())
    }

    #[test]
    fn test_new_succeeds_when_reservations_fit() {
        // Built-in defaults: Σ reserved = 128 (Interactive) + 32 (System) = 160.
        let s = default_settings();
        assert!(PriorityScheduler::new(&s, 256).is_ok());
    }

    #[test]
    fn test_new_rejects_when_reservations_exceed_capacity() {
        let s = default_settings();
        let result = PriorityScheduler::new(&s, 100);
        assert!(matches!(
            result,
            Err(SchedulerInitError::ReservationsExceedCapacity {
                reserved: 160,
                capacity: 100
            })
        ));
    }

    #[test]
    fn test_acquire_returns_permit_when_slot_available() {
        let s = default_settings();
        let scheduler = PriorityScheduler::new(&s, 256).unwrap();
        let permit = scheduler.acquire_inflight(Class::Default, rid("req-1"));
        assert!(permit.is_some());
    }

    #[test]
    fn test_acquire_returns_none_when_reservation_guard_blocks() {
        // Capacity 200, reservations 160 (Interactive=128, System=32). Bulk
        // is the lowest class so it gets capacity - 160 = 40 slots before
        // the guard refuses further admissions.
        let s = default_settings();
        let scheduler = PriorityScheduler::new(&s, 200).unwrap();
        let mut held = Vec::new();
        for i in 0..40 {
            held.push(
                scheduler
                    .acquire_inflight(Class::Bulk, rid(&format!("req-{i}")))
                    .expect("under guard"),
            );
        }
        assert!(scheduler
            .acquire_inflight(Class::Bulk, rid("overflow"))
            .is_none());
    }

    #[tokio::test]
    async fn test_permit_drop_releases_slot_and_notifies_dispatcher() {
        let s = default_settings();
        let scheduler = PriorityScheduler::new(&s, 256).unwrap();
        let permit = scheduler
            .acquire_inflight(Class::Default, rid("req-1"))
            .expect("admitted");
        assert_eq!(scheduler.slot_pool.inflight(Class::Default), 1);

        let notified = scheduler.release_notify.notified();
        drop(permit);

        assert_eq!(scheduler.slot_pool.inflight(Class::Default), 0);
        tokio::time::timeout(Duration::from_millis(100), notified)
            .await
            .expect("release_notify fires on drop");
    }

    #[tokio::test]
    async fn test_registry_inserts_on_acquire_and_removes_on_drop() {
        let s = default_settings();
        let scheduler = PriorityScheduler::new(&s, 256).unwrap();
        let id = rid("req-1");
        let permit = scheduler
            .acquire_inflight(Class::Default, id.clone())
            .expect("admitted");
        assert!(scheduler.inflight_registry.read().contains_key(&id));
        drop(permit);
        assert!(!scheduler.inflight_registry.read().contains_key(&id));
    }

    // ── admit ────────────────────────────────────────────────────────

    /// Build settings with zero reservations on every class (so we can
    /// run admit tests against small capacities without tripping the
    /// reserved-vs-capacity guard) and an override on one class's
    /// queue_size + queue_timeout.
    fn settings_with(class: Class, queue_size: u32, queue_timeout_secs: u64) -> SchedulerSettings {
        use std::collections::HashMap as StdMap;

        let mut classes = StdMap::new();
        for c in Class::ALL {
            let mut cfg = ClassConfig::default_for(c);
            cfg.reserved = 0;
            if c == class {
                cfg.queue_size = queue_size;
                cfg.queue_size_per_slot = 0.0;
                cfg.queue_timeout_secs = queue_timeout_secs;
            }
            classes.insert(c, cfg);
        }
        let yaml = PrioritySchedulerYaml {
            classes,
            tenant_policies: StdMap::new(),
        };
        SchedulerSettings::from_cli_and_yaml(true, Class::Default, 32, Some(&yaml)).unwrap()
    }

    #[tokio::test]
    async fn test_admit_fast_path_when_slot_available() {
        let s = default_settings();
        let scheduler = PriorityScheduler::new(&s, 256).unwrap();
        let outcome = scheduler
            .admit(Class::Default, rid("req-1"), CancellationToken::new())
            .await;
        assert!(matches!(outcome, AdmitOutcome::Admitted(_)));
    }

    #[tokio::test]
    async fn test_admit_rejects_when_queue_full() {
        // Capacity 1 (slot held), Default queue_size=1 (one waiter pre-stuffed
        // directly into the queue): the next admit takes the slow path and
        // hits a full queue.
        let s = settings_with(Class::Default, 1, 60);
        let scheduler = PriorityScheduler::new(&s, 1).unwrap();
        let _held = scheduler
            .acquire_inflight(Class::Default, rid("held"))
            .expect("admitted directly");

        let (queued_tx, _queued_rx) = oneshot::channel();
        let queued = Waiter::new(
            Class::Default,
            CancellationToken::new(),
            rid("queued"),
            queued_tx,
        );
        scheduler.class_queues[Class::Default as usize]
            .try_enqueue(queued)
            .expect("queue had room for one");

        let outcome = scheduler
            .admit(Class::Default, rid("w2"), CancellationToken::new())
            .await;
        assert!(matches!(
            outcome,
            AdmitOutcome::Rejected(RejectionReason::QueueFull)
        ));
    }

    #[tokio::test(start_paused = true)]
    async fn test_admit_rejects_on_queue_timeout() {
        // Capacity 1 forces enqueue; queue_timeout=1s; never release → QueueTimeout.
        let s = settings_with(Class::Default, 16, 1);
        let scheduler = PriorityScheduler::new(&s, 1).unwrap();
        let _held = scheduler
            .acquire_inflight(Class::Default, rid("held"))
            .expect("admitted directly");

        let admit_future = scheduler.admit(Class::Default, rid("w1"), CancellationToken::new());
        let outcome = admit_future.await;
        assert!(matches!(
            outcome,
            AdmitOutcome::Rejected(RejectionReason::QueueTimeout)
        ));
    }

    #[tokio::test(start_paused = true)]
    async fn test_admit_timeout_marks_waiter_for_gc() {
        // After QueueTimeout fires, the Waiter still sits in the FIFO
        // until something reaps it. The cancel token on the Waiter must
        // be fired so a future drop_cancelled_head call evicts it,
        // preventing false QueueFull on later admissions.
        let s = settings_with(Class::Default, 16, 1);
        let scheduler = PriorityScheduler::new(&s, 1).unwrap();
        let _held = scheduler
            .acquire_inflight(Class::Default, rid("held"))
            .expect("admitted directly");

        let outcome = scheduler
            .admit(Class::Default, rid("w1"), CancellationToken::new())
            .await;
        assert!(matches!(
            outcome,
            AdmitOutcome::Rejected(RejectionReason::QueueTimeout)
        ));

        // The waiter is still in the queue (one entry), but cancelled.
        assert_eq!(scheduler.class_queues[Class::Default as usize].depth(), 1);
        scheduler.class_queues[Class::Default as usize].drop_cancelled_head();
        assert_eq!(
            scheduler.class_queues[Class::Default as usize].depth(),
            0,
            "drop_cancelled_head must reap the timed-out waiter"
        );
    }

    #[tokio::test]
    async fn test_admit_rejects_on_client_cancel() {
        // Pre-cancel the token before admit is called; the fast path fails
        // (slot held), the slow path enqueues, then the cancel arm of select!
        // fires immediately.
        let s = settings_with(Class::Default, 16, 60);
        let scheduler = PriorityScheduler::new(&s, 1).unwrap();
        let _held = scheduler
            .acquire_inflight(Class::Default, rid("held"))
            .expect("admitted directly");

        let cancel = CancellationToken::new();
        cancel.cancel();
        let outcome = scheduler.admit(Class::Default, rid("w1"), cancel).await;
        assert!(matches!(
            outcome,
            AdmitOutcome::Rejected(RejectionReason::ClientCancelled)
        ));
    }

    // ── wake_next_waiter / dispatcher ───────────────────────────────

    fn enqueue_waiter(
        scheduler: &Arc<PriorityScheduler>,
        class: Class,
    ) -> oneshot::Receiver<SchedulerPermit> {
        let (tx, rx) = oneshot::channel();
        let waiter = Waiter::new(class, CancellationToken::new(), rid("queued"), tx);
        scheduler.class_queues[class as usize]
            .try_enqueue(waiter)
            .expect("queue had room");
        rx
    }

    #[test]
    fn test_wake_next_waiter_returns_false_when_no_slot_available() {
        let s = settings_with(Class::Default, 8, 60);
        let scheduler = PriorityScheduler::new(&s, 1).unwrap();
        let _held = scheduler
            .acquire_inflight(Class::Default, rid("held"))
            .unwrap();
        let _rx = enqueue_waiter(&scheduler, Class::Default);
        assert!(!scheduler.wake_next_waiter(), "no slot to give");
    }

    #[tokio::test]
    async fn test_wake_next_waiter_admits_queued_waiter() {
        let s = settings_with(Class::Default, 8, 60);
        let scheduler = PriorityScheduler::new(&s, 1).unwrap();
        let held = scheduler
            .acquire_inflight(Class::Default, rid("held"))
            .unwrap();
        let mut rx = enqueue_waiter(&scheduler, Class::Default);
        drop(held);
        assert!(scheduler.wake_next_waiter());
        let permit = rx.try_recv().expect("permit delivered");
        assert_eq!(permit.handle().class(), Class::Default);
    }

    #[tokio::test]
    async fn test_wake_next_waiter_honors_priority_order() {
        // Interactive beats Bulk when both are queued and a slot frees.
        let s = settings_with(Class::Bulk, 8, 60); // Bulk queue=8 (interactive defaults too)
        let scheduler = PriorityScheduler::new(&s, 1).unwrap();
        let held = scheduler
            .acquire_inflight(Class::Default, rid("held"))
            .unwrap();

        let mut bulk_rx = enqueue_waiter(&scheduler, Class::Bulk);
        let mut interactive_rx = enqueue_waiter(&scheduler, Class::Interactive);

        drop(held);
        assert!(scheduler.wake_next_waiter());
        // Interactive should be served first; bulk still waiting.
        assert!(interactive_rx.try_recv().is_ok());
        assert!(bulk_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn test_starvation_override_promotes_stale_bulk_head() {
        // Bulk starvation_threshold=0s (anything older than 0 wins), with
        // capacity entirely reserved by Interactive — without the
        // override Bulk could never advance.
        use std::collections::HashMap as StdMap;
        let mut classes = StdMap::new();
        for c in Class::ALL {
            let mut cfg = ClassConfig::default_for(c);
            cfg.reserved = 0;
            // Set tiny starvation threshold for Bulk so an immediate
            // enqueue is "stale" by the time wake fires.
            if c == Class::Bulk {
                cfg.starvation_threshold_secs = 1;
                cfg.queue_size = 4;
                cfg.queue_size_per_slot = 0.0;
            }
            // Interactive holds the full capacity in reservation.
            if c == Class::Interactive {
                cfg.reserved = 1;
            }
            classes.insert(c, cfg);
        }
        let yaml = PrioritySchedulerYaml {
            classes,
            tenant_policies: StdMap::new(),
        };
        let settings =
            SchedulerSettings::from_cli_and_yaml(true, Class::Default, 32, Some(&yaml)).unwrap();

        let scheduler = PriorityScheduler::new(&settings, 1).unwrap();
        let mut bulk_rx = enqueue_waiter(&scheduler, Class::Bulk);

        // Sleep past the starvation threshold so head_age > threshold.
        tokio::time::sleep(Duration::from_millis(1100)).await;

        // No regular path admits Bulk under the Interactive reservation,
        // but the starvation override should.
        assert!(scheduler.wake_next_waiter(), "starvation override fires");
        assert!(bulk_rx.try_recv().is_ok());
    }

    fn dummy_capacity_watch(initial: u16) -> watch::Receiver<u16> {
        // Construct a watch receiver whose sender is intentionally kept
        // alive for the test's duration. Leaking is fine — these tests
        // run in isolation.
        let (tx, rx) = watch::channel(initial);
        std::mem::forget(tx);
        rx
    }

    #[tokio::test]
    async fn test_spawn_dispatcher_admits_queued_waiter_on_release() {
        let s = settings_with(Class::Default, 8, 60);
        let scheduler = PriorityScheduler::new(&s, 1).unwrap();
        scheduler.spawn_dispatcher(dummy_capacity_watch(1));

        let held = scheduler
            .acquire_inflight(Class::Default, rid("held"))
            .unwrap();

        // Kick off an admit that must go to the slow path.
        let scheduler_for_admit = Arc::clone(&scheduler);
        let admit_future = async move {
            scheduler_for_admit
                .admit(Class::Default, rid("queued"), CancellationToken::new())
                .await
        };

        // Race: release the slot, then await admit. The dispatcher
        // should observe the release and admit the queued waiter.
        let (admit_outcome, ()) = tokio::join!(admit_future, async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            drop(held);
        });

        assert!(matches!(admit_outcome, AdmitOutcome::Admitted(_)));
    }

    // ── apply_new_capacity / capacity watch ─────────────────────────

    #[test]
    fn test_apply_new_capacity_grow_fires_release_notify() {
        let s = default_settings();
        let scheduler = PriorityScheduler::new(&s, 256).unwrap();
        // Notify already has whatever permits prior tests fired; reset by
        // taking a fresh notified() before the apply.
        let notified = scheduler.release_notify.notified();
        scheduler.apply_new_capacity(512);
        assert_eq!(scheduler.slot_pool.capacity(), 512);
        // Grow path must have signaled the dispatcher.
        tokio::pin!(notified);
        let polled = futures::FutureExt::now_or_never(notified);
        assert!(polled.is_some(), "release_notify fires on capacity grow");
    }

    #[test]
    fn test_apply_new_capacity_shrink_below_reserved_scales_reservations() {
        // Built-in defaults: Interactive reserves 128, System reserves 32
        // = 160 total. Shrink to 80; reservations should scale by 0.5 so
        // the new sum fits.
        let s = default_settings();
        let scheduler = PriorityScheduler::new(&s, 256).unwrap();
        scheduler.apply_new_capacity(80);
        assert_eq!(scheduler.slot_pool.capacity(), 80);
        let new_total: u32 = Class::ALL
            .iter()
            .map(|c| u32::from(scheduler.slot_pool.reserved(*c)))
            .sum();
        assert!(
            new_total <= 80,
            "scaled reservations ({new_total}) must fit under new capacity"
        );
        // Each individual class is also scaled down proportionally
        // (floor rounding).
        assert_eq!(scheduler.slot_pool.reserved(Class::Interactive), 64);
        assert_eq!(scheduler.slot_pool.reserved(Class::System), 16);
    }

    #[test]
    fn test_apply_new_capacity_no_op_when_value_unchanged() {
        let s = default_settings();
        let scheduler = PriorityScheduler::new(&s, 256).unwrap();
        let before = scheduler.slot_pool.reserved(Class::Interactive);
        scheduler.apply_new_capacity(256);
        assert_eq!(scheduler.slot_pool.reserved(Class::Interactive), before);
    }

    #[tokio::test]
    async fn test_dispatcher_keeps_serving_after_capacity_watch_sender_drops() {
        // After the WorkerCapacity sender is dropped, the dispatcher must
        // continue handling release events from release_notify. The earlier
        // implementation hot-looped on the closed watch arm; this test
        // exercises the post-drop path end-to-end.
        let s = settings_with(Class::Default, 8, 60);
        let scheduler = PriorityScheduler::new(&s, 1).unwrap();
        let (capacity_tx, capacity_rx) = watch::channel(1u16);
        scheduler.spawn_dispatcher(capacity_rx);

        drop(capacity_tx); // close the watch — the failing branch trigger.
                           // Yield so the dispatcher observes the drop and disables that arm.
        tokio::time::sleep(Duration::from_millis(10)).await;

        let held = scheduler
            .acquire_inflight(Class::Default, rid("held"))
            .unwrap();
        let scheduler_for_admit = Arc::clone(&scheduler);
        let admit_future = async move {
            scheduler_for_admit
                .admit(Class::Default, rid("queued"), CancellationToken::new())
                .await
        };
        let (outcome, ()) = tokio::join!(admit_future, async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            drop(held);
        });
        assert!(matches!(outcome, AdmitOutcome::Admitted(_)));
    }

    #[tokio::test]
    async fn test_capacity_watch_grow_drains_queue() {
        // Capacity 1 (slot held), queue has one waiter. Grow capacity to
        // 2 via the watch — dispatcher should drain the queued waiter.
        let s = settings_with(Class::Default, 8, 60);
        let scheduler = PriorityScheduler::new(&s, 1).unwrap();
        let (tx, rx) = watch::channel(1u16);
        scheduler.spawn_dispatcher(rx);

        let _held = scheduler
            .acquire_inflight(Class::Default, rid("held"))
            .unwrap();

        let scheduler_for_admit = Arc::clone(&scheduler);
        let admit_future = async move {
            scheduler_for_admit
                .admit(Class::Default, rid("queued"), CancellationToken::new())
                .await
        };

        let (outcome, ()) = tokio::join!(admit_future, async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            tx.send(2).unwrap();
        });

        assert!(matches!(outcome, AdmitOutcome::Admitted(_)));
        assert_eq!(scheduler.slot_pool.capacity(), 2);
    }

    #[tokio::test]
    async fn test_permit_keeps_scheduler_alive_via_arc() {
        let s = default_settings();
        let scheduler = PriorityScheduler::new(&s, 256).unwrap();
        let weak = Arc::downgrade(&scheduler);
        let permit = scheduler
            .acquire_inflight(Class::Default, rid("req-1"))
            .expect("admitted");
        drop(scheduler);
        // Permit holds a strong ref, so weak still upgrades.
        assert!(weak.upgrade().is_some());
        drop(permit);
        // All strong refs gone now.
        assert!(weak.upgrade().is_none());
    }
}
