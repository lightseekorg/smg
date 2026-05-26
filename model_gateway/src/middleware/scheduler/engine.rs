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
use tokio::sync::{oneshot, Notify};
use tokio_util::sync::CancellationToken;

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
    pub(super) release_notify: Notify,
    class_config: [ClassRuntimeConfig; 4],
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
            release_notify: Notify::new(),
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

        // Slow path: enqueue and wait.
        let (tx, rx) = oneshot::channel::<SchedulerPermit>();
        let waiter = Waiter::new(class, cancel.clone(), request_id, tx);
        if self.class_queues[class as usize]
            .try_enqueue(waiter)
            .is_err()
        {
            return AdmitOutcome::Rejected(RejectionReason::QueueFull);
        }

        let timeout = self.class_config[class as usize].queue_timeout;
        tokio::select! {
            result = rx => match result {
                Ok(permit) => AdmitOutcome::Admitted(permit),
                // The dispatcher dropped our sender without admitting us.
                // Treat as a cancellation rather than a timeout — the
                // dispatcher only drops if it knows we no longer need a slot.
                Err(_) => AdmitOutcome::Rejected(RejectionReason::ClientCancelled),
            },
            () = tokio::time::sleep(timeout) => AdmitOutcome::Rejected(RejectionReason::QueueTimeout),
            () = cancel.cancelled() => AdmitOutcome::Rejected(RejectionReason::ClientCancelled),
        }
    }

    /// Remove a handle from the registry, release its slot, and notify
    /// the dispatcher. Called from [`SchedulerPermit`]'s `Drop`.
    fn release_inflight(&self, handle: &InflightHandle) {
        self.inflight_registry.write().remove(handle.request_id());
        self.slot_pool.release(handle.class());
        self.release_notify.notify_one();
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
