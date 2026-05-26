//! Packed slot accounting for the priority scheduler.
//!
//! Per-class in-flight counts are packed into a single `AtomicU64` (four
//! `u16` lanes) so an admission check can both verify availability and
//! claim a slot in a single compare-and-swap. This keeps the reservation
//! guard race-free under concurrent admissions to different classes — a
//! property that a per-class counter array cannot give us cheaply.
//!
//! Lane layout (low to high u16): `[Bulk, Default, Interactive, System]`,
//! indexed by [`Class`] cast to `usize`.

use std::sync::atomic::{AtomicU16, AtomicU64, Ordering};

use super::Class;

/// Decode a packed `u64` into per-class in-flight counts, lane order
/// `[Bulk, Default, Interactive, System]`.
pub(super) fn unpack(packed: u64) -> [u16; 4] {
    [
        (packed & 0xFFFF) as u16,
        ((packed >> 16) & 0xFFFF) as u16,
        ((packed >> 32) & 0xFFFF) as u16,
        ((packed >> 48) & 0xFFFF) as u16,
    ]
}

/// Encode per-class in-flight counts into a single `u64`.
pub(super) fn pack(counts: [u16; 4]) -> u64 {
    u64::from(counts[0])
        | (u64::from(counts[1]) << 16)
        | (u64::from(counts[2]) << 32)
        | (u64::from(counts[3]) << 48)
}

/// How many slots a `class` admission may consume right now, given the
/// current per-class counts, reservations, and total capacity.
///
/// Higher-priority classes hold their reservations: a `Bulk` admission
/// cannot consume a slot that `System` has reserved but not yet used.
/// Once a higher class has actually consumed its reservation, the hold
/// collapses 1:1 and the slot returns to the shared pool.
///
/// A class's own reservation never shrinks its own headroom.
pub(super) fn slots_available_to(
    counts: [u16; 4],
    reserved: [u16; 4],
    capacity: u16,
    class: Class,
) -> u16 {
    let used_total: u16 = counts.iter().copied().fold(0, u16::saturating_add);
    let unavailable_to_us: u16 = Class::ALL
        .iter()
        .filter(|&&c| c > class)
        .map(|&c| reserved[c as usize].saturating_sub(counts[c as usize]))
        .fold(0, u16::saturating_add);
    capacity
        .saturating_sub(used_total)
        .saturating_sub(unavailable_to_us)
}

/// Slot pool: capacity, per-class reservations, and packed per-class
/// in-flight counts. The admission middleware acquires a slot here at
/// admission and releases it from the response-body wrapper.
pub struct SlotPool {
    capacity: AtomicU16,
    inflight_packed: AtomicU64,
    reserved: [AtomicU16; 4],
}

impl SlotPool {
    /// Build a pool with the given capacity and per-class reservations
    /// (lane order `[Bulk, Default, Interactive, System]`).
    pub fn new(capacity: u16, reserved: [u16; 4]) -> Self {
        Self {
            capacity: AtomicU16::new(capacity),
            inflight_packed: AtomicU64::new(0),
            reserved: reserved.map(AtomicU16::new),
        }
    }

    /// Current in-flight count for a class. Single atomic load + a few
    /// shifts; safe to call from any thread.
    pub fn inflight(&self, class: Class) -> u16 {
        unpack(self.inflight_packed.load(Ordering::Acquire))[class as usize]
    }

    fn snapshot_reserved(&self) -> [u16; 4] {
        [
            self.reserved[0].load(Ordering::Relaxed),
            self.reserved[1].load(Ordering::Relaxed),
            self.reserved[2].load(Ordering::Relaxed),
            self.reserved[3].load(Ordering::Relaxed),
        ]
    }

    /// Reservation-aware acquire. Returns `true` after a successful CAS;
    /// `false` if `class` has no headroom under the reservation guard.
    /// Bounded loop — re-reads on each failed CAS and exits as soon as
    /// the slot is genuinely full.
    pub fn try_acquire(&self, class: Class) -> bool {
        let lane = class as usize;
        let capacity = self.capacity.load(Ordering::Acquire);
        let reserved = self.snapshot_reserved();
        let mut cur = self.inflight_packed.load(Ordering::Acquire);
        loop {
            let mut counts = unpack(cur);
            if slots_available_to(counts, reserved, capacity, class) == 0 {
                return false;
            }
            counts[lane] = counts[lane].saturating_add(1);
            let new = pack(counts);
            match self.inflight_packed.compare_exchange_weak(
                cur,
                new,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(observed) => cur = observed,
            }
        }
    }

    /// Acquire ignoring the reservation guard — only the total-capacity
    /// ceiling applies. Used by the dispatcher's starvation override so
    /// a starved low-class waiter can consume a reserved-but-unused slot
    /// rather than wait indefinitely.
    pub fn try_acquire_ignoring_reservations(&self, class: Class) -> bool {
        let lane = class as usize;
        let capacity = self.capacity.load(Ordering::Acquire);
        let mut cur = self.inflight_packed.load(Ordering::Acquire);
        loop {
            let mut counts = unpack(cur);
            let used_total: u16 = counts.iter().copied().fold(0, u16::saturating_add);
            if used_total >= capacity {
                return false;
            }
            counts[lane] = counts[lane].saturating_add(1);
            match self.inflight_packed.compare_exchange_weak(
                cur,
                pack(counts),
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(observed) => cur = observed,
            }
        }
    }

    /// Release a previously acquired slot. Saturates at zero so a stray
    /// extra release cannot underflow the lane (defensive — production
    /// callers acquire and release in `SchedulerPermit::Drop`).
    ///
    /// Returns early without issuing a CAS when the lane is already
    /// zero. An unchanged-value atomic write would still trigger cache-
    /// line invalidation across cores via the coherence protocol, so
    /// the short-circuit matters on hot release paths.
    pub fn release(&self, class: Class) {
        let lane = class as usize;
        let mut cur = self.inflight_packed.load(Ordering::Acquire);
        loop {
            let counts = unpack(cur);
            if counts[lane] == 0 {
                return;
            }
            let mut new_counts = counts;
            new_counts[lane] -= 1;
            match self.inflight_packed.compare_exchange_weak(
                cur,
                pack(new_counts),
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(observed) => cur = observed,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::middleware::scheduler::Class;

    // ── pack/unpack pure-function round trip ─────────────────────────

    #[test]
    fn test_pack_unpack_round_trip() {
        let samples: [[u16; 4]; 6] = [
            [0, 0, 0, 0],
            [1, 2, 3, 4],
            [u16::MAX, 0, u16::MAX, 0],
            [0, u16::MAX, 0, u16::MAX],
            [10_000, 20_000, 30_000, 40_000],
            [u16::MAX; 4],
        ];
        for s in samples {
            assert_eq!(unpack(pack(s)), s);
        }
    }

    // ── slots_available_to: reservation honored across classes ───────

    #[test]
    fn test_slots_available_to_subtracts_higher_class_reservation() {
        // Capacity 128, System holds a 32-slot reservation (lane 3), no
        // inflight. A Bulk admission may use at most 128 - 32 = 96.
        let counts = [0, 0, 0, 0];
        let reserved = [0, 0, 0, 32];
        assert_eq!(slots_available_to(counts, reserved, 128, Class::Bulk), 96);
        // The reserving class itself gets the full capacity — its own
        // reservation is "available to" it.
        assert_eq!(
            slots_available_to(counts, reserved, 128, Class::System),
            128
        );
    }

    #[test]
    fn test_slots_available_to_ignores_same_class_reservation() {
        // A class's own reservation doesn't shrink its own headroom.
        let counts = [0, 0, 0, 0];
        let reserved = [32, 0, 0, 0];
        assert_eq!(slots_available_to(counts, reserved, 128, Class::Bulk), 128);
    }

    #[test]
    fn test_slots_available_to_consumed_reservation_releases_headroom() {
        // Once a higher class is already consuming its reservation, the
        // reservation hold collapses 1:1 — Bulk regains those slots.
        let counts = [0, 0, 0, 32];
        let reserved = [0, 0, 0, 32];
        // used_total = 32, unavailable_to_bulk = max(0, 32-32) = 0
        // → 128 - 32 - 0 = 96
        assert_eq!(slots_available_to(counts, reserved, 128, Class::Bulk), 96);
    }

    #[test]
    fn test_slots_available_to_returns_zero_when_used_meets_capacity() {
        let counts = [10, 10, 10, 10];
        let reserved = [0, 0, 0, 0];
        assert_eq!(slots_available_to(counts, reserved, 40, Class::Default), 0);
    }

    // ── try_acquire_slot / release_slot ──────────────────────────────

    #[test]
    fn test_try_acquire_increments_lane_then_release_decrements() {
        let pool = SlotPool::new(8, [0, 0, 0, 0]);
        assert!(pool.try_acquire(Class::Default));
        assert_eq!(pool.inflight(Class::Default), 1);
        pool.release(Class::Default);
        assert_eq!(pool.inflight(Class::Default), 0);
    }

    #[test]
    fn test_try_acquire_respects_higher_class_reservation() {
        // Capacity 4, System reserves 2. Bulk can take at most 2.
        let pool = SlotPool::new(4, [0, 0, 0, 2]);
        assert!(pool.try_acquire(Class::Bulk));
        assert!(pool.try_acquire(Class::Bulk));
        assert!(
            !pool.try_acquire(Class::Bulk),
            "Bulk must not consume System's reservation"
        );
        // System itself still has room.
        assert!(pool.try_acquire(Class::System));
        assert!(pool.try_acquire(Class::System));
        assert!(!pool.try_acquire(Class::System), "now truly full");
    }

    #[test]
    fn test_try_acquire_returns_false_when_full_without_spinning() {
        let pool = SlotPool::new(1, [0, 0, 0, 0]);
        assert!(pool.try_acquire(Class::Default));
        // Second call must return promptly, not spin forever.
        assert!(!pool.try_acquire(Class::Default));
    }

    #[test]
    fn test_try_acquire_ignoring_reservations_bypasses_guard() {
        // Capacity 4, System reserves 4 (locks Bulk out under the guard).
        let pool = SlotPool::new(4, [0, 0, 0, 4]);
        assert!(!pool.try_acquire(Class::Bulk), "guarded acquire is blocked");
        assert!(
            pool.try_acquire_ignoring_reservations(Class::Bulk),
            "starvation-override path bypasses the reservation guard"
        );
    }

    #[test]
    fn test_release_saturates_at_zero() {
        // Defensive: an extra release shouldn't underflow the lane.
        let pool = SlotPool::new(1, [0, 0, 0, 0]);
        pool.release(Class::Default);
        assert_eq!(pool.inflight(Class::Default), 0);
    }

    // ── concurrent contention: no over- or under-admission ───────────

    #[test]
    fn test_concurrent_admission_admits_exactly_capacity() {
        // 200 threads racing to acquire one Default slot against a pool
        // sized for 64. The CAS loop must guarantee exactly 64 successes
        // — no over-admission (race) or under-admission (giving up too
        // early). Uses std::thread rather than tokio so the test
        // exercises real OS-level concurrency without a runtime.
        use std::thread;

        let pool = Arc::new(SlotPool::new(64, [0, 0, 0, 0]));
        let handles: Vec<_> = (0..200)
            .map(|_| {
                let pool = pool.clone();
                thread::spawn(move || pool.try_acquire(Class::Default))
            })
            .collect();
        let successes = handles
            .into_iter()
            .filter_map(|h| h.join().ok())
            .filter(|admitted| *admitted)
            .count();
        assert_eq!(successes, 64);
        assert_eq!(pool.inflight(Class::Default), 64);
    }
}
