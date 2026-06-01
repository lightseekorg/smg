//! Admission-path microbenchmarks for the priority scheduler.
//!
//! The priority-admission scheduler sits on the request hot path: every
//! inbound request calls [`PriorityScheduler::admit`] before it reaches a
//! backend. This bench quantifies that per-request cost and contrasts it
//! with the legacy [`TokenBucket`] limiter it replaces.
//!
//! Groups:
//!
//! - **fast_path_admit** — the number that matters. Ample capacity, a slot
//!   is always free, so `admit` takes the synchronous fast path: a packed
//!   CAS on the slot pool, an `Arc` clone, an `InflightHandle` alloc, and a
//!   registry insert. Each iteration drops the permit so the next starts
//!   from an identical state (slot released, registry empty). Two flavors:
//!     - `acquire_inflight_sync`: the synchronous core, no runtime — the
//!       purest admission cost.
//!     - `admit_async`: the real entry point as production calls it, driven
//!       through a `current_thread` runtime with `block_on`.
//! - **admit_at_capacity** — scheduler saturated, target class has a
//!   zero-depth queue, so `admit` fails the fast path and the immediate
//!   enqueue rejects with `QueueFull`. The rejection hot path (a 429).
//! - **preemption_victim_search** — capacity full of N pre-marked
//!   (post-TTFT, non-preemptible) low-class inflights. A preempt-capable
//!   admission scans the whole inflight registry looking for a victim,
//!   finds none, and falls through to a `QueueFull` rejection. Isolates the
//!   O(N) registry scan cost.
//! - **baseline_token_bucket** — `TokenBucket::try_acquire(1.0)` +
//!   `return_tokens(1.0)`, the thing being replaced. Measured both
//!   sync-direct and wrapped in `block_on`, so the per-iteration runtime
//!   overhead can be subtracted when comparing against `admit_async`.

#![expect(clippy::unwrap_used, clippy::expect_used)]

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use smg::middleware::{
    scheduler::{
        AdmitOutcome, Class, ClassConfig, PriorityScheduler, PrioritySchedulerYaml,
        RejectionReason, SchedulerPermit, SchedulerSettings,
    },
    TokenBucket,
};
use smg_auth::RequestId;
use tokio::runtime::Builder;
use tokio_util::sync::CancellationToken;

/// Build settings with `reserved = 0` on every class (so admission is
/// purely capacity-bound and small capacities don't trip the
/// reserved-vs-capacity guard) and the given per-class `queue_size`.
/// Other per-class fields stay at their built-in defaults.
fn settings(queue_size: u32) -> SchedulerSettings {
    use std::collections::HashMap;

    let mut classes = HashMap::new();
    for c in Class::ALL {
        let mut cfg = ClassConfig::default_for(c);
        cfg.reserved = 0;
        cfg.queue_size = queue_size;
        classes.insert(c, cfg);
    }
    let yaml = PrioritySchedulerYaml {
        classes,
        tenant_policies: HashMap::new(),
    };
    SchedulerSettings::from_cli_and_yaml(true, Class::Default, 32, Some(&yaml)).unwrap()
}

fn rid(s: &str) -> RequestId {
    RequestId(s.to_string())
}

/// fast_path_admit: the per-request admission hot path with a slot always
/// available.
fn bench_fast_path_admit(c: &mut Criterion) {
    let mut group = c.benchmark_group("fast_path_admit");

    // Ample capacity, generous queue (never reached on this path).
    let scheduler = PriorityScheduler::new(&settings(512), 4096).unwrap();
    // Stable request id reused every iteration. `admit`/`acquire_inflight`
    // take ownership, so the clone (a `String` clone) is part of the
    // measured per-request cost — exactly what production pays, since each
    // real request carries a distinct id.
    let request_id = rid("bench-fast-path");

    // (a) Synchronous core: try_acquire CAS + registry insert + handle
    // alloc, then drop releases the slot and clears the registry. No async
    // machinery — the purest admission cost.
    group.bench_function("acquire_inflight_sync", |b| {
        b.iter(|| {
            let permit = scheduler
                .acquire_inflight(black_box(Class::Default), black_box(request_id.clone()))
                .expect("slot available");
            black_box(&permit);
            // Drop here releases the slot for the next iteration.
        });
    });

    // (b) Real entry point, async, driven through a current-thread runtime.
    // Difference vs (a) is the cost of entering the future + the fast-path
    // branch checks in `admit`.
    let rt = Builder::new_current_thread().enable_all().build().unwrap();
    group.bench_function("admit_async", |b| {
        b.iter(|| {
            rt.block_on(async {
                let outcome = scheduler
                    .admit(
                        black_box(Class::Default),
                        black_box(request_id.clone()),
                        CancellationToken::new(),
                    )
                    .await;
                // Ample capacity → always Admitted. Hold then drop the
                // outcome (and its permit) so the slot frees for the next
                // iteration. `assert!` keeps the path honest without the
                // (deny-listed) `unreachable!`.
                assert!(matches!(outcome, AdmitOutcome::Admitted(_)));
                black_box(outcome);
            });
        });
    });

    group.finish();
}

/// admit_at_capacity: scheduler saturated, queue depth 0 → immediate
/// `QueueFull` rejection. Uses `Class::Default` (can_preempt = false) so the
/// preemption block is skipped and the path is purely fast-path-miss →
/// enqueue-reject.
fn bench_admit_at_capacity(c: &mut Criterion) {
    let mut group = c.benchmark_group("admit_at_capacity");

    const CAP: u16 = 256;
    // queue_size = 0 so the slow-path enqueue rejects instantly with no
    // queue contention and no awaiting.
    let scheduler = PriorityScheduler::new(&settings(0), CAP).unwrap();

    // Saturate every slot with held Default permits.
    let mut held: Vec<SchedulerPermit> = Vec::with_capacity(CAP as usize);
    for i in 0..CAP {
        held.push(
            scheduler
                .acquire_inflight(Class::Default, rid(&format!("held-{i}")))
                .expect("under capacity"),
        );
    }

    let rt = Builder::new_current_thread().enable_all().build().unwrap();
    let request_id = rid("bench-at-capacity");
    group.bench_function("queue_full_reject", |b| {
        b.iter(|| {
            rt.block_on(async {
                let outcome = scheduler
                    .admit(
                        black_box(Class::Default),
                        black_box(request_id.clone()),
                        CancellationToken::new(),
                    )
                    .await;
                assert!(matches!(
                    outcome,
                    AdmitOutcome::Rejected(RejectionReason::QueueFull)
                ));
            });
        });
    });

    drop(held);
    group.finish();
}

/// preemption_victim_search: with N inflight low-class requests, all marked
/// post-TTFT (non-preemptible), a preempt-capable admission scans the full
/// registry, finds no eligible victim, and rejects with `QueueFull`
/// (queue_size = 0). Isolates the O(N) `find_preemption_victim` scan.
fn bench_preemption_victim_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("preemption_victim_search");

    for &n in &[64usize, 256, 1024] {
        let cap = n as u16;
        let scheduler = PriorityScheduler::new(&settings(0), cap).unwrap();

        // Fill capacity with Bulk inflights and mark each past TTFT so it is
        // present in the registry but NOT a valid preemption victim. The
        // search must therefore iterate all N and reject every one.
        let mut held: Vec<SchedulerPermit> = Vec::with_capacity(n);
        for i in 0..n {
            let permit = scheduler
                .acquire_inflight(Class::Bulk, rid(&format!("victim-{i}")))
                .expect("under capacity");
            // try_mark_first_byte(1) → is_preemptible() == false.
            assert!(permit.handle().try_mark_first_byte(1));
            held.push(permit);
        }

        let rt = Builder::new_current_thread().enable_all().build().unwrap();
        let request_id = rid("preemptor");
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                rt.block_on(async {
                    // Interactive can_preempt = true → runs the victim scan.
                    // No victim is eligible → falls through to enqueue →
                    // QueueFull (queue_size = 0).
                    let outcome = scheduler
                        .admit(
                            black_box(Class::Interactive),
                            black_box(request_id.clone()),
                            CancellationToken::new(),
                        )
                        .await;
                    assert!(matches!(
                        outcome,
                        AdmitOutcome::Rejected(RejectionReason::QueueFull)
                    ));
                });
            });
        });

        drop(held);
    }

    group.finish();
}

/// baseline_token_bucket: the legacy limiter being replaced. A successful
/// `try_acquire(1.0)` plus the matching `return_tokens(1.0)` — the analogue
/// of one admit + one permit release. Measured sync-direct and through
/// `block_on` so the runtime overhead in `admit_async` can be subtracted.
fn bench_baseline_token_bucket(c: &mut Criterion) {
    let mut group = c.benchmark_group("baseline_token_bucket");

    // Pure concurrency limiter: large capacity, refill_rate = 0 (tokens only
    // return via return_tokens), mirroring the scheduler's slot semantics.
    let bucket = TokenBucket::new(4096, 0);

    // Sync-direct: the actual cost of the limiter's acquire+release.
    group.bench_function("try_acquire_return_sync", |b| {
        b.iter(|| {
            black_box(bucket.try_acquire(black_box(1.0))).unwrap();
            bucket.return_tokens(black_box(1.0));
        });
    });

    // block_on-wrapped: same work under the same per-iteration runtime
    // overhead as `fast_path_admit/admit_async`, for an apples-to-apples
    // delta.
    let rt = Builder::new_current_thread().enable_all().build().unwrap();
    group.bench_function("try_acquire_return_block_on", |b| {
        b.iter(|| {
            rt.block_on(async {
                black_box(bucket.try_acquire(black_box(1.0))).unwrap();
                bucket.return_tokens(black_box(1.0));
            });
        });
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets =
        bench_fast_path_admit,
        bench_admit_at_capacity,
        bench_preemption_victim_search,
        bench_baseline_token_bucket,
}
criterion_main!(benches);
