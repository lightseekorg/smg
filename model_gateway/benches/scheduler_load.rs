//! End-to-end concurrent **load** benchmark for the priority scheduler.
//!
//! Unlike `scheduler_admission.rs` (a single-threaded micro-benchmark of one
//! `admit()` call), this drives the *real* [`PriorityScheduler`] — with its
//! dispatcher running so queued waiters are actually admitted on slot release —
//! under genuine multi-threaded concurrency, modeling a realistic request
//! lifecycle:
//!
//! ```text
//!   admit() ──▶ pre-TTFT window (preemptible) ──▶ first byte ──▶ service ──▶ release
//!                     │ preempted (cancel fires)                  │
//!                     └──────────── release ───────────────────────
//! ```
//!
//! Two views:
//!
//! - **contended_throughput** — capacity is ample so every request is admitted;
//!   `K` concurrent tasks loop `admit → release` with no service hold. This is
//!   the scheduler's sustained ops/sec under real lock/atomic contention (the
//!   slot-pool CAS, the per-class queue mutex, the inflight-registry rwlock) as
//!   `K` rises. Answers "is the scheduler itself fast under concurrency?".
//!
//! - **saturated_load** — capacity is fixed and offered concurrency
//!   oversubscribes it, with a modeled per-request service time so queues
//!   actually form. Prints a per-class admit-latency (p50/p99) + outcome report
//!   (admitted / 429 queue-full / 503 preempted), then times sustained
//!   throughput. Shows the scheduler *doing its job* under load.

#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    clippy::print_stderr,
    reason = "benchmark harness: unwrap/expect on known-good setup; tokio::spawn is the point; eprintln is the report output"
)]

use std::{
    sync::{
        atomic::{AtomicI64, AtomicU64, Ordering},
        Arc, Once,
    },
    time::{Duration, Instant},
};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use smg::middleware::scheduler::{
    AdmitOutcome, Class, ClassConfig, PriorityScheduler, PrioritySchedulerYaml, RejectionReason,
    SchedulerSettings,
};
use smg_auth::RequestId;
use tokio::{runtime::Runtime, sync::watch};
use tokio_util::sync::CancellationToken;

/// Realistic settings: built-in per-class defaults (reservations,
/// preemption, queue depths) left intact, so the load test exercises the
/// real reservation + preemption behavior — not a flattened config.
fn settings() -> SchedulerSettings {
    use std::collections::HashMap;

    let mut classes = HashMap::new();
    for c in Class::ALL {
        classes.insert(c, ClassConfig::default_for(c));
    }
    let yaml = PrioritySchedulerYaml {
        classes,
        tenant_policies: HashMap::new(),
    };
    SchedulerSettings::from_cli_and_yaml(true, Class::Default, 32, Some(&yaml)).unwrap()
}

/// Build a scheduler with its dispatcher running at a fixed capacity. The
/// returned `watch::Sender` must be kept alive for the dispatcher to live.
fn build_running(capacity: u16) -> (Arc<PriorityScheduler>, watch::Sender<u16>) {
    let scheduler = PriorityScheduler::new(&settings(), capacity).expect("valid settings");
    let (cap_tx, cap_rx) = watch::channel(capacity);
    scheduler.spawn_dispatcher(cap_rx);
    (scheduler, cap_tx)
}

/// Weighted request class mix, ~ a real gateway: mostly `default`, a strong
/// `interactive` minority, a little `bulk` and `system`.
fn pick_class(seq: u64) -> Class {
    match seq % 20 {
        0..=11 => Class::Default,      // 60%
        12..=16 => Class::Interactive, // 25%
        17..=18 => Class::Bulk,        // 10%
        _ => Class::System,            // 5%
    }
}

#[derive(Default)]
struct Stats {
    admitted: u64,
    queue_full: u64,
    queue_timeout: u64,
    preempted: u64, // victim: cancel fired in the pre-TTFT window
    cancelled: u64,
    /// Admit latency (ns) for admitted requests, per `Class as usize`.
    lat: [Vec<u32>; 4],
}

impl Stats {
    fn merge(&mut self, o: Stats) {
        self.admitted += o.admitted;
        self.queue_full += o.queue_full;
        self.queue_timeout += o.queue_timeout;
        self.preempted += o.preempted;
        self.cancelled += o.cancelled;
        for i in 0..4 {
            self.lat[i].extend_from_slice(&o.lat[i]);
        }
    }
}

fn pct(sorted: &[u32], p: f64) -> u32 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = (((sorted.len() - 1) as f64) * p).round() as usize;
    sorted[idx]
}

/// Run `total` requests across `concurrency` concurrent client tasks against a
/// shared running scheduler. Each admitted request models a `service`-long hold
/// with a `ttft`-long preemptible head; if `service` is zero it releases
/// immediately (pure-contention throughput mode).
async fn run_load(
    scheduler: &Arc<PriorityScheduler>,
    seq: &Arc<AtomicU64>,
    concurrency: usize,
    total: u64,
    service: Duration,
    ttft: Duration,
) -> Stats {
    let remaining = Arc::new(AtomicI64::new(total as i64));
    let mut handles = Vec::with_capacity(concurrency);

    for _ in 0..concurrency {
        let scheduler = scheduler.clone();
        let remaining = remaining.clone();
        let seq = seq.clone();
        handles.push(tokio::spawn(async move {
            let mut st = Stats::default();
            while remaining.fetch_sub(1, Ordering::Relaxed) > 0 {
                let id = seq.fetch_add(1, Ordering::Relaxed);
                let class = pick_class(id);
                let request_id = RequestId(format!("ld-{id}"));

                let t0 = Instant::now();
                let outcome = scheduler
                    .admit(class, request_id, CancellationToken::new())
                    .await;
                let lat = t0.elapsed().as_nanos().min(u128::from(u32::MAX)) as u32;

                match outcome {
                    AdmitOutcome::Admitted(permit) => {
                        st.admitted += 1;
                        st.lat[class as usize].push(lat);
                        if service.is_zero() {
                            // pure-contention mode: release immediately.
                            drop(permit);
                        } else {
                            // Pre-TTFT window: preemptible. Race reaching the
                            // first byte against the scheduler preempting us.
                            let cancel = permit.cancel_token();
                            tokio::select! {
                                () = tokio::time::sleep(ttft) => {
                                    let _ = permit.try_mark_first_byte(); // now protected
                                    let rest = service.saturating_sub(ttft);
                                    if !rest.is_zero() {
                                        tokio::time::sleep(rest).await;
                                    }
                                }
                                () = cancel.cancelled() => {
                                    st.preempted += 1; // bumped before first byte
                                }
                            }
                            drop(permit); // releases the slot
                        }
                    }
                    AdmitOutcome::Rejected(RejectionReason::QueueFull) => st.queue_full += 1,
                    AdmitOutcome::Rejected(RejectionReason::QueueTimeout) => st.queue_timeout += 1,
                    AdmitOutcome::Rejected(RejectionReason::Preempted) => st.preempted += 1,
                    AdmitOutcome::Rejected(RejectionReason::ClientCancelled) => st.cancelled += 1,
                }
            }
            st
        }));
    }

    let mut total_stats = Stats::default();
    for h in handles {
        total_stats.merge(h.await.unwrap());
    }
    total_stats
}

fn multi_thread_rt() -> Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
}

/// Print a one-shot saturated-load report (latency percentiles + outcomes).
fn print_report(label: &str, st: &mut Stats) {
    let total = st.admitted + st.queue_full + st.queue_timeout + st.preempted + st.cancelled;
    eprintln!("\n=== scheduler load report: {label} ===");
    eprintln!(
        "requests={total}  admitted={}  429_queue_full={}  408_timeout={}  503_preempted={}  client_cancelled={}",
        st.admitted, st.queue_full, st.queue_timeout, st.preempted, st.cancelled
    );
    eprintln!("admit latency by class (microseconds):");
    eprintln!(
        "  {:<12} {:>8} {:>10} {:>10}",
        "class", "count", "p50_us", "p99_us"
    );
    for c in Class::ALL {
        let v = &mut st.lat[c as usize];
        if v.is_empty() {
            continue;
        }
        v.sort_unstable();
        eprintln!(
            "  {:<12} {:>8} {:>10.1} {:>10.1}",
            format!("{c:?}"),
            v.len(),
            f64::from(pct(v, 0.50)) / 1000.0,
            f64::from(pct(v, 0.99)) / 1000.0,
        );
    }
    eprintln!();
}

/// contended_throughput: ample capacity (everything admits), K concurrent
/// tasks loop admit→release with no hold. Pure scheduler ops/sec under
/// real K-way contention on the slot pool, queue locks, and registry.
fn bench_contended_throughput(c: &mut Criterion) {
    let rt = multi_thread_rt();
    // spawn_dispatcher uses tokio::spawn → build inside the runtime context,
    // but drop the enter guard before block_on (block_on can't nest in enter).
    let (scheduler, _cap_tx) = {
        let _enter = rt.enter();
        build_running(8192)
    };
    let seq = Arc::new(AtomicU64::new(0));

    let mut group = c.benchmark_group("contended_throughput");
    const N: u64 = 100_000;
    group.throughput(Throughput::Elements(N));
    for k in [1usize, 8, 32, 128, 512] {
        group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, &k| {
            b.iter(|| {
                rt.block_on(run_load(
                    &scheduler,
                    &seq,
                    k,
                    N,
                    Duration::ZERO,
                    Duration::ZERO,
                ));
            });
        });
    }
    group.finish();
}

/// saturated_load: fixed capacity, offered concurrency oversubscribes it, with
/// a modeled service time so queues form and preemption can fire. Prints a
/// per-class latency + outcome report once, then times sustained throughput.
fn bench_saturated_load(c: &mut Criterion) {
    const CAPACITY: u16 = 256;
    const OFFERED: usize = 1024; // 4x oversubscription
    let service = Duration::from_micros(800);
    let ttft = Duration::from_micros(150);

    let rt = multi_thread_rt();
    let (scheduler, _cap_tx) = {
        let _enter = rt.enter();
        build_running(CAPACITY)
    };
    let seq = Arc::new(AtomicU64::new(0));

    // One-shot instrumented run → human-readable report (printed once).
    static REPORT: Once = Once::new();
    REPORT.call_once(|| {
        let mut st = rt.block_on(run_load(&scheduler, &seq, OFFERED, 16_384, service, ttft));
        print_report(
            &format!(
                "cap={CAPACITY} offered={OFFERED} service={}us ttft={}us",
                service.as_micros(),
                ttft.as_micros()
            ),
            &mut st,
        );
    });

    let mut group = c.benchmark_group("saturated_load");
    const N: u64 = 8_192;
    group.throughput(Throughput::Elements(N));
    group.bench_function(format!("offered_{OFFERED}"), |b| {
        b.iter(|| {
            rt.block_on(run_load(&scheduler, &seq, OFFERED, N, service, ttft));
        });
    });
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = bench_contended_throughput, bench_saturated_load,
}
criterion_main!(benches);
