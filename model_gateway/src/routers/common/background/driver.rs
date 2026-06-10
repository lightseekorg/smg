//! Background-job driver.
//!
//! A thin loop over the [`BackgroundResponseRepository`] trait — *not* a bespoke
//! queue or scheduling algorithm. The durable claim / lease / retry logic lives
//! in the repository; this driver only decides *when* to call it:
//!
//! - a **startup claim pass** that drains everything claimable as soon as the
//!   process comes up (so a replica restart picks up in-flight work
//!   immediately),
//! - a **claim-tick loop** that periodically (jittered) — and on an in-process
//!   wakeup [`Notify`] — claims runnable jobs while concurrency permits are
//!   available and hands each to the [`BackgroundWorker`],
//! - a **sweeper loop** that periodically calls
//!   [`BackgroundResponseRepository::requeue_expired`] and nudges the claim loop
//!   when it reclaims anything.
//!
//! The periodic tick + startup pass are the correctness path; the `Notify`
//! nudge is a latency optimization (mirroring `LISTEN/NOTIFY` for the durable
//! backends, which is explicitly *not* required for correctness — see the
//! recovered design, Part A.9).

use std::{sync::Arc, time::Duration};

use rand::Rng as _;
use smg_data_connector::BackgroundResponseRepository;
use tokio::{
    sync::{Notify, OwnedSemaphorePermit, Semaphore},
    task::JoinHandle,
};
use tracing::{debug, error, info};
use uuid::Uuid;

use super::{supervisor::spawn_supervised_periodic, worker::BackgroundWorker};
use crate::config::BackgroundConfig;

/// Upper bound on the random jitter added to the claim-tick interval, expressed
/// as a fraction of the base interval. Spreads claim ticks across replicas so
/// they don't stampede the backend on the same cadence.
const CLAIM_JITTER_FRACTION: f64 = 0.2;

/// Drives background-job execution by polling the repository trait.
///
/// Holds an `Arc<dyn BackgroundResponseRepository>` (durable claim/lease logic),
/// an `Arc<dyn BackgroundWorker>` (per-job execution), the [`BackgroundConfig`]
/// that tunes the loops, a stable `worker_id` used for leases, a [`Semaphore`]
/// sized to the configured worker concurrency, and a claim-wakeup [`Notify`].
pub struct BackgroundDriver {
    repository: Arc<dyn BackgroundResponseRepository>,
    worker: Arc<dyn BackgroundWorker>,
    config: BackgroundConfig,
    worker_id: String,
    permits: Arc<Semaphore>,
    claim_wakeup: Arc<Notify>,
}

/// Handles for the supervised loops a [`BackgroundDriver`] spawns. Hold this
/// for the lifetime of the process; dropping the contained [`BackgroundDriver`]
/// `Arc` (and these handles) stops the loops.
pub struct BackgroundDriverHandle {
    /// Kept alive so the loops' `Weak` upgrades keep succeeding.
    _driver: Arc<BackgroundDriver>,
    _claim_tick: JoinHandle<()>,
    _sweeper: JoinHandle<()>,
}

impl BackgroundDriver {
    /// Construct a driver. `worker_id` is generated here and is stable for
    /// the life of this driver instance — it identifies this replica's
    /// leases to the repository.
    pub fn new(
        repository: Arc<dyn BackgroundResponseRepository>,
        worker: Arc<dyn BackgroundWorker>,
        config: BackgroundConfig,
    ) -> Self {
        let concurrency = config.worker_concurrency.max(1) as usize;
        let worker_id = format!("bg-{}", Uuid::now_v7());
        Self {
            repository,
            worker,
            config,
            worker_id,
            permits: Arc::new(Semaphore::new(concurrency)),
            claim_wakeup: Arc::new(Notify::new()),
        }
    }

    /// The stable lease identity for this driver instance.
    pub fn worker_id(&self) -> &str {
        &self.worker_id
    }

    /// Wake the claim loop early (e.g. right after a job is enqueued). This is a
    /// latency optimization only; the periodic tick would claim the job anyway.
    pub fn notify_claim(&self) {
        self.claim_wakeup.notify_one();
    }

    /// Consume the driver, run the startup claim pass, and spawn the
    /// supervised claim-tick and sweeper loops.
    ///
    /// Returns a [`BackgroundDriverHandle`] that owns the driver `Arc` and
    /// both task handles; keep it alive for the process lifetime.
    ///
    /// Not called at startup in BGM-PR-06: the driver is wired and started by
    /// BGM-PR-07 (#1221) once a real [`BackgroundWorker`] exists. Until then a
    /// `background=true` request stays durably `queued` (per #1614).
    pub async fn spawn(self) -> BackgroundDriverHandle {
        let driver = Arc::new(self);

        info!(
            worker_id = %driver.worker_id,
            concurrency = driver.config.worker_concurrency,
            poll_interval_ms = driver.config.poll_interval_ms,
            sweep_interval_secs = driver.config.sweep_interval_secs,
            "starting background driver"
        );

        // Startup claim pass: drain everything claimable right now so a fresh
        // replica picks up pending work without waiting for the first tick.
        driver.claim_drain().await;

        let claim_interval = driver.jittered_claim_interval();
        let claim_tick = spawn_supervised_periodic(
            "background-claim-tick",
            Arc::downgrade(&driver),
            claim_interval,
            Some(Arc::downgrade(&driver.claim_wakeup)),
            |s: Arc<BackgroundDriver>| async move {
                s.claim_drain().await;
            },
        );

        let sweeper = spawn_supervised_periodic(
            "background-sweeper",
            Arc::downgrade(&driver),
            driver.config.sweep_interval(),
            // The sweeper is purely time-driven; it has no early wakeup.
            None,
            |s: Arc<BackgroundDriver>| async move {
                s.sweep_once().await;
            },
        );

        BackgroundDriverHandle {
            _driver: driver,
            _claim_tick: claim_tick,
            _sweeper: sweeper,
        }
    }

    /// Compute the claim-tick interval with a small additive jitter so replicas
    /// don't poll in lockstep.
    fn jittered_claim_interval(&self) -> Duration {
        let base = self.config.poll_interval();
        let jitter_ceiling = base.as_secs_f64() * CLAIM_JITTER_FRACTION;
        let jitter = rand::rng().random_range(0.0..=jitter_ceiling.max(0.0));
        base + Duration::from_secs_f64(jitter)
    }

    /// Claim and dispatch runnable jobs until either no concurrency permit is
    /// free or the repository has nothing left to claim. Each dispatched job
    /// runs on its own task holding a concurrency permit until it finishes.
    async fn claim_drain(self: &Arc<Self>) {
        loop {
            // Acquire an owned permit *before* claiming so a claimed job always
            // has a slot to run in; if none is free, stop and let the running
            // jobs (or the next tick) make progress.
            let Ok(permit) = Arc::clone(&self.permits).try_acquire_owned() else {
                debug!(
                    worker_id = %self.worker_id,
                    "background claim loop: no concurrency permit available; yielding"
                );
                return;
            };

            let now = chrono::Utc::now();
            let lease = self.config.lease_duration();
            let claimed = match self
                .repository
                .claim_next(&self.worker_id, now, lease)
                .await
            {
                Ok(job) => job,
                Err(e) => {
                    // Drop the permit (it falls out of scope) and back off until
                    // the next tick rather than hot-looping on a backend error.
                    error!(
                        worker_id = %self.worker_id,
                        error = %e,
                        "background claim_next failed; will retry on next tick"
                    );
                    return;
                }
            };

            let Some(job) = claimed else {
                // Nothing to claim; the permit is released as it drops.
                return;
            };

            self.dispatch(job, permit);
        }
    }

    /// Spawn the execution of one claimed job, moving the concurrency `permit`
    /// into the task so the slot stays reserved until execution finishes.
    fn dispatch(
        self: &Arc<Self>,
        job: smg_data_connector::LeasedJob,
        permit: OwnedSemaphorePermit,
    ) {
        let worker = Arc::clone(&self.worker);
        // Cloned into the task so we can nudge the claim loop the instant this
        // job's permit frees, instead of leaving the freed slot idle until the
        // next poll tick (which would cap throughput at one batch per interval).
        let claim_wakeup = Arc::clone(&self.claim_wakeup);
        let response_id = job.response_id.0.clone();
        debug!(
            worker_id = %self.worker_id,
            response_id = %response_id,
            "dispatching background job"
        );
        #[expect(
            clippy::disallowed_methods,
            reason = "per-job execution task; the moved permit bounds concurrency and is released on completion"
        )]
        tokio::spawn(async move {
            worker.execute(job).await;
            // Release the permit, then wake the claim loop so it immediately
            // re-claims into the freed slot rather than idling until the next
            // poll tick.
            drop(permit);
            claim_wakeup.notify_one();
            debug!(response_id = %response_id, "background job execution finished");
        });
    }

    /// Run one sweeper pass: requeue expired leases and, if anything was
    /// reclaimed, nudge the claim loop so the reclaimed jobs are picked up
    /// without waiting for the next claim tick.
    async fn sweep_once(self: &Arc<Self>) {
        let now = chrono::Utc::now();
        match self.repository.requeue_expired(now).await {
            Ok(0) => {}
            Ok(n) => {
                info!(
                    worker_id = %self.worker_id,
                    requeued = n,
                    "background sweeper requeued expired leases"
                );
                self.claim_wakeup.notify_one();
            }
            Err(e) => {
                error!(
                    worker_id = %self.worker_id,
                    error = %e,
                    "background sweeper requeue_expired failed"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashSet,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Mutex,
        },
        time::Duration,
    };

    use async_trait::async_trait;
    use chrono::Utc;
    use serde_json::json;
    use smg_data_connector::{
        BackgroundResponseRepository, EnqueueRequest, FinalizeRequest, FinalizeStatus, LeasedJob,
        MemoryBackgroundRepository, MemoryResponseStorage, ResponseId,
    };
    use tokio::time::timeout;

    use super::*;
    use crate::routers::common::background::worker::BackgroundWorker;

    fn enqueue_req(id: &str) -> EnqueueRequest {
        EnqueueRequest::new(
            ResponseId::from(id),
            "gpt-5.1".to_string(),
            json!({"model": "gpt-5.1"}),
            json!([]),
            json!({"id": id, "status": "queued"}),
            false,
            0,
        )
    }

    /// Records the response ids it sees and finalizes each job as completed so
    /// the row terminalizes (and is not re-claimed). Optionally blocks on a
    /// barrier so a test can observe the concurrency cap.
    struct RecordingWorker {
        repository: Arc<dyn BackgroundResponseRepository>,
        seen: Mutex<Vec<String>>,
        in_flight: AtomicUsize,
        max_in_flight: AtomicUsize,
        hold: Option<Arc<Notify>>,
    }

    impl RecordingWorker {
        fn new(
            repository: Arc<dyn BackgroundResponseRepository>,
            hold: Option<Arc<Notify>>,
        ) -> Self {
            Self {
                repository,
                seen: Mutex::new(Vec::new()),
                in_flight: AtomicUsize::new(0),
                max_in_flight: AtomicUsize::new(0),
                hold,
            }
        }

        fn seen_ids(&self) -> Vec<String> {
            self.seen.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl BackgroundWorker for RecordingWorker {
        async fn execute(&self, job: LeasedJob) {
            let cur = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_in_flight.fetch_max(cur, Ordering::SeqCst);
            self.seen.lock().unwrap().push(job.response_id.0.clone());

            if let Some(hold) = &self.hold {
                hold.notified().await;
            }

            let now = Utc::now();
            let _ = self
                .repository
                .finalize(
                    FinalizeRequest::new(
                        job.response_id.clone(),
                        job.worker_id.clone(),
                        FinalizeStatus::Completed,
                        json!({"id": job.response_id.0, "status": "completed"}),
                        now,
                    ),
                    now,
                )
                .await;

            self.in_flight.fetch_sub(1, Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn startup_pass_dispatches_all_jobs_to_terminal() {
        let rs = Arc::new(MemoryResponseStorage::new());
        let repo: Arc<dyn BackgroundResponseRepository> =
            Arc::new(MemoryBackgroundRepository::new(Arc::clone(&rs)));
        for i in 0..5 {
            repo.enqueue(enqueue_req(&format!("r{i}")), None, None)
                .await
                .unwrap();
        }

        let worker = Arc::new(RecordingWorker::new(Arc::clone(&repo), None));
        let worker_dyn: Arc<dyn BackgroundWorker> = Arc::clone(&worker) as _;
        let config = BackgroundConfig {
            worker_concurrency: 4,
            ..Default::default()
        };
        let driver = BackgroundDriver::new(Arc::clone(&repo), worker_dyn, config);
        let _handle = driver.spawn().await;

        // All five enqueued jobs must be seen and terminalized.
        let done = timeout(Duration::from_secs(5), async {
            loop {
                let seen: HashSet<String> = worker.seen_ids().into_iter().collect();
                if seen.len() == 5 {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await;
        assert!(
            done.is_ok(),
            "not all jobs dispatched: {:?}",
            worker.seen_ids()
        );

        // The queue must be empty: nothing left to claim.
        let leftover = repo
            .claim_next("probe", Utc::now(), Duration::from_secs(30))
            .await
            .unwrap();
        assert!(leftover.is_none(), "expected all jobs terminalized");
    }

    #[tokio::test]
    async fn respects_concurrency_cap() {
        let rs = Arc::new(MemoryResponseStorage::new());
        let repo: Arc<dyn BackgroundResponseRepository> =
            Arc::new(MemoryBackgroundRepository::new(Arc::clone(&rs)));
        for i in 0..6 {
            repo.enqueue(enqueue_req(&format!("r{i}")), None, None)
                .await
                .unwrap();
        }

        // Workers block on `hold` until we release them, so we can observe how
        // many run concurrently. Cap is 2.
        let hold = Arc::new(Notify::new());
        let worker = Arc::new(RecordingWorker::new(
            Arc::clone(&repo),
            Some(Arc::clone(&hold)),
        ));
        let worker_dyn: Arc<dyn BackgroundWorker> = Arc::clone(&worker) as _;
        let config = BackgroundConfig {
            worker_concurrency: 2,
            ..Default::default()
        };
        let driver = BackgroundDriver::new(Arc::clone(&repo), worker_dyn, config);
        let _handle = driver.spawn().await;

        // Wait until 2 jobs are in flight (the cap), then confirm it never
        // exceeds 2 while they're held.
        let reached = timeout(Duration::from_secs(3), async {
            loop {
                if worker.in_flight.load(Ordering::SeqCst) >= 2 {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await;
        assert!(reached.is_ok(), "never reached the concurrency cap");
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            worker.in_flight.load(Ordering::SeqCst) <= 2,
            "exceeded concurrency cap"
        );

        // Release all held workers; everything must drain.
        for _ in 0..6 {
            hold.notify_one();
        }
        // Some workers were not yet started (blocked behind the cap); keep
        // notifying as they come online.
        let drained = timeout(Duration::from_secs(5), async {
            loop {
                hold.notify_one();
                let seen: HashSet<String> = worker.seen_ids().into_iter().collect();
                if seen.len() == 6 {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await;
        assert!(drained.is_ok(), "jobs did not drain after release");
        assert!(
            worker.max_in_flight.load(Ordering::SeqCst) <= 2,
            "max concurrency {} exceeded cap of 2",
            worker.max_in_flight.load(Ordering::SeqCst)
        );
    }

    #[tokio::test]
    async fn sweeper_reclaims_expired_lease() {
        // Claim a job out-of-band with a short lease, let it expire, and assert
        // the driver's sweeper reclaims it (queue becomes claimable again).
        let rs = Arc::new(MemoryResponseStorage::new());
        let repo: Arc<dyn BackgroundResponseRepository> =
            Arc::new(MemoryBackgroundRepository::new(Arc::clone(&rs)));
        repo.enqueue(enqueue_req("r1"), None, None).await.unwrap();

        // Out-of-band claim with a 1s lease by a different worker so the
        // driver's own claim loop can't grab it first.
        let claimed = repo
            .claim_next("external", Utc::now(), Duration::from_secs(1))
            .await
            .unwrap()
            .expect("claimed");
        assert_eq!(claimed.response_id, ResponseId::from("r1"));

        // A no-op worker: we only care about the sweeper reclaiming, not
        // execution. (It will finalize whatever the driver later claims.)
        struct NoopFinalizeWorker {
            repository: Arc<dyn BackgroundResponseRepository>,
        }
        #[async_trait]
        impl BackgroundWorker for NoopFinalizeWorker {
            async fn execute(&self, job: LeasedJob) {
                let now = Utc::now();
                let _ = self
                    .repository
                    .finalize(
                        FinalizeRequest::new(
                            job.response_id.clone(),
                            job.worker_id.clone(),
                            FinalizeStatus::Completed,
                            json!({"id": job.response_id.0, "status": "completed"}),
                            now,
                        ),
                        now,
                    )
                    .await;
            }
        }

        let worker: Arc<dyn BackgroundWorker> = Arc::new(NoopFinalizeWorker {
            repository: Arc::clone(&repo),
        });
        // Fast sweep so the test doesn't wait long; long poll so the driver's
        // claim loop doesn't race the assertion before the lease expires.
        let config = BackgroundConfig {
            worker_concurrency: 4,
            lease_duration_secs: 1,
            sweep_interval_secs: 1,
            poll_interval_ms: 50,
            ..Default::default()
        };
        let driver = BackgroundDriver::new(Arc::clone(&repo), worker, config);
        let _handle = driver.spawn().await;

        // After the lease (1s) expires and the sweeper (1s) runs, the row is
        // requeued and the driver re-claims + finalizes it. Eventually the
        // queue is empty (job terminalized via the re-claim path).
        let terminalized = timeout(Duration::from_secs(8), async {
            loop {
                let probe = repo
                    .claim_next("probe", Utc::now(), Duration::from_secs(30))
                    .await
                    .unwrap();
                // `probe` claims only if the driver hasn't already taken it.
                // Once terminal, neither probe nor driver can claim, so put
                // any probe-claim back by finalizing it (keeps the loop honest).
                match probe {
                    Some(job) => {
                        let now = Utc::now();
                        let _ = repo
                            .finalize(
                                FinalizeRequest::new(
                                    job.response_id.clone(),
                                    "probe".to_string(),
                                    FinalizeStatus::Completed,
                                    json!({"status": "completed"}),
                                    now,
                                ),
                                now,
                            )
                            .await;
                        break;
                    }
                    None => {
                        // Could be in-progress (driver holds it) or terminal.
                        // Check the stored payload: terminal means done.
                        use smg_data_connector::ResponseStorage;
                        let stored = rs.get_response(&ResponseId::from("r1")).await.unwrap();
                        if let Some(s) = stored {
                            if s.raw_response["status"] == "completed" {
                                break;
                            }
                        }
                        tokio::time::sleep(Duration::from_millis(50)).await;
                    }
                }
            }
        })
        .await;
        assert!(
            terminalized.is_ok(),
            "sweeper did not reclaim the expired-lease job"
        );
    }
}
