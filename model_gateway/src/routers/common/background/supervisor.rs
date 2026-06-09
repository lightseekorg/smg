//! Supervised periodic task primitive.
//!
//! A small generic helper that runs an async callback on a fixed interval,
//! additionally waking early when an optional [`Notify`] is signalled, and
//! restarts the whole loop after a short backoff if the callback panics.
//!
//! This mirrors the supervision shape of
//! [`crate::worker::capacity::WorkerCapacity::spawn`] (catch-unwind +
//! restart-with-backoff + `Weak` references so the spawned task never keeps the
//! owning state alive), but factored out as a reusable primitive so the
//! background scheduler's claim-tick and sweeper loops share one
//! implementation instead of hand-rolling a third and fourth copy.

use std::{future::Future, sync::Weak, time::Duration};

use futures::FutureExt as _;
use tokio::{sync::Notify, task::JoinHandle, time::MissedTickBehavior};

/// Backoff applied before restarting the loop after the callback panics.
const PANIC_RESTART_BACKOFF: Duration = Duration::from_secs(1);

/// Spawn a supervised task that invokes `tick` every `interval`, or sooner when
/// `wakeup` (if provided) is notified.
///
/// The task owns a `Weak<S>` to some shared state `S`. On each wake it upgrades
/// the weak reference and passes the strong `Arc<S>` to `tick`; if the upgrade
/// fails (the owner dropped its last `Arc`), the task exits cleanly. This keeps
/// the long-lived task from forming an `Arc` cycle with the state it drives —
/// exactly the lifecycle contract `WorkerCapacity` documents.
///
/// `tick` is invoked once immediately on each wake (interval tick or notify);
/// any panic inside it is caught, logged, and the supervisor restarts the inner
/// loop after [`PANIC_RESTART_BACKOFF`]. A clean return from the inner loop
/// (only reachable when the weak state is gone) terminates the task.
///
/// `name` is used purely for log lines so operators can tell the loops apart.
///
/// Returns the [`JoinHandle`]; callers should retain it for the lifetime of the
/// process (dropping it detaches the task, dropping the last `Arc<S>` stops it).
pub fn spawn_supervised_periodic<S, F, Fut>(
    name: &'static str,
    state: Weak<S>,
    interval: Duration,
    wakeup: Option<Weak<Notify>>,
    tick: F,
) -> JoinHandle<()>
where
    S: Send + Sync + 'static,
    F: Fn(std::sync::Arc<S>) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = ()> + Send,
{
    #[expect(
        clippy::disallowed_methods,
        reason = "supervised long-lived task: panics are caught and the loop restarts; holds Weak refs so it cannot keep the owning state alive"
    )]
    tokio::spawn(async move {
        loop {
            let result = std::panic::AssertUnwindSafe(run_loop(
                name,
                state.clone(),
                interval,
                wakeup.clone(),
                &tick,
            ))
            .catch_unwind()
            .await;

            match result {
                Ok(()) => break,
                Err(payload) => {
                    let msg = panic_message(&payload);
                    tracing::error!(
                        task = name,
                        panic.message = %msg,
                        backoff_secs = PANIC_RESTART_BACKOFF.as_secs(),
                        "supervised periodic task panicked; restarting"
                    );
                    tokio::time::sleep(PANIC_RESTART_BACKOFF).await;
                }
            }
        }
    })
}

/// The inner loop, separated so the supervisor can `catch_unwind` around it.
async fn run_loop<S, F, Fut>(
    name: &'static str,
    state: Weak<S>,
    interval: Duration,
    wakeup: Option<Weak<Notify>>,
    tick: &F,
) where
    S: Send + Sync + 'static,
    F: Fn(std::sync::Arc<S>) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = ()> + Send,
{
    let mut ticker = tokio::time::interval(interval);
    // A panic-restart or a slow tick must not produce a burst of catch-up
    // ticks; one wake per missed period is enough for a poll loop.
    ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
    // The first `tick()` on a fresh interval returns immediately; the caller's
    // startup pass (if any) already drained the queue, so consume that eager
    // tick up front and wait a full period before the first scheduled wake.
    ticker.tick().await;

    loop {
        // Upgrade the wakeup each iteration so a dropped owner lets it go.
        let notified = async {
            match wakeup.as_ref().and_then(Weak::upgrade) {
                Some(n) => n.notified().await,
                // No wakeup configured (or it was dropped): fall back to the
                // interval alone by never resolving this branch.
                None => std::future::pending::<()>().await,
            }
        };

        tokio::select! {
            _ = ticker.tick() => {}
            () = notified => {}
        }

        let Some(state) = state.upgrade() else {
            tracing::debug!(
                task = name,
                "supervised periodic task exiting: owner dropped"
            );
            return;
        };
        tick(state).await;
    }
}

/// Best-effort extraction of a panic message for logging.
fn panic_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "(non-string panic)".to_string())
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    };

    use tokio::time::{timeout, Duration};

    use super::*;

    struct Counter {
        ticks: AtomicU32,
    }

    #[tokio::test]
    async fn fires_on_notify_before_interval_elapses() {
        let state = Arc::new(Counter {
            ticks: AtomicU32::new(0),
        });
        let notify = Arc::new(Notify::new());

        // Long interval so any tick within the test window must come from the
        // notify path, not the timer.
        let _handle = spawn_supervised_periodic(
            "test-notify",
            Arc::downgrade(&state),
            Duration::from_secs(3600),
            Some(Arc::downgrade(&notify)),
            |s: Arc<Counter>| async move {
                s.ticks.fetch_add(1, Ordering::SeqCst);
            },
        );

        notify.notify_one();

        let observed = timeout(Duration::from_secs(2), async {
            loop {
                if state.ticks.load(Ordering::SeqCst) >= 1 {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await;
        assert!(observed.is_ok(), "notify did not trigger a tick");
    }

    #[tokio::test]
    async fn fires_on_interval_without_notify() {
        let state = Arc::new(Counter {
            ticks: AtomicU32::new(0),
        });

        let _handle = spawn_supervised_periodic(
            "test-interval",
            Arc::downgrade(&state),
            Duration::from_millis(20),
            None,
            |s: Arc<Counter>| async move {
                s.ticks.fetch_add(1, Ordering::SeqCst);
            },
        );

        let observed = timeout(Duration::from_secs(2), async {
            loop {
                if state.ticks.load(Ordering::SeqCst) >= 2 {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await;
        assert!(observed.is_ok(), "interval did not produce repeated ticks");
    }

    #[tokio::test]
    async fn restarts_loop_after_panic() {
        let state = Arc::new(Counter {
            ticks: AtomicU32::new(0),
        });

        // Each tick panics after bumping the counter. The supervisor must
        // restart the loop (after backoff) so the counter keeps climbing past
        // the first panic.
        let _handle = spawn_supervised_periodic(
            "test-panic",
            Arc::downgrade(&state),
            Duration::from_millis(10),
            None,
            |s: Arc<Counter>| async move {
                s.ticks.fetch_add(1, Ordering::SeqCst);
                panic!("boom");
            },
        );

        // Backoff is 1s per restart, so within ~4s we expect at least 2 ticks
        // (the initial tick + at least one post-restart tick).
        let observed = timeout(Duration::from_secs(4), async {
            loop {
                if state.ticks.load(Ordering::SeqCst) >= 2 {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await;
        assert!(observed.is_ok(), "loop did not restart after panic");
    }

    #[tokio::test]
    async fn exits_when_owner_dropped() {
        let state = Arc::new(Counter {
            ticks: AtomicU32::new(0),
        });
        let weak = Arc::downgrade(&state);

        let handle = spawn_supervised_periodic(
            "test-exit",
            weak.clone(),
            Duration::from_millis(10),
            None,
            |s: Arc<Counter>| async move {
                s.ticks.fetch_add(1, Ordering::SeqCst);
            },
        );

        // Drop the only strong ref; the next upgrade fails and the task ends.
        drop(state);

        let joined = timeout(Duration::from_secs(2), handle).await;
        assert!(
            joined.is_ok(),
            "task did not exit after owner dropped its Arc"
        );
        assert!(weak.upgrade().is_none(), "state should be fully dropped");
    }
}
