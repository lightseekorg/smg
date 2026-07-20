use std::{sync::Arc, time::Duration};

use tokio::sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError};

use super::{Worker, WorkerLoadGuard};
use crate::observability::metrics::Metrics;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillAdmissionRejection {
    QueueFull,
    QueueTimeout,
    Closed,
}

#[derive(Debug)]
struct PrefillAdmissionState {
    slots: Arc<Semaphore>,
    queue_slots: Arc<Semaphore>,
    #[cfg(test)]
    queue_capacity: usize,
}

/// Gateway-wide admission gate for the short-lived prefill phase of PD requests.
#[derive(Clone, Debug)]
pub struct PrefillAdmission {
    state: Arc<PrefillAdmissionState>,
    queue_timeout: Duration,
}

impl PrefillAdmission {
    pub fn new(max_concurrent: usize, queue_size: usize, queue_timeout: Duration) -> Self {
        debug_assert!(max_concurrent > 0);
        Self {
            state: Arc::new(PrefillAdmissionState {
                slots: Arc::new(Semaphore::new(max_concurrent)),
                queue_slots: Arc::new(Semaphore::new(queue_size)),
                #[cfg(test)]
                queue_capacity: queue_size,
            }),
            queue_timeout,
        }
    }

    pub async fn admit(&self) -> Result<PrefillPermit, PrefillAdmissionRejection> {
        match Arc::clone(&self.state.slots).try_acquire_owned() {
            Ok(permit) => return Ok(PrefillPermit::new(permit)),
            Err(TryAcquireError::Closed) => {
                Metrics::record_prefill_admission_rejection("closed");
                return Err(PrefillAdmissionRejection::Closed);
            }
            Err(TryAcquireError::NoPermits) => {}
        }

        let queue_permit = match Arc::clone(&self.state.queue_slots).try_acquire_owned() {
            Ok(permit) => permit,
            Err(TryAcquireError::Closed) => {
                Metrics::record_prefill_admission_rejection("closed");
                return Err(PrefillAdmissionRejection::Closed);
            }
            Err(TryAcquireError::NoPermits) => {
                // A slot may have been released after the fast path failed.
                // Retry it before rejecting solely because the wait queue is full.
                match Arc::clone(&self.state.slots).try_acquire_owned() {
                    Ok(permit) => {
                        return Ok(PrefillPermit::new(permit));
                    }
                    Err(TryAcquireError::Closed) => {
                        Metrics::record_prefill_admission_rejection("closed");
                        return Err(PrefillAdmissionRejection::Closed);
                    }
                    Err(TryAcquireError::NoPermits) => {
                        Metrics::record_prefill_admission_rejection("queue_full");
                        return Err(PrefillAdmissionRejection::QueueFull);
                    }
                }
            }
        };
        let queued = QueuedPrefill::new(queue_permit);
        let wait_start = std::time::Instant::now();

        let result = tokio::time::timeout(
            self.queue_timeout,
            Arc::clone(&self.state.slots).acquire_owned(),
        )
        .await;
        Metrics::record_prefill_admission_wait(wait_start.elapsed());
        drop(queued);

        match result {
            Ok(Ok(permit)) => Ok(PrefillPermit::new(permit)),
            Ok(Err(_)) => {
                Metrics::record_prefill_admission_rejection("closed");
                Err(PrefillAdmissionRejection::Closed)
            }
            Err(_) => {
                Metrics::record_prefill_admission_rejection("queue_timeout");
                Err(PrefillAdmissionRejection::QueueTimeout)
            }
        }
    }

    #[cfg(test)]
    fn available_permits(&self) -> usize {
        self.state.slots.available_permits()
    }

    #[cfg(test)]
    fn queued_requests(&self) -> usize {
        self.state.queue_capacity - self.state.queue_slots.available_permits()
    }
}

#[derive(Debug)]
pub struct PrefillPermit {
    _permit: OwnedSemaphorePermit,
}

/// Resources whose lifetime is exactly one prefill phase.
pub struct PrefillPhaseGuard {
    _load: WorkerLoadGuard,
    _permit: Option<PrefillPermit>,
}

impl PrefillPhaseGuard {
    pub fn new(
        worker: Arc<dyn Worker>,
        headers: Option<&http::HeaderMap>,
        permit: Option<PrefillPermit>,
    ) -> Self {
        Self {
            _load: WorkerLoadGuard::new(worker, headers),
            _permit: permit,
        }
    }
}

impl PrefillPermit {
    fn new(permit: OwnedSemaphorePermit) -> Self {
        Metrics::increment_prefill_admission_inflight();
        Self { _permit: permit }
    }
}

impl Drop for PrefillPermit {
    fn drop(&mut self) {
        Metrics::decrement_prefill_admission_inflight();
    }
}

struct QueuedPrefill {
    _permit: OwnedSemaphorePermit,
}

impl QueuedPrefill {
    fn new(permit: OwnedSemaphorePermit) -> Self {
        Metrics::increment_prefill_admission_queued();
        Self { _permit: permit }
    }
}

impl Drop for QueuedPrefill {
    fn drop(&mut self) {
        Metrics::decrement_prefill_admission_queued();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker::{BasicWorkerBuilder, WorkerType};

    async fn wait_for_queued(gate: &PrefillAdmission, expected: usize) {
        tokio::time::timeout(Duration::from_secs(1), async {
            while gate.queued_requests() != expected {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn permit_is_released_on_drop() {
        let gate = PrefillAdmission::new(1, 0, Duration::from_secs(1));
        let permit = gate.admit().await.unwrap();
        assert_eq!(gate.available_permits(), 0);
        drop(permit);
        assert_eq!(gate.available_permits(), 1);
    }

    #[tokio::test]
    async fn rejects_when_queue_is_disabled() {
        let gate = PrefillAdmission::new(1, 0, Duration::from_secs(1));
        let _permit = gate.admit().await.unwrap();
        assert!(matches!(
            gate.admit().await,
            Err(PrefillAdmissionRejection::QueueFull)
        ));
    }

    #[tokio::test]
    async fn rejects_when_queue_is_full() {
        let gate = PrefillAdmission::new(1, 1, Duration::from_secs(1));
        let permit = gate.admit().await.unwrap();
        let waiter = tokio::spawn({
            let gate = gate.clone();
            async move { gate.admit().await }
        });
        wait_for_queued(&gate, 1).await;

        assert!(matches!(
            gate.admit().await,
            Err(PrefillAdmissionRejection::QueueFull)
        ));

        waiter.abort();
        assert!(waiter.await.unwrap_err().is_cancelled());
        drop(permit);
    }

    #[tokio::test(start_paused = true)]
    async fn queued_request_times_out() {
        let gate = PrefillAdmission::new(1, 1, Duration::from_secs(5));
        let _permit = gate.admit().await.unwrap();
        let waiter = tokio::spawn({
            let gate = gate.clone();
            async move { gate.admit().await }
        });

        tokio::time::advance(Duration::from_secs(5)).await;
        assert!(matches!(
            waiter.await.unwrap(),
            Err(PrefillAdmissionRejection::QueueTimeout)
        ));
    }

    #[tokio::test]
    async fn queued_request_acquires_released_slot() {
        let gate = PrefillAdmission::new(1, 1, Duration::from_secs(1));
        let permit = gate.admit().await.unwrap();
        let waiter = tokio::spawn({
            let gate = gate.clone();
            async move { gate.admit().await }
        });

        tokio::task::yield_now().await;
        drop(permit);
        let next = waiter.await.unwrap().unwrap();
        assert_eq!(gate.available_permits(), 0);
        drop(next);
        assert_eq!(gate.available_permits(), 1);
    }

    #[tokio::test]
    async fn queued_requests_are_admitted_in_fifo_order() {
        let gate = PrefillAdmission::new(1, 2, Duration::from_secs(1));
        let permit = gate.admit().await.unwrap();
        let (acquired_tx, mut acquired_rx) = tokio::sync::mpsc::unbounded_channel();
        let (release_first_tx, release_first_rx) = tokio::sync::oneshot::channel();

        let first = tokio::spawn({
            let gate = gate.clone();
            let acquired_tx = acquired_tx.clone();
            async move {
                let _permit = gate.admit().await.unwrap();
                acquired_tx.send(1).unwrap();
                let _ = release_first_rx.await;
            }
        });
        wait_for_queued(&gate, 1).await;

        let second = tokio::spawn({
            let gate = gate.clone();
            async move {
                let _permit = gate.admit().await.unwrap();
                acquired_tx.send(2).unwrap();
            }
        });
        wait_for_queued(&gate, 2).await;

        drop(permit);
        assert_eq!(acquired_rx.recv().await, Some(1));
        assert_eq!(gate.queued_requests(), 1);
        release_first_tx.send(()).unwrap();
        assert_eq!(acquired_rx.recv().await, Some(2));

        first.await.unwrap();
        second.await.unwrap();
        assert_eq!(gate.available_permits(), 1);
    }

    #[tokio::test]
    async fn cancelled_waiter_releases_queue_capacity() {
        let gate = PrefillAdmission::new(1, 1, Duration::from_secs(1));
        let permit = gate.admit().await.unwrap();
        let cancelled = tokio::spawn({
            let gate = gate.clone();
            async move { gate.admit().await }
        });
        wait_for_queued(&gate, 1).await;
        cancelled.abort();
        assert!(cancelled.await.unwrap_err().is_cancelled());
        assert_eq!(gate.queued_requests(), 0);

        let next = tokio::spawn({
            let gate = gate.clone();
            async move { gate.admit().await }
        });
        tokio::task::yield_now().await;
        drop(permit);

        let next_permit = next.await.unwrap().unwrap();
        drop(next_permit);
        assert_eq!(gate.available_permits(), 1);
    }

    #[tokio::test]
    async fn phase_guard_releases_worker_load_and_admission_slot_together() {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://prefill")
                .worker_type(WorkerType::Prefill)
                .build(),
        );
        let gate = PrefillAdmission::new(1, 0, Duration::from_secs(1));
        let permit = gate.admit().await.unwrap();
        let guard = PrefillPhaseGuard::new(worker.clone(), None, Some(permit));

        assert_eq!(worker.load(), 1);
        assert_eq!(gate.available_permits(), 0);
        drop(guard);
        assert_eq!(worker.load(), 0);
        assert_eq!(gate.available_permits(), 1);
    }
}
