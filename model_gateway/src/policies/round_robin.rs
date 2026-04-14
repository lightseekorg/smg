//! Round-robin load balancing policy

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use super::{get_healthy_worker_indices, LoadBalancingPolicy, SelectWorkerInfo};
use crate::worker::Worker;

/// Round-robin selection policy
///
/// Selects workers in sequential order, cycling through all healthy workers.
#[derive(Debug, Default)]
pub struct RoundRobinPolicy {
    counter: AtomicUsize,
}

impl RoundRobinPolicy {
    pub fn new() -> Self {
        Self {
            counter: AtomicUsize::new(0),
        }
    }
}

impl LoadBalancingPolicy for RoundRobinPolicy {
    fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        _info: &SelectWorkerInfo,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);

        if healthy_indices.is_empty() {
            return None;
        }

        // Get and increment counter atomically
        let count = self.counter.fetch_add(1, Ordering::Relaxed);
        let selected_idx = count % healthy_indices.len();

        Some(healthy_indices[selected_idx])
    }

    fn name(&self) -> &'static str {
        "round_robin"
    }

    fn reset(&self) {
        self.counter.store(0, Ordering::Relaxed);
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::worker::{HealthCheckConfig, WorkerStatus};

    use super::*;
    use crate::worker::{BasicWorkerBuilder, WorkerType};

    fn no_health_check() -> HealthCheckConfig {
        HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        }
    }

    #[test]
    fn test_round_robin_selection() {
        let policy = RoundRobinPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w3:8000")
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ),
        ];

        // Should select workers in order: 0, 1, 2, 0, 1, 2, ...
        let info = SelectWorkerInfo::default();
        assert_eq!(policy.select_worker(&workers, &info), Some(0));
        assert_eq!(policy.select_worker(&workers, &info), Some(1));
        assert_eq!(policy.select_worker(&workers, &info), Some(2));
        assert_eq!(policy.select_worker(&workers, &info), Some(0));
        assert_eq!(policy.select_worker(&workers, &info), Some(1));
    }

    #[test]
    fn test_round_robin_with_unhealthy_workers() {
        let policy = RoundRobinPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w3:8000")
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ),
        ];

        // Mark middle worker as unhealthy
        workers[1].set_status(WorkerStatus::NotReady);

        // Should skip unhealthy worker: 0, 2, 0, 2, ...
        let info = SelectWorkerInfo::default();
        assert_eq!(policy.select_worker(&workers, &info), Some(0));
        assert_eq!(policy.select_worker(&workers, &info), Some(2));
        assert_eq!(policy.select_worker(&workers, &info), Some(0));
        assert_eq!(policy.select_worker(&workers, &info), Some(2));
    }

    #[test]
    fn test_round_robin_reset() {
        let policy = RoundRobinPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ),
        ];

        // Advance the counter
        let info = SelectWorkerInfo::default();
        assert_eq!(policy.select_worker(&workers, &info), Some(0));
        assert_eq!(policy.select_worker(&workers, &info), Some(1));

        // Reset should start from beginning
        policy.reset();
        assert_eq!(policy.select_worker(&workers, &info), Some(0));
    }
}
