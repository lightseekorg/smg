use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use openai_protocol::worker::WorkerLoadResponse;
use tracing::debug;

use super::{get_healthy_worker_indices, LoadBalancingPolicy, SelectWorkerInfo};
use crate::worker::Worker;

/// Default KV-pressure weight (request-equivalents per unit of M/M/1 congestion).
pub const DEFAULT_LAMBDA: f64 = 1.5;

#[derive(Debug)]
pub struct LeastLoadPolicy {
    /// Cached load reports from the worker monitor (keyed by worker URL).
    cached_loads: RwLock<HashMap<String, WorkerLoadResponse>>,
    /// KV-pressure weight.
    lambda: f64,
}

impl LeastLoadPolicy {
    pub fn new() -> Self {
        Self::with_lambda(DEFAULT_LAMBDA)
    }

    pub fn with_lambda(lambda: f64) -> Self {
        Self {
            cached_loads: RwLock::new(HashMap::new()),
            lambda: if lambda.is_finite() && lambda >= 0.0 {
                lambda
            } else {
                DEFAULT_LAMBDA
            },
        }
    }

    /// Least-load score for a worker (lower is better).
    fn score(
        &self,
        worker: &Arc<dyn Worker>,
        loads: Option<&HashMap<String, WorkerLoadResponse>>,
    ) -> f64 {
        let in_flight = worker.load() as f64;
        // KV-cache utilization from the latest load report; absent -> 0 (no barrier).
        let k = loads
            .and_then(|m| m.get(worker.url()))
            .map(|l| l.effective_token_usage().clamp(0.0, 0.999))
            .unwrap_or(0.0);
        in_flight + self.lambda * k / (1.0 - k)
    }
}

impl LoadBalancingPolicy for LeastLoadPolicy {
    fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        _info: &SelectWorkerInfo,
    ) -> Option<usize> {
        let healthy = get_healthy_worker_indices(workers);
        if healthy.is_empty() {
            return None;
        }
        if healthy.len() == 1 {
            return Some(healthy[0]);
        }

        let guard = self.cached_loads.read().ok();
        let loads = guard.as_deref();

        let mut best = healthy[0];
        let mut best_score = self.score(&workers[best], loads);
        for &idx in &healthy[1..] {
            let s = self.score(&workers[idx], loads);
            if s < best_score {
                best = idx;
                best_score = s;
            }
        }

        debug!(
            "least_load selected {} (score {:.4}, in_flight {})",
            workers[best].url(),
            best_score,
            workers[best].load()
        );
        workers[best].increment_processed();
        Some(best)
    }

    fn name(&self) -> &'static str {
        "least_load"
    }

    fn update_loads(&self, loads: &HashMap<String, WorkerLoadResponse>) {
        if let Ok(mut cached) = self.cached_loads.write() {
            cached.extend(loads.iter().map(|(k, v)| (k.clone(), v.clone())));
        }
    }

    fn remove_worker(&self, url: &str) {
        if let Ok(mut cached) = self.cached_loads.write() {
            cached.remove(url);
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Default for LeastLoadPolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::worker::{HealthCheckConfig, SchedulerLoadSnapshot};

    use super::*;
    use crate::worker::{BasicWorkerBuilder, WorkerType};

    fn no_health_check() -> HealthCheckConfig {
        HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        }
    }

    fn make_load(token_usage: f64) -> WorkerLoadResponse {
        WorkerLoadResponse {
            timestamp: String::new(),
            dp_rank_count: 1,
            loads: vec![SchedulerLoadSnapshot {
                dp_rank: 0,
                num_running_reqs: 0,
                num_waiting_reqs: 0,
                num_total_reqs: 0,
                num_used_tokens: 0,
                max_total_num_tokens: 0,
                token_usage,
                gen_throughput: 0.0,
                cache_hit_rate: 0.0,
                utilization: 0.0,
                max_running_requests: 0,
            }],
        }
    }

    fn mk(url: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .health_config(no_health_check())
                .build(),
        )
    }

    #[test]
    fn picks_lowest_in_flight() {
        let policy = LeastLoadPolicy::new();
        let a = mk("http://a:8000");
        let b = mk("http://b:8000");
        for _ in 0..5 {
            a.increment_load();
        }
        let workers = vec![a, b];
        // No load reports -> pure in-flight; b (0) beats a (5).
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(1)
        );
    }

    #[test]
    fn kv_barrier_avoids_full_worker() {
        let policy = LeastLoadPolicy::with_lambda(2.0);
        let a = mk("http://a:8000"); // idle but KV-full
        let b = mk("http://b:8000"); // 1 in-flight but KV empty
        b.increment_load();
        let workers = vec![a, b];
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), make_load(0.95)); // barrier 2*0.95/0.05 = 38
        loads.insert("http://b:8000".to_string(), make_load(0.0));
        policy.update_loads(&loads);
        // a: 0 + 38 = 38 ; b: 1 + 0 = 1 -> pick b.
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(1)
        );
    }

    #[test]
    fn single_worker_always_selected() {
        let policy = LeastLoadPolicy::new();
        let workers = vec![mk("http://a:8000")];
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(0)
        );
    }

    #[test]
    fn remove_worker_prunes_cached_load() {
        let policy = LeastLoadPolicy::new();
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), make_load(0.5));
        loads.insert("http://b:8000".to_string(), make_load(0.3));
        policy.update_loads(&loads);
        assert_eq!(policy.cached_loads.read().unwrap().len(), 2);

        // Removing a worker drops only its entry (no unbounded growth on churn).
        policy.remove_worker("http://a:8000");
        let cached = policy.cached_loads.read().unwrap();
        assert_eq!(cached.len(), 1);
        assert!(!cached.contains_key("http://a:8000"));
        assert!(cached.contains_key("http://b:8000"));
    }
}
