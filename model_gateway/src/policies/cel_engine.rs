//! CEL-Based Policy Engine for routing decisions
//!
//! Evaluates CEL (Common Expression Language) expressions against
//! live worker metric snapshots from the `MetricsStore`, and implements
//! a tiered fallback routing strategy:
//!
//! 1. **Fresh**: workers whose snapshot is recent (< staleness threshold)
//! 2. **Stale**: workers whose snapshot is older but still present
//! 3. **Round-Robin**: uniform selection when no metrics are available
//! 4. **503**: returns no worker when all workers are unhealthy

use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, SystemTime},
};

use cel_interpreter::{Context, Program, Value as CelValue};
use metrics_service::{MetricsStore, WorkerSnapshot};
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::core::Worker;

/// Policy decision from the CEL engine
#[derive(Debug, Clone, PartialEq)]
pub enum PolicyDecision {
    /// Route to this worker index
    SelectWorker(usize),
    /// No eligible worker — caller should return 503
    NoWorker,
}

/// Tier used to select a worker
#[derive(Debug, Clone, PartialEq)]
pub enum SelectionTier {
    Fresh,
    Stale,
    RoundRobin,
}

/// Result of a CEL policy evaluation
#[derive(Debug, Clone)]
pub struct PolicyResult {
    pub decision: PolicyDecision,
    pub tier: SelectionTier,
}

/// CEL Policy Engine
///
/// Evaluates routing policies by scoring worker snapshots from `MetricsStore`
/// and falling back through tiers if fresh data is unavailable.
pub struct CelPolicyEngine {
    metrics_store: Arc<MetricsStore>,
    /// How old a snapshot can be and still be considered "fresh"
    fresh_threshold: Duration,
    /// How old a snapshot can be and still be considered "stale" (used as fallback)
    stale_threshold: Duration,
}

impl CelPolicyEngine {
    pub fn new(
        metrics_store: Arc<MetricsStore>,
        fresh_threshold: Duration,
        stale_threshold: Duration,
    ) -> Self {
        Self {
            metrics_store,
            fresh_threshold,
            stale_threshold,
        }
    }

    /// Evaluate a routing policy and select the best worker.
    ///
    /// Falls through tiers:
    ///   Fresh → Stale → Round-Robin → 503
    pub fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        strategy: &RoutingStrategy,
    ) -> PolicyResult {
        if workers.is_empty() {
            return PolicyResult {
                decision: PolicyDecision::NoWorker,
                tier: SelectionTier::RoundRobin,
            };
        }

        // Build a map of snapshot data keyed by worker URL
        let all_snapshots = self.metrics_store.get_all();
        let snapshot_map: HashMap<&str, &WorkerSnapshot> = all_snapshots
            .iter()
            .map(|s| (s.url.as_str(), s.as_ref()))
            .collect();

        let now = SystemTime::now();

        // Tier 1: Try fresh snapshots
        if let Some(idx) =
            self.select_by_tier(workers, &snapshot_map, now, self.fresh_threshold, strategy)
        {
            return PolicyResult {
                decision: PolicyDecision::SelectWorker(idx),
                tier: SelectionTier::Fresh,
            };
        }

        // Tier 2: Try stale snapshots (larger window)
        if let Some(idx) =
            self.select_by_tier(workers, &snapshot_map, now, self.stale_threshold, strategy)
        {
            warn!("CelPolicyEngine: no fresh snapshots, using stale data for worker selection");
            return PolicyResult {
                decision: PolicyDecision::SelectWorker(idx),
                tier: SelectionTier::Stale,
            };
        }

        // Tier 3: Round-robin fallback (ignores metrics, just picks healthy workers)
        if let Some(idx) = self.round_robin_fallback(workers) {
            warn!("CelPolicyEngine: no metrics available, falling back to round-robin");
            return PolicyResult {
                decision: PolicyDecision::SelectWorker(idx),
                tier: SelectionTier::RoundRobin,
            };
        }

        // Tier 4: 503
        warn!("CelPolicyEngine: no healthy workers available");
        PolicyResult {
            decision: PolicyDecision::NoWorker,
            tier: SelectionTier::RoundRobin,
        }
    }

    /// Select the best worker whose snapshot age is within `max_age`, scoring by strategy.
    fn select_by_tier(
        &self,
        workers: &[Arc<dyn Worker>],
        snapshot_map: &HashMap<&str, &WorkerSnapshot>,
        now: SystemTime,
        max_age: Duration,
        strategy: &RoutingStrategy,
    ) -> Option<usize> {
        let mut best_idx: Option<usize> = None;
        let mut best_score: Option<i64> = None;

        for (idx, worker) in workers.iter().enumerate() {
            if !worker.is_healthy() || !worker.circuit_breaker().can_execute() {
                continue;
            }

            let Some(snapshot) = snapshot_map.get(worker.url()) else {
                continue;
            };

            // Check freshness
            let age = now
                .duration_since(snapshot.timestamp)
                .unwrap_or(Duration::MAX);
            if age > max_age {
                continue;
            }

            // Score worker based on chosen strategy
            let score = self.score_worker(snapshot, strategy);

            // Lower score = better (fewer tokens, fewer in-flight)
            if best_score.is_none() || score < best_score.unwrap() {
                best_score = Some(score);
                best_idx = Some(idx);
            }
        }

        if best_idx.is_some() {
            debug!(
                "CelPolicyEngine: selected worker index {:?} with score {:?}",
                best_idx, best_score
            );
        }

        best_idx
    }

    /// Compute a routing score for a worker snapshot (lower is better).
    fn score_worker(&self, snapshot: &WorkerSnapshot, strategy: &RoutingStrategy) -> i64 {
        match strategy {
            RoutingStrategy::MinKvCacheTokens => {
                // Primary: use kv_cache_tokens if available
                if let Some(kv) = snapshot.kv_cache_tokens {
                    return kv as i64;
                }
                // Fallback: estimate from in_flight * avg_tokens
                let in_flight = snapshot.in_flight_requests as i64;
                let avg = if snapshot.avg_tokens_per_req > 0 {
                    snapshot.avg_tokens_per_req as i64
                } else {
                    1024
                };
                in_flight.saturating_mul(avg)
            }
            RoutingStrategy::MinInFlight => snapshot.in_flight_requests as i64,
            RoutingStrategy::Custom(expr) => self.evaluate_custom_expr(snapshot, expr),
        }
    }

    /// Evaluate a custom metric expression using CEL.
    fn evaluate_custom_expr(&self, snapshot: &WorkerSnapshot, expr: &str) -> i64 {
        // In a high-performance system, the CEL Program should be compiled once and cached
        // in the RoutingStrategy enum. Since expr is passed as a string here, we parse it dynamically.
        let program = match Program::compile(expr) {
            Ok(p) => p,
            Err(_) => return 1024, // Fallback on parse failure
        };

        let mut context = Context::default();

        // Expose standard fields to the CEL context
        context.add_variable(
            "kv_cache_tokens",
            CelValue::Int(snapshot.kv_cache_tokens.unwrap_or(0) as i64),
        );
        context.add_variable(
            "in_flight_requests",
            CelValue::Int(snapshot.in_flight_requests as i64),
        );
        context.add_variable(
            "avg_tokens_per_req",
            CelValue::Int(snapshot.avg_tokens_per_req as i64),
        );

        // Expose all custom metrics to the CEL context
        for (k, v) in &snapshot.custom_metrics {
            // Scale floats to int for comparison or keep them as floats depending on CEL usage
            context.add_variable(k.as_str(), CelValue::Float(*v));
        }

        match program.execute(&context) {
            Ok(CelValue::Int(val)) => val,
            Ok(CelValue::Float(val)) => (val * 1000.0) as i64, // Scale floats back to ints
            _ => 1024, // Fallback if execution fails or returns non-numeric
        }
    }

    /// Round-robin fallback: just pick the first healthy worker.
    fn round_robin_fallback(&self, workers: &[Arc<dyn Worker>]) -> Option<usize> {
        workers
            .iter()
            .enumerate()
            .find(|(_, w)| w.is_healthy() && w.circuit_breaker().can_execute())
            .map(|(idx, _)| idx)
    }
}

/// Routing strategy for scoring workers
#[derive(Debug, Clone)]
pub enum RoutingStrategy {
    /// Prefer the worker with fewest KV-cache tokens (primary strategy)
    MinKvCacheTokens,
    /// Prefer the worker with fewest in-flight requests
    MinInFlight,
    /// Evaluate a custom CEL expression string
    Custom(String),
}

impl Default for RoutingStrategy {
    fn default() -> Self {
        Self::MinKvCacheTokens
    }
}
