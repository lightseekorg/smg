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
    any::Any,
    collections::HashMap,
    fmt,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, SystemTime},
};

use cel_interpreter::{Context, Program, Value as CelValue};
use metrics_service::{MetricsStore, WorkerSnapshot};
use tracing::{debug, warn};

use super::{LoadBalancingPolicy, SelectWorkerInfo};
use crate::core::Worker;

// Public types

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

/// Routing strategy for scoring workers
#[derive(Clone, Default)]
pub enum RoutingStrategy {
    /// Prefer the worker with fewest KV-cache tokens (primary strategy)
    #[default]
    MinKvCacheTokens,
    /// Prefer the worker with fewest in-flight requests
    MinInFlight,
    /// Evaluate a pre-compiled CEL expression.
    ///
    /// Build with [`RoutingStrategy::custom`] to compile once and reuse.
    Custom {
        /// Original expression string (for Debug / introspection)
        expr: String,
        /// Pre-compiled CEL program — zero allocation on the hot path
        program: Arc<Program>,
    },
}

impl RoutingStrategy {
    /// Compile a CEL expression and return a `Custom` strategy.
    ///
    /// Returns an error string if the expression fails to parse/compile.
    pub fn custom(expr: impl Into<String>) -> Result<Self, String> {
        let expr = expr.into();
        let program =
            Program::compile(&expr).map_err(|e| format!("CEL compile error for '{expr}': {e}"))?;
        Ok(Self::Custom {
            expr,
            program: Arc::new(program),
        })
    }
}

impl fmt::Debug for RoutingStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MinKvCacheTokens => write!(f, "MinKvCacheTokens"),
            Self::MinInFlight => write!(f, "MinInFlight"),
            Self::Custom { expr, .. } => f.debug_struct("Custom").field("expr", expr).finish(),
        }
    }
}

/// CEL Policy Engine
///
/// Evaluates routing policies by scoring worker snapshots from `MetricsStore`
/// and falling back through tiers if fresh data is unavailable.
///
/// Also implements [`LoadBalancingPolicy`] so it can be used as a drop-in
/// policy under the `"metrics_driven"` name.
pub struct CelPolicyEngine {
    metrics_store: Arc<MetricsStore>,
    /// How old a snapshot can be and still be considered "fresh"
    fresh_threshold: Duration,
    /// How old a snapshot can be and still be considered "stale" (used as fallback)
    stale_threshold: Duration,
    /// Routing strategy to use when this engine acts as a `LoadBalancingPolicy`
    strategy: RoutingStrategy,
    /// Round-robin counter for the fallback tier
    rr_counter: AtomicUsize,
}

impl fmt::Debug for CelPolicyEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CelPolicyEngine")
            .field("fresh_threshold", &self.fresh_threshold)
            .field("stale_threshold", &self.stale_threshold)
            .field("strategy", &self.strategy)
            .finish_non_exhaustive()
    }
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
            strategy: RoutingStrategy::default(),
            rr_counter: AtomicUsize::new(0),
        }
    }

    /// Create with an explicit routing strategy (used when wiring as a
    /// `LoadBalancingPolicy`).
    pub fn with_strategy(
        metrics_store: Arc<MetricsStore>,
        fresh_threshold: Duration,
        stale_threshold: Duration,
        strategy: RoutingStrategy,
    ) -> Self {
        Self {
            metrics_store,
            fresh_threshold,
            stale_threshold,
            strategy,
            rr_counter: AtomicUsize::new(0),
        }
    }

    /// Evaluate a routing policy and select the best worker.
    ///
    /// Falls through tiers:
    ///   Fresh → Stale → Round-Robin → 503
    pub fn select_worker_with_strategy(
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
            Self::select_by_tier(workers, &snapshot_map, now, self.fresh_threshold, strategy)
        {
            return PolicyResult {
                decision: PolicyDecision::SelectWorker(idx),
                tier: SelectionTier::Fresh,
            };
        }

        // Tier 2: Try stale snapshots (larger window)
        if let Some(idx) =
            Self::select_by_tier(workers, &snapshot_map, now, self.stale_threshold, strategy)
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
            let score = Self::score_worker(snapshot, strategy);

            // Lower score = better (fewer tokens, fewer in-flight)
            if best_score.is_none_or(|best| score < best) {
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
    fn score_worker(snapshot: &WorkerSnapshot, strategy: &RoutingStrategy) -> i64 {
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
            RoutingStrategy::Custom { program, .. } => {
                Self::evaluate_compiled_expr(snapshot, program)
            }
        }
    }

    /// Evaluate a **pre-compiled** CEL program against a snapshot.
    ///
    /// The `Program` is already compiled — this is O(1) setup + expression
    /// evaluation cost, no allocation for parsing.
    fn evaluate_compiled_expr(snapshot: &WorkerSnapshot, program: &Program) -> i64 {
        let mut context = Context::default();

        // Expose standard fields to the CEL context
        let _ = context.add_variable(
            "kv_cache_tokens",
            CelValue::Int(snapshot.kv_cache_tokens.unwrap_or(0) as i64),
        );
        let _ = context.add_variable(
            "in_flight_requests",
            CelValue::Int(snapshot.in_flight_requests as i64),
        );
        let _ = context.add_variable(
            "avg_tokens_per_req",
            CelValue::Int(snapshot.avg_tokens_per_req as i64),
        );

        // Expose all custom metrics to the CEL context
        for (k, v) in &snapshot.custom_metrics {
            let _ = context.add_variable(k.as_str(), CelValue::Float(*v));
        }

        match program.execute(&context) {
            Ok(CelValue::Int(val)) => val,
            Ok(CelValue::Float(val)) => (val * 1000.0) as i64, // Scale floats back to ints
            _ => 1024, // Fallback if execution fails or returns non-numeric
        }
    }

    /// Round-robin fallback: picks the next healthy worker in ring order.
    fn round_robin_fallback(&self, workers: &[Arc<dyn Worker>]) -> Option<usize> {
        let healthy: Vec<usize> = workers
            .iter()
            .enumerate()
            .filter(|(_, w)| w.is_healthy() && w.circuit_breaker().can_execute())
            .map(|(idx, _)| idx)
            .collect();

        if healthy.is_empty() {
            return None;
        }
        let pos = self.rr_counter.fetch_add(1, Ordering::Relaxed) % healthy.len();
        Some(healthy[pos])
    }
}

// LoadBalancingPolicy impl — makes CelPolicyEngine a plug-in policy

impl LoadBalancingPolicy for CelPolicyEngine {
    fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        _info: &SelectWorkerInfo,
    ) -> Option<usize> {
        let result = self.select_worker_with_strategy(workers, &self.strategy);
        match result.decision {
            PolicyDecision::SelectWorker(idx) => Some(idx),
            PolicyDecision::NoWorker => None,
        }
    }

    fn name(&self) -> &'static str {
        "metrics_driven"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use metrics_service::{EventBus, MetricSource, MetricsStore, WorkerSnapshot};

    use super::*;

    // helpers

    fn make_store() -> Arc<MetricsStore> {
        let bus = Arc::new(EventBus::new(64));
        Arc::new(MetricsStore::new(bus, Duration::from_secs(60)))
    }

    fn make_engine(store: Arc<MetricsStore>) -> CelPolicyEngine {
        CelPolicyEngine::new(store, Duration::from_secs(10), Duration::from_secs(60))
    }

    /// Push a fresh snapshot into the store for `url`.
    fn push(store: &MetricsStore, url: &str, kv: isize, inflight: isize) {
        let mut snap = WorkerSnapshot::new(url.to_string(), MetricSource::Piggyback);
        snap.kv_cache_tokens = Some(kv);
        snap.in_flight_requests = inflight;
        store.update(snap);
    }

    // scoring tests (don't need real Worker objects)

    #[test]
    fn test_min_kv_cache_tokens_scoring() {
        let store = make_store();
        push(&store, "http://w1", 100, 5);
        push(&store, "http://w2", 500, 1);

        let _engine = make_engine(Arc::clone(&store));

        let snap1 = store.get("http://w1").unwrap();
        let snap2 = store.get("http://w2").unwrap();

        let score1 = CelPolicyEngine::score_worker(&snap1, &RoutingStrategy::MinKvCacheTokens);
        let score2 = CelPolicyEngine::score_worker(&snap2, &RoutingStrategy::MinKvCacheTokens);

        assert!(
            score1 < score2,
            "w1 (kv=100) should score lower than w2 (kv=500)"
        );
    }

    #[test]
    fn test_min_in_flight_scoring() {
        let store = make_store();
        push(&store, "http://w1", 50, 10);
        push(&store, "http://w2", 200, 2);

        let _engine = make_engine(Arc::clone(&store));

        let snap1 = store.get("http://w1").unwrap();
        let snap2 = store.get("http://w2").unwrap();

        let score1 = CelPolicyEngine::score_worker(&snap1, &RoutingStrategy::MinInFlight);
        let score2 = CelPolicyEngine::score_worker(&snap2, &RoutingStrategy::MinInFlight);

        // w2 has fewer in-flight (2 < 10) so it should score lower
        assert!(
            score2 < score1,
            "w2 (inflight=2) should score lower than w1 (inflight=10)"
        );
    }

    #[test]
    fn test_cel_strategy_compilation() {
        let strategy = RoutingStrategy::custom("in_flight_requests * 2");
        assert!(
            matches!(strategy, Ok(RoutingStrategy::Custom { .. })),
            "Valid CEL expr should compile successfully"
        );
    }

    #[test]
    fn test_invalid_cel_strategy_compilation() {
        let strategy = RoutingStrategy::custom("not.a.valid.cel.expr(((");
        assert!(strategy.is_err(), "Invalid CEL expr should return an error");
    }

    #[test]
    fn test_cel_scoring_with_compiled_program() {
        let store = make_store();
        push(&store, "http://w1", 0, 5);

        let _engine = make_engine(Arc::clone(&store));
        let strategy = RoutingStrategy::custom("in_flight_requests").expect("should compile");

        let snap = store.get("http://w1").unwrap();
        let score = CelPolicyEngine::score_worker(&snap, &strategy);
        assert_eq!(score, 5, "CEL 'in_flight_requests' should return 5");
    }

    #[test]
    fn test_routing_strategy_debug() {
        let s = RoutingStrategy::MinKvCacheTokens;
        assert_eq!(format!("{s:?}"), "MinKvCacheTokens");

        let s = RoutingStrategy::MinInFlight;
        assert_eq!(format!("{s:?}"), "MinInFlight");

        let s = RoutingStrategy::custom("kv_cache_tokens").unwrap();
        let dbg = format!("{s:?}");
        assert!(dbg.contains("Custom"), "Custom debug should mention Custom");
    }
}
