//! Prometheus metrics for the priority scheduler.
//!
//! Split to match design §9: this module holds the **operational** counters
//! and the queue-wait histogram (recorded on the admission / queue paths).
//! The point-in-time **capacity / autoscaling** gauges (inflight, queue
//! depth, utilization, per-tenant, …) are computed by the sampler task so
//! the hot admission path only does cheap counter increments.
//!
//! All names carry the `smg_` prefix to match the rest of the gateway's
//! metrics. Class/outcome labels are `&'static str` (no per-request
//! allocation).

use std::time::Duration;

use metrics::{counter, describe_counter, describe_histogram, histogram};

use super::Class;

const ADMIT_TOTAL: &str = "smg_scheduler_admit_total";
const QUEUE_WAIT_SECONDS: &str = "smg_scheduler_queue_wait_seconds";
const PREEMPTION_TOTAL: &str = "smg_scheduler_preemption_total";
const CLAMP_TOTAL: &str = "smg_scheduler_clamp_total";
const UNKNOWN_PRIORITY_TOTAL: &str = "smg_scheduler_unknown_priority_value_total";
const STARVATION_PROMOTION_TOTAL: &str = "smg_scheduler_starvation_promotion_total";

/// `outcome` label values for [`record_admit`].
pub mod outcome {
    /// Admitted (fast path or after queueing — not distinguished).
    pub const ADMITTED: &str = "admitted";
    /// Per-class queue was at its limit.
    pub const REJECTED_QUEUE_FULL: &str = "rejected_queue_full";
    /// Queued waiter aged past `queue_timeout`.
    pub const REJECTED_QUEUE_TIMEOUT: &str = "rejected_queue_timeout";
    /// The request was admitted but then preempted before producing a byte.
    pub const PREEMPTED: &str = "preempted";
    /// The caller's client disconnected before admission completed.
    pub const CLIENT_CANCELLED: &str = "client_cancelled";
}

/// Register descriptions. Called once from `observability::metrics::init_metrics`.
pub fn describe() {
    describe_counter!(
        ADMIT_TOTAL,
        "Priority-scheduler admission outcomes by class and outcome"
    );
    describe_histogram!(
        QUEUE_WAIT_SECONDS,
        "Time a request spent queued before admission, timeout, or cancel"
    );
    describe_counter!(
        PREEMPTION_TOTAL,
        "Successful preemptions by victim class and preempting class"
    );
    describe_counter!(
        CLAMP_TOTAL,
        "Requests whose priority was clamped below the requested class by tenant policy"
    );
    describe_counter!(
        UNKNOWN_PRIORITY_TOTAL,
        "Requests with an unrecognized priority header value (treated as default)"
    );
    describe_counter!(
        STARVATION_PROMOTION_TOTAL,
        "Queued waiters admitted via the starvation override path"
    );
}

/// Record the outcome of an admission attempt for `class`.
pub fn record_admit(class: Class, outcome: &'static str) {
    counter!(ADMIT_TOTAL, "class" => class.as_str(), "outcome" => outcome).increment(1);
}

/// Record the time a request waited in a class queue.
pub fn record_queue_wait(class: Class, wait: Duration) {
    histogram!(QUEUE_WAIT_SECONDS, "class" => class.as_str()).record(wait.as_secs_f64());
}

/// Record a successful preemption.
pub fn record_preemption(victim_class: Class, by_class: Class) {
    counter!(
        PREEMPTION_TOTAL,
        "victim_class" => victim_class.as_str(),
        "by_class" => by_class.as_str()
    )
    .increment(1);
}

/// Record a priority clamp (only when the effective class is below the
/// requested class).
pub fn record_clamp(requested: Class, effective: Class) {
    counter!(
        CLAMP_TOTAL,
        "requested_class" => requested.as_str(),
        "effective_class" => effective.as_str()
    )
    .increment(1);
}

/// Record an unrecognized priority header value.
pub fn record_unknown_priority() {
    counter!(UNKNOWN_PRIORITY_TOTAL).increment(1);
}

/// Record a starvation-override promotion.
pub fn record_starvation_promotion(class: Class) {
    counter!(STARVATION_PROMOTION_TOTAL, "class" => class.as_str()).increment(1);
}
