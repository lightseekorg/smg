//! Startup wiring for the priority scheduler: builds the admission mode
//! the route layer branches on.

use std::sync::Arc;

use tracing::{error, info};

use super::{
    Class, PriorityScheduler, SchedulerSettings, StaticTenantPolicyResolver, TenantPolicyResolver,
};
use crate::{
    config::types::RouterConfig,
    middleware::token_bucket::TokenBucket,
    worker::{CapacityTrackerSettings, WorkerCapacity, WorkerRegistry},
};

/// State handed to `priority_admission_middleware` via `from_fn_with_state`.
/// Cheap to clone (all `Arc`).
pub struct SchedulerState {
    pub scheduler: Arc<PriorityScheduler>,
    pub resolver: Arc<dyn TenantPolicyResolver>,
    /// Per-second RPS sibling check, run before admission. Set only when an
    /// explicit `rate_limit_tokens_per_second` is configured; the bucket's
    /// concurrency-cap role is owned by the scheduler, so we must not consult
    /// it as a concurrency limiter (that would double-limit). `None` =
    /// no RPS limit.
    pub rate_limiter: Option<Arc<TokenBucket>>,
}

/// Which admission path the protected routes use. Chosen once at startup.
#[derive(Clone)]
pub enum AdmissionMode {
    /// Legacy `concurrency_limit_middleware` (default; zero behavior change).
    Legacy,
    /// Priority scheduler enabled.
    Priority(Arc<SchedulerState>),
}

impl AdmissionMode {
    /// Build the admission mode from runtime config.
    ///
    /// When `priority_scheduler_enabled` is false, returns `Legacy` without
    /// constructing anything. When true, constructs `WorkerCapacity` over
    /// the worker fleet, builds the scheduler against its current capacity,
    /// spawns the dispatcher on its watch channel, and returns
    /// `Priority(..)`.
    ///
    /// On any startup error (bad YAML, reservations exceed capacity), logs
    /// at ERROR and falls back to `Legacy` rather than aborting the whole
    /// gateway — a misconfigured scheduler must not take the data plane down.
    pub fn from_config(
        rc: &RouterConfig,
        registry: Arc<WorkerRegistry>,
        rate_limiter: Option<Arc<TokenBucket>>,
    ) -> Self {
        if !rc.priority_scheduler_enabled {
            return Self::Legacy;
        }
        match Self::try_build_priority(rc, registry, rate_limiter) {
            Ok(mode) => {
                info!("priority scheduler enabled");
                mode
            }
            Err(e) => {
                error!(
                    error = %e,
                    "priority scheduler failed to start; falling back to legacy admission"
                );
                Self::Legacy
            }
        }
    }

    fn try_build_priority(
        rc: &RouterConfig,
        registry: Arc<WorkerRegistry>,
        rate_limiter: Option<Arc<TokenBucket>>,
    ) -> Result<Self, String> {
        // Tier-4 fallback for WorkerCapacity comes from the legacy
        // --max-concurrent-requests (clamped to u16; <=0 means "disabled",
        // for which we keep the tracker default).
        let cap_settings = CapacityTrackerSettings {
            legacy_max_concurrent_requests: u16::try_from(rc.max_concurrent_requests)
                .unwrap_or_else(|_| {
                    CapacityTrackerSettings::default().legacy_max_concurrent_requests
                }),
            ..CapacityTrackerSettings::default()
        };
        let worker_capacity = WorkerCapacity::spawn(registry, cap_settings);

        let default_max_class = Class::parse_header(&rc.priority_scheduler_default_max_class);
        let yaml = load_yaml(rc.priority_scheduler_config.as_deref())?;
        let settings = SchedulerSettings::from_cli_and_yaml(
            true,
            default_max_class,
            rc.priority_scheduler_tenant_metric_top_n,
            yaml.as_ref(),
        )
        .map_err(|e| e.to_string())?;

        let scheduler = PriorityScheduler::new(&settings, worker_capacity.current())
            .map_err(|e| e.to_string())?;
        scheduler.spawn_dispatcher(worker_capacity.watch());

        let resolver: Arc<dyn TenantPolicyResolver> =
            Arc::new(StaticTenantPolicyResolver::from_settings(&settings));

        // The scheduler owns concurrency, so the shared bucket only survives
        // as an RPS check when an explicit per-second limit is configured.
        let rate_limiter = match rc.rate_limit_tokens_per_second {
            Some(rps) if rps > 0 => rate_limiter,
            _ => None,
        };

        Ok(Self::Priority(Arc::new(SchedulerState {
            scheduler,
            resolver,
            rate_limiter,
        })))
    }
}

/// Load + parse the optional priority-scheduler YAML file.
fn load_yaml(path: Option<&str>) -> Result<Option<super::PrioritySchedulerYaml>, String> {
    let Some(path) = path else {
        return Ok(None);
    };
    let contents = std::fs::read_to_string(path).map_err(|e| format!("reading {path}: {e}"))?;
    let parsed = serde_yaml::from_str(&contents).map_err(|e| format!("parsing {path}: {e}"))?;
    Ok(Some(parsed))
}
