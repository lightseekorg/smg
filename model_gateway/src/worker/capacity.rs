//! Aggregate backend capacity tracking.
//!
//! See `.claude/priority-scheduling/01-worker-capacity-design.md` for the
//! full design rationale.

use std::sync::Arc;

use super::Worker;

/// Which precedence tier produced the most recently computed capacity value.
/// Exposed as a Prometheus gauge so operators can debug capacity decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CapacitySource {
    /// Tier 4 — no workers known yet (or unhealthy fleet). Using legacy
    /// `--max-concurrent-requests` value as a fallback.
    LegacyFallback = 0,
    /// Tier 1 — operator pinned via `--worker-capacity-override`.
    Override = 1,
    /// Tier 2 — every healthy worker reported `max_running_requests`. Sum
    /// across the fleet.
    WorkerReported = 2,
    /// Tier 3 — at least one worker did not report; reporters contribute
    /// their reported value, non-reporters contribute `slots_per_worker`.
    /// Also used when no worker reports but workers exist (pure tier 3).
    Mixed = 3,
}

impl CapacitySource {
    pub fn from_u8(raw: u8) -> Self {
        match raw {
            1 => Self::Override,
            2 => Self::WorkerReported,
            3 => Self::Mixed,
            _ => Self::LegacyFallback,
        }
    }
}

/// Configuration for the WorkerCapacity tracker.
///
/// Built once at gateway startup from CLI flags. None of the fields
/// change at runtime (use a future `ArcSwap<Settings>` if hot-reload
/// is ever needed).
#[derive(Debug, Clone)]
pub struct CapacityTrackerSettings {
    /// Tier 1 override. `Some(n)` with `n > 0` pins capacity to `n`
    /// regardless of the fleet. `None` (or originally-zero) means
    /// "derive from workers."
    pub override_capacity: Option<u16>,
    /// Per-worker slot count used for tier 3 (mixed) and pure tier-3
    /// (workers known, none report).
    pub slots_per_worker: u16,
    /// Tier 4 fallback when no healthy workers are known.
    /// Typically sourced from the existing `--max-concurrent-requests` flag.
    pub legacy_max_concurrent_requests: u16,
}

impl CapacityTrackerSettings {
    /// Constructor that maps an `i32` override (0 = derive) to the
    /// internal `Option<u16>` representation. Matches the shape of
    /// the `--worker-capacity-override` CLI flag.
    pub fn with_override(raw: i32) -> Self {
        Self {
            override_capacity: u16::try_from(raw).ok().filter(|n| *n > 0),
            ..Self::default()
        }
    }
}

impl Default for CapacityTrackerSettings {
    fn default() -> Self {
        Self {
            override_capacity: None,
            slots_per_worker: 64,
            legacy_max_concurrent_requests: 1024,
        }
    }
}

/// Compute capacity and source tier from settings + a snapshot of healthy workers.
///
/// Pure function: no I/O, no atomics. Easily unit-testable.
/// Callers are responsible for filtering `workers` to only healthy ones.
pub(super) fn recompute(
    settings: &CapacityTrackerSettings,
    workers: &[Arc<dyn Worker>],
) -> (u16, CapacitySource) {
    // Tier 1: operator override always wins.
    if let Some(n) = settings.override_capacity {
        if n > 0 {
            return (n, CapacitySource::Override);
        }
    }

    let total_workers = workers.len();
    if total_workers == 0 {
        return (
            settings.legacy_max_concurrent_requests,
            CapacitySource::LegacyFallback,
        );
    }

    let mut sum_reported: u32 = 0;
    let mut reporters: usize = 0;
    let mut non_reporters: usize = 0;
    for w in workers {
        match w.max_running_requests() {
            Some(n) => {
                sum_reported = sum_reported.saturating_add(u32::from(n));
                reporters += 1;
            }
            None => non_reporters += 1,
        }
    }

    if non_reporters == 0 {
        // Tier 2: every worker reported.
        let capped = sum_reported.min(u32::from(u16::MAX)) as u16;
        return (capped, CapacitySource::WorkerReported);
    }

    // Tier 3 (Mixed): reporters contribute reported, non-reporters contribute slots_per_worker.
    // Also covers "zero reporters" since non_reporters > 0 here.
    let _ = reporters; // reporters count is implied; non_reporters drives the formula
    let from_non_reporters =
        (non_reporters as u32).saturating_mul(u32::from(settings.slots_per_worker));
    let total = sum_reported.saturating_add(from_non_reporters);
    let capped = total.min(u32::from(u16::MAX)) as u16;
    (capped, CapacitySource::Mixed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capacity_source_round_trips_through_u8() {
        for src in [
            CapacitySource::LegacyFallback,
            CapacitySource::Override,
            CapacitySource::WorkerReported,
            CapacitySource::Mixed,
        ] {
            let raw = src as u8;
            assert_eq!(CapacitySource::from_u8(raw), src);
        }
    }

    #[test]
    fn test_capacity_source_unknown_u8_decodes_to_legacy() {
        assert_eq!(CapacitySource::from_u8(99), CapacitySource::LegacyFallback);
    }

    #[test]
    fn test_settings_default_has_sensible_values() {
        let s = CapacityTrackerSettings::default();
        assert_eq!(s.override_capacity, None);
        assert_eq!(s.slots_per_worker, 64);
        assert_eq!(s.legacy_max_concurrent_requests, 1024);
    }

    #[test]
    fn test_settings_override_zero_treated_as_none() {
        let s = CapacityTrackerSettings::with_override(0);
        assert!(s.override_capacity.is_none(), "0 is the sentinel for 'derive'");
    }

    #[test]
    fn test_settings_override_nonzero_preserved() {
        let s = CapacityTrackerSettings::with_override(2048);
        assert_eq!(s.override_capacity, Some(2048));
    }

    use std::collections::HashMap;

    use crate::worker::BasicWorkerBuilder;

    fn worker_with_capacity(
        url: &str,
        reported: Option<u16>,
    ) -> std::sync::Arc<dyn crate::worker::Worker> {
        let mut labels = HashMap::new();
        if let Some(n) = reported {
            labels.insert("max_running_requests".to_string(), n.to_string());
        }
        std::sync::Arc::new(BasicWorkerBuilder::new(url).labels(labels).build())
    }

    #[test]
    fn test_recompute_tier1_override_wins_even_with_workers() {
        let settings = CapacityTrackerSettings {
            override_capacity: Some(512),
            ..Default::default()
        };
        let workers = vec![worker_with_capacity("http://w1", Some(256))];
        let (capacity, source) = recompute(&settings, &workers);
        assert_eq!(capacity, 512);
        assert_eq!(source, CapacitySource::Override);
    }

    #[test]
    fn test_recompute_tier4_no_workers_uses_legacy_fallback() {
        let settings = CapacityTrackerSettings {
            legacy_max_concurrent_requests: 256,
            ..Default::default()
        };
        let (capacity, source) = recompute(&settings, &[]);
        assert_eq!(capacity, 256);
        assert_eq!(source, CapacitySource::LegacyFallback);
    }

    #[test]
    fn test_recompute_tier2_all_workers_report() {
        let settings = CapacityTrackerSettings::default();
        let workers = vec![
            worker_with_capacity("http://w1", Some(256)),
            worker_with_capacity("http://w2", Some(128)),
        ];
        let (capacity, source) = recompute(&settings, &workers);
        assert_eq!(capacity, 384);
        assert_eq!(source, CapacitySource::WorkerReported);
    }

    #[test]
    fn test_recompute_tier3_mixed_reporters_and_non_reporters() {
        let settings = CapacityTrackerSettings {
            slots_per_worker: 64,
            ..Default::default()
        };
        let workers = vec![
            worker_with_capacity("http://w1", Some(256)),
            worker_with_capacity("http://w2", None),
        ];
        let (capacity, source) = recompute(&settings, &workers);
        assert_eq!(capacity, 256 + 64);
        assert_eq!(source, CapacitySource::Mixed);
    }

    #[test]
    fn test_recompute_tier3_zero_reporters_uses_worker_count() {
        let settings = CapacityTrackerSettings {
            slots_per_worker: 64,
            ..Default::default()
        };
        let workers = vec![
            worker_with_capacity("http://w1", None),
            worker_with_capacity("http://w2", None),
            worker_with_capacity("http://w3", None),
        ];
        let (capacity, source) = recompute(&settings, &workers);
        assert_eq!(capacity, 3 * 64);
        assert_eq!(source, CapacitySource::Mixed);
    }

    #[test]
    fn test_recompute_saturates_at_u16_max() {
        let settings = CapacityTrackerSettings::default();
        // 1000 workers × 100 slots = 100_000, exceeds u16::MAX (65_535).
        let workers: Vec<_> = (0..1000)
            .map(|i| worker_with_capacity(&format!("http://w{i}"), Some(100)))
            .collect();
        let (capacity, source) = recompute(&settings, &workers);
        assert_eq!(capacity, u16::MAX);
        assert_eq!(source, CapacitySource::WorkerReported);
    }
}
