//! Aggregate backend capacity tracking.
//!
//! See `.claude/priority-scheduling/01-worker-capacity-design.md` for the
//! full design rationale.

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
}
