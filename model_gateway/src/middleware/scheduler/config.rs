//! Scheduler config: on-disk YAML shape + runtime form.

use std::{collections::HashMap, time::Duration};

use serde::{Deserialize, Serialize};

use super::Class;

/// Per-class configuration as it appears in the optional YAML file.
///
/// Lives separately from [`ClassRuntimeConfig`] because the YAML form
/// uses primitive types friendly to serde and human editing, while the
/// runtime form pre-converts seconds into [`Duration`] so hot paths
/// don't repeat the conversion.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ClassConfig {
    /// Slots reserved for this class. Higher-class reservations are
    /// honored by lower-class admissions via the packed-CAS slot
    /// accounting in [`super::scheduler`].
    pub reserved: u16,
    /// Absolute floor on the per-class queue depth.
    pub queue_size: u32,
    /// Optional multiplier: effective limit =
    /// `max(queue_size, ceil(queue_size_per_slot * capacity))`.
    /// `0.0` disables the multiplier (use the absolute floor only).
    pub queue_size_per_slot: f32,
    /// How long a queued waiter waits before the admission middleware
    /// returns 408. Seconds at rest; converted to [`Duration`] in
    /// [`ClassRuntimeConfig`].
    pub queue_timeout_secs: u64,
    /// Head-of-queue age past which the dispatcher promotes a waiter
    /// out of normal priority order to avoid starvation. Seconds at
    /// rest; converted to [`Duration`] in [`ClassRuntimeConfig`].
    pub starvation_threshold_secs: u64,
    /// Whether admissions in this class are allowed to preempt a
    /// lower-class inflight request that has not yet emitted its first
    /// byte. Higher classes default to `true`; lower classes default
    /// to `false`.
    pub can_preempt: bool,
}

impl ClassConfig {
    /// Built-in defaults per `02-priority-scheduler-design.md` §3.
    /// These are what every class gets when no YAML file supplies an
    /// override.
    pub fn default_for(class: Class) -> Self {
        match class {
            Class::System => Self {
                reserved: 32,
                queue_size: 64,
                queue_size_per_slot: 0.0,
                queue_timeout_secs: 30,
                starvation_threshold_secs: 5,
                can_preempt: true,
            },
            Class::Interactive => Self {
                reserved: 128,
                queue_size: 256,
                queue_size_per_slot: 0.25,
                queue_timeout_secs: 30,
                starvation_threshold_secs: 5,
                can_preempt: true,
            },
            Class::Default => Self {
                reserved: 0,
                queue_size: 512,
                queue_size_per_slot: 0.5,
                queue_timeout_secs: 60,
                starvation_threshold_secs: 30,
                can_preempt: false,
            },
            Class::Bulk => Self {
                reserved: 0,
                queue_size: 1024,
                queue_size_per_slot: 1.0,
                queue_timeout_secs: 300,
                starvation_threshold_secs: 120,
                can_preempt: false,
            },
        }
    }
}

/// Runtime view of [`ClassConfig`] — only the fields the dispatcher
/// reads on its hot path, with seconds pre-converted to [`Duration`].
/// `reserved`, `queue_size`, and `queue_size_per_slot` live elsewhere
/// (the packed-CAS array and the per-class queue impl), so they don't
/// appear here.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClassRuntimeConfig {
    pub queue_timeout: Duration,
    pub starvation_threshold: Duration,
    pub can_preempt: bool,
}

impl ClassRuntimeConfig {
    pub fn from_class_config(cfg: &ClassConfig) -> Self {
        Self {
            queue_timeout: Duration::from_secs(cfg.queue_timeout_secs),
            starvation_threshold: Duration::from_secs(cfg.starvation_threshold_secs),
            can_preempt: cfg.can_preempt,
        }
    }
}

/// Per-tenant policy entry in the YAML file.
///
/// Future fields (`weight`, `slot_quota`, `rps_cap`) are additive: adding
/// them is non-breaking because the trait
/// [`super::policy::TenantPolicyResolver`] returns the whole struct.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TenantPolicyConfig {
    pub max_class: Class,
}

/// Optional YAML config loaded via `--priority-scheduler-config <path>`.
///
/// Both maps are absent-as-empty: an empty document parses to
/// `PrioritySchedulerYaml::default()`, and downstream
/// [`super::SchedulerSettings::from_cli_and_yaml`] fills in built-in
/// defaults for any class that wasn't overridden.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PrioritySchedulerYaml {
    #[serde(default)]
    pub classes: HashMap<Class, ClassConfig>,
    #[serde(default)]
    pub tenant_policies: HashMap<String, TenantPolicyConfig>,
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::middleware::scheduler::Class;

    #[test]
    fn test_default_for_system() {
        let cfg = ClassConfig::default_for(Class::System);
        assert_eq!(cfg.reserved, 32);
        assert_eq!(cfg.queue_size, 64);
        assert_eq!(cfg.queue_size_per_slot, 0.0);
        assert_eq!(cfg.queue_timeout_secs, 30);
        assert_eq!(cfg.starvation_threshold_secs, 5);
        assert!(cfg.can_preempt);
    }

    #[test]
    fn test_default_for_interactive() {
        let cfg = ClassConfig::default_for(Class::Interactive);
        assert_eq!(cfg.reserved, 128);
        assert_eq!(cfg.queue_size, 256);
        assert_eq!(cfg.queue_size_per_slot, 0.25);
        assert_eq!(cfg.queue_timeout_secs, 30);
        assert_eq!(cfg.starvation_threshold_secs, 5);
        assert!(cfg.can_preempt);
    }

    #[test]
    fn test_default_for_default() {
        let cfg = ClassConfig::default_for(Class::Default);
        assert_eq!(cfg.reserved, 0);
        assert_eq!(cfg.queue_size, 512);
        assert_eq!(cfg.queue_size_per_slot, 0.5);
        assert_eq!(cfg.queue_timeout_secs, 60);
        assert_eq!(cfg.starvation_threshold_secs, 30);
        assert!(!cfg.can_preempt);
    }

    #[test]
    fn test_default_for_bulk() {
        let cfg = ClassConfig::default_for(Class::Bulk);
        assert_eq!(cfg.reserved, 0);
        assert_eq!(cfg.queue_size, 1024);
        assert_eq!(cfg.queue_size_per_slot, 1.0);
        assert_eq!(cfg.queue_timeout_secs, 300);
        assert_eq!(cfg.starvation_threshold_secs, 120);
        assert!(!cfg.can_preempt);
    }

    #[test]
    fn test_runtime_config_converts_seconds_to_duration() {
        let cfg = ClassConfig::default_for(Class::Default);
        let runtime = ClassRuntimeConfig::from_class_config(&cfg);
        assert_eq!(runtime.queue_timeout, Duration::from_secs(60));
        assert_eq!(runtime.starvation_threshold, Duration::from_secs(30));
        assert!(!runtime.can_preempt);
    }

    #[test]
    fn test_runtime_config_preserves_can_preempt_flag() {
        let interactive =
            ClassRuntimeConfig::from_class_config(&ClassConfig::default_for(Class::Interactive));
        assert!(interactive.can_preempt);
        let bulk = ClassRuntimeConfig::from_class_config(&ClassConfig::default_for(Class::Bulk));
        assert!(!bulk.can_preempt);
    }

    // ── PrioritySchedulerYaml serde ───────────────────────────────────

    #[test]
    fn test_yaml_empty_document_yields_default() {
        let parsed: PrioritySchedulerYaml = serde_yaml::from_str("").unwrap();
        assert!(parsed.classes.is_empty());
        assert!(parsed.tenant_policies.is_empty());
    }

    #[test]
    fn test_yaml_partial_class_override_round_trips() {
        let yaml = r"
classes:
  interactive:
    reserved: 200
    queue_size: 256
    queue_size_per_slot: 0.25
    queue_timeout_secs: 30
    starvation_threshold_secs: 5
    can_preempt: true
";
        let parsed: PrioritySchedulerYaml = serde_yaml::from_str(yaml).unwrap();
        let interactive = parsed
            .classes
            .get(&Class::Interactive)
            .expect("interactive present");
        assert_eq!(interactive.reserved, 200);
        // Only one class entry — others are absent (settings layer fills defaults).
        assert_eq!(parsed.classes.len(), 1);
        assert!(parsed.tenant_policies.is_empty());
    }

    #[test]
    fn test_yaml_tenant_policy_round_trips() {
        let yaml = r#"
tenant_policies:
  "auth:acme":
    max_class: interactive
  "auth:internal-cron":
    max_class: system
"#;
        let parsed: PrioritySchedulerYaml = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(parsed.tenant_policies.len(), 2);
        assert_eq!(
            parsed.tenant_policies["auth:acme"].max_class,
            Class::Interactive
        );
        assert_eq!(
            parsed.tenant_policies["auth:internal-cron"].max_class,
            Class::System
        );
    }

    #[test]
    fn test_yaml_unknown_class_value_is_serde_error() {
        let yaml = r#"
tenant_policies:
  "auth:acme":
    max_class: garbage
"#;
        let result: Result<PrioritySchedulerYaml, _> = serde_yaml::from_str(yaml);
        assert!(result.is_err(), "expected serde error for unknown class");
    }

    #[test]
    fn test_yaml_class_name_serializes_as_lowercase() {
        let mut classes = HashMap::new();
        classes.insert(Class::Bulk, ClassConfig::default_for(Class::Bulk));
        let yaml = PrioritySchedulerYaml {
            classes,
            tenant_policies: Default::default(),
        };
        let rendered = serde_yaml::to_string(&yaml).unwrap();
        assert!(
            rendered.contains("bulk:"),
            "class key should serialize as lowercase: {rendered}"
        );
    }
}
