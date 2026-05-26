//! Scheduler config: on-disk YAML shape + runtime form.

use std::{collections::HashMap, time::Duration};

use serde::{Deserialize, Serialize};

use super::Class;
use crate::tenant::TenantKey;

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
    /// Built-in defaults. These are what every class gets when no YAML
    /// file supplies an override.
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
/// [`SchedulerSettings::from_cli_and_yaml`] fills in built-in defaults
/// for any class that wasn't overridden.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PrioritySchedulerYaml {
    #[serde(default)]
    pub classes: HashMap<Class, ClassConfig>,
    #[serde(default)]
    pub tenant_policies: HashMap<String, TenantPolicyConfig>,
}

/// Per-field validation failures discovered while assembling
/// [`SchedulerSettings`]. Capacity-vs-reserved validation is not in
/// scope here — the scheduler performs it at construction time, once
/// the live backend capacity is known.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum SettingsValidationError {
    #[error("class {class:?}: queue_size_per_slot must be a finite, non-negative number")]
    InvalidMultiplier { class: Class },
    #[error("class {class:?}: queue_timeout_secs must be > 0")]
    ZeroQueueTimeout { class: Class },
    #[error("class {class:?}: starvation_threshold_secs must be > 0")]
    ZeroStarvationThreshold { class: Class },
}

/// Runtime scheduler configuration assembled from CLI flags + the
/// optional YAML file + built-in defaults. Built once at startup and
/// then read-only.
///
/// Capacity is not stored here; the scheduler reads it live from
/// [`crate::worker::WorkerCapacity`] and reacts to changes via its
/// watch channel. Settings stay fixed.
#[derive(Debug, Clone)]
pub struct SchedulerSettings {
    /// Master switch. When `false`, the priority scheduler is not
    /// constructed and the gateway falls back to its legacy
    /// concurrency-limit admission path.
    pub enabled: bool,
    /// Tenant policy applied when a tenant is not in `tenant_policies`.
    pub default_max_class: Class,
    /// Per-class config, indexed by `class as usize`. Always populated
    /// for all four classes — YAML overrides land on top of
    /// [`ClassConfig::default_for`].
    classes: [ClassConfig; 4],
    /// Per-tenant clamp lookup. Keys come from
    /// [`crate::tenant::RouteRequestMeta::tenant_key`].
    pub tenant_policies: HashMap<TenantKey, TenantPolicyConfig>,
    /// Cap on the number of tenants emitted as labels for
    /// `scheduler_tenant_*` gauges. Everything past the cap is
    /// bucketed under `tenant="other"`.
    pub tenant_metric_top_n: u32,
}

impl SchedulerSettings {
    /// Indexed accessor for the per-class config.
    pub fn class_config(&self, class: Class) -> &ClassConfig {
        &self.classes[class as usize]
    }

    /// Assemble settings from CLI flags + optional YAML, validating
    /// per-field invariants. Capacity-vs-reserved validation is the
    /// scheduler's job at construction time.
    ///
    /// Merge order: built-in defaults → YAML overrides per class. CLI
    /// flags supply the top-level fields (`enabled`, `default_max_class`,
    /// `tenant_metric_top_n`) since they aren't representable in the YAML.
    pub fn from_cli_and_yaml(
        enabled: bool,
        default_max_class: Class,
        tenant_metric_top_n: u32,
        yaml: Option<&PrioritySchedulerYaml>,
    ) -> Result<Self, SettingsValidationError> {
        let mut classes: [ClassConfig; 4] = [
            ClassConfig::default_for(Class::Bulk),
            ClassConfig::default_for(Class::Default),
            ClassConfig::default_for(Class::Interactive),
            ClassConfig::default_for(Class::System),
        ];

        if let Some(yaml) = yaml {
            for (class, override_cfg) in &yaml.classes {
                classes[*class as usize] = *override_cfg;
            }
        }

        for class in Class::ALL {
            let cfg = &classes[class as usize];
            if !cfg.queue_size_per_slot.is_finite() || cfg.queue_size_per_slot < 0.0 {
                return Err(SettingsValidationError::InvalidMultiplier { class });
            }
            if cfg.queue_timeout_secs == 0 {
                return Err(SettingsValidationError::ZeroQueueTimeout { class });
            }
            if cfg.starvation_threshold_secs == 0 {
                return Err(SettingsValidationError::ZeroStarvationThreshold { class });
            }
        }

        let tenant_policies = yaml
            .map(|y| {
                y.tenant_policies
                    .iter()
                    .map(|(k, v)| (TenantKey::new(k), *v))
                    .collect()
            })
            .unwrap_or_default();

        Ok(Self {
            enabled,
            default_max_class,
            classes,
            tenant_policies,
            tenant_metric_top_n,
        })
    }
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

    // ── SchedulerSettings::from_cli_and_yaml ─────────────────────────

    use crate::tenant::TenantKey;

    #[test]
    fn test_settings_no_yaml_uses_builtin_defaults() {
        let s = SchedulerSettings::from_cli_and_yaml(false, Class::Default, 32, None).unwrap();
        assert!(!s.enabled);
        assert_eq!(s.default_max_class, Class::Default);
        assert_eq!(s.tenant_metric_top_n, 32);
        for class in Class::ALL {
            assert_eq!(s.class_config(class), &ClassConfig::default_for(class));
        }
        assert!(s.tenant_policies.is_empty());
    }

    #[test]
    fn test_settings_yaml_partial_override_merges_with_defaults() {
        let mut classes = HashMap::new();
        let mut interactive = ClassConfig::default_for(Class::Interactive);
        interactive.reserved = 64;
        classes.insert(Class::Interactive, interactive);
        let yaml = PrioritySchedulerYaml {
            classes,
            tenant_policies: Default::default(),
        };
        let s =
            SchedulerSettings::from_cli_and_yaml(true, Class::Default, 32, Some(&yaml)).unwrap();
        assert_eq!(s.class_config(Class::Interactive).reserved, 64);
        // Bulk untouched — still equal to built-in default.
        assert_eq!(
            s.class_config(Class::Bulk),
            &ClassConfig::default_for(Class::Bulk)
        );
    }

    #[test]
    fn test_settings_yaml_tenant_policy_propagates() {
        let mut tenant_policies = HashMap::new();
        tenant_policies.insert(
            "auth:acme".to_string(),
            TenantPolicyConfig {
                max_class: Class::Interactive,
            },
        );
        let yaml = PrioritySchedulerYaml {
            classes: Default::default(),
            tenant_policies,
        };
        let s =
            SchedulerSettings::from_cli_and_yaml(true, Class::Default, 32, Some(&yaml)).unwrap();
        assert_eq!(s.tenant_policies.len(), 1);
        let key = TenantKey::new("auth:acme");
        assert_eq!(s.tenant_policies[&key].max_class, Class::Interactive);
    }

    #[test]
    fn test_settings_rejects_negative_multiplier() {
        let mut classes = HashMap::new();
        let mut interactive = ClassConfig::default_for(Class::Interactive);
        interactive.queue_size_per_slot = -1.0;
        classes.insert(Class::Interactive, interactive);
        let yaml = PrioritySchedulerYaml {
            classes,
            tenant_policies: Default::default(),
        };
        let err = SchedulerSettings::from_cli_and_yaml(true, Class::Default, 32, Some(&yaml))
            .unwrap_err();
        assert!(matches!(
            err,
            SettingsValidationError::InvalidMultiplier {
                class: Class::Interactive
            }
        ));
    }

    #[test]
    fn test_settings_rejects_zero_queue_timeout() {
        let mut classes = HashMap::new();
        let mut bulk = ClassConfig::default_for(Class::Bulk);
        bulk.queue_timeout_secs = 0;
        classes.insert(Class::Bulk, bulk);
        let yaml = PrioritySchedulerYaml {
            classes,
            tenant_policies: Default::default(),
        };
        let err = SchedulerSettings::from_cli_and_yaml(true, Class::Default, 32, Some(&yaml))
            .unwrap_err();
        assert!(matches!(
            err,
            SettingsValidationError::ZeroQueueTimeout { class: Class::Bulk }
        ));
    }

    #[test]
    fn test_settings_rejects_nan_multiplier_from_yaml() {
        // serde_yaml parses `.nan` into f32::NAN. Without an explicit
        // is_finite() check, the comparison `< 0.0` is false for NaN and the
        // bad value slips through to garbage queue-limit math later.
        let yaml = r"
classes:
  interactive:
    reserved: 128
    queue_size: 256
    queue_size_per_slot: .nan
    queue_timeout_secs: 30
    starvation_threshold_secs: 5
    can_preempt: true
";
        let parsed: PrioritySchedulerYaml = serde_yaml::from_str(yaml).unwrap();
        assert!(
            parsed.classes[&Class::Interactive]
                .queue_size_per_slot
                .is_nan(),
            "serde_yaml deserialized .nan as NaN"
        );
        let err = SchedulerSettings::from_cli_and_yaml(true, Class::Default, 32, Some(&parsed))
            .unwrap_err();
        assert!(matches!(
            err,
            SettingsValidationError::InvalidMultiplier {
                class: Class::Interactive
            }
        ));
    }

    #[test]
    fn test_settings_rejects_infinity_multiplier_from_yaml() {
        let yaml = r"
classes:
  interactive:
    reserved: 128
    queue_size: 256
    queue_size_per_slot: .inf
    queue_timeout_secs: 30
    starvation_threshold_secs: 5
    can_preempt: true
";
        let parsed: PrioritySchedulerYaml = serde_yaml::from_str(yaml).unwrap();
        assert!(
            parsed.classes[&Class::Interactive]
                .queue_size_per_slot
                .is_infinite(),
            "serde_yaml deserialized .inf as +Infinity"
        );
        let err = SchedulerSettings::from_cli_and_yaml(true, Class::Default, 32, Some(&parsed))
            .unwrap_err();
        assert!(matches!(
            err,
            SettingsValidationError::InvalidMultiplier {
                class: Class::Interactive
            }
        ));
    }

    #[test]
    fn test_settings_rejects_zero_starvation_threshold() {
        let mut classes = HashMap::new();
        let mut bulk = ClassConfig::default_for(Class::Bulk);
        bulk.starvation_threshold_secs = 0;
        classes.insert(Class::Bulk, bulk);
        let yaml = PrioritySchedulerYaml {
            classes,
            tenant_policies: Default::default(),
        };
        let err = SchedulerSettings::from_cli_and_yaml(true, Class::Default, 32, Some(&yaml))
            .unwrap_err();
        assert!(matches!(
            err,
            SettingsValidationError::ZeroStarvationThreshold { class: Class::Bulk }
        ));
    }
}
