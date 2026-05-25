//! Priority-aware admission scheduler.
//!
//! See `.claude/docs/scheduler/02-priority-scheduler-design.md` for the
//! full design rationale and `02-priority-scheduler-plan.md` for the
//! implementation sequencing.

pub mod class;
pub mod config;
pub mod policy;

pub use class::{Class, PRIORITY_HEADER};
pub use config::{
    ClassConfig, ClassRuntimeConfig, PrioritySchedulerYaml, SchedulerSettings,
    SettingsValidationError, TenantPolicyConfig,
};
pub use policy::{StaticTenantPolicyResolver, TenantPolicy, TenantPolicyResolver};
