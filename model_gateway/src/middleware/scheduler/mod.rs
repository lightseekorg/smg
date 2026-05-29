//! Priority-aware admission scheduler.

pub mod body;
pub mod class;
pub mod config;
pub mod engine;
pub mod inflight;
pub mod policy;
pub mod queue;
pub mod slots;

pub use body::SchedulerGuardBody;
pub use class::{Class, PRIORITY_HEADER};
pub use config::{
    ClassConfig, ClassRuntimeConfig, PrioritySchedulerYaml, SchedulerSettings,
    SettingsValidationError, TenantPolicyConfig,
};
pub use engine::{
    AdmitOutcome, PriorityScheduler, RejectionReason, SchedulerInitError, SchedulerPermit,
};
pub use policy::{StaticTenantPolicyResolver, TenantPolicy, TenantPolicyResolver};
