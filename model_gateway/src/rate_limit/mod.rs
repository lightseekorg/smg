//! Per-tenant LLM token/request rate limiting.
//!
//! This is the policy layer only: config schema, validation, and
//! compilation into an immutable, cheap-to-look-up `CompiledPolicySet`.
//! Nothing here does any rate-limit accounting or enforcement yet — the
//! reserve/settle engine, in-memory backend, and CLI/startup wiring land in
//! a follow-up change. This module is not yet reachable from any request
//! path.

mod config;
mod policy;

pub use config::{
    ModelMatcherSpec, ModelRuleSpec, RateLimitConfigError, RateLimitYaml, TenantPolicySpec,
};
pub use policy::{CompiledPolicySet, ScopeLimits};
