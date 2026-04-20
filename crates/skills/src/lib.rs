//! Skills domain types and service scaffolding.
//!
//! This crate intentionally starts small. The first integration step only
//! establishes the stable crate boundary that later PRs will fill in with
//! parsing, storage, CRUD, and execution logic.

pub mod api;
pub mod config;
pub mod types;
pub mod validation;

pub use api::{SkillService, SkillServiceMode};
pub use config::{
    SkillsAdminConfig, SkillsAdminOperation, SkillsBlobStoreBackend, SkillsBlobStoreConfig,
    SkillsBudgetLimit, SkillsCacheConfig, SkillsConfig, SkillsDependenciesConfig,
    SkillsExecutionAsyncMode, SkillsExecutionConfig, SkillsExecutionModeOverrides,
    SkillsInstructionBudgetConfig, SkillsMissingMcpPolicy, SkillsRateLimitsConfig,
    SkillsResolutionMode, SkillsRetentionConfig, SkillsRetentionMode, SkillsTenancyConfig,
    SkillsToolLoopConfig, SkillsZdrConfig,
};
pub use types::{
    ParsedSkillBundle, SkillDependencyTool, SkillFileRecord, SkillInterfaceMetadata,
    SkillParseWarning, SkillParseWarningKind, SkillPolicyMetadata, SkillRecord,
    SkillSidecarDependencies, SkillVersionRecord,
};
pub use validation::{parse_skill_bundle, SkillParseError};
