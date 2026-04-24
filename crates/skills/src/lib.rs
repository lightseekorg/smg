//! Skills domain types and service scaffolding.
//!
//! This crate intentionally starts small. The first integration step only
//! establishes the stable crate boundary that later PRs will fill in with
//! parsing, storage, CRUD, and execution logic.

pub mod api;
pub mod config;
pub mod memory;
pub mod storage;
pub mod types;
pub mod validation;

pub use api::{
    CreateSkillRequest, CreateSkillVersionRequest, DeletedSkillVersionResult, SkillCreateResult,
    SkillService, SkillServiceError, SkillServiceMode, SkillUpload, UpdateSkillRequest,
    UpdateSkillVersionRequest, UploadedSkillFile,
};
pub use config::{
    SkillUploadLimits, SkillsAdminConfig, SkillsAdminOperation, SkillsBlobStoreBackend,
    SkillsBlobStoreConfig, SkillsBudgetLimit, SkillsCacheConfig, SkillsConfig,
    SkillsDependenciesConfig, SkillsExecutionAsyncMode, SkillsExecutionConfig,
    SkillsExecutionModeOverrides, SkillsInstructionBudgetConfig, SkillsMissingMcpPolicy,
    SkillsRateLimitsConfig, SkillsResolutionMode, SkillsRetentionConfig, SkillsRetentionMode,
    SkillsTenancyConfig, SkillsToolLoopConfig, SkillsZdrConfig,
};
pub use memory::InMemorySkillStore;
pub use storage::{
    BundleTokenStore, ContinuationCookieStore, SkillMetadataStore, SkillsStoreError,
    SkillsStoreResult, TenantAliasStore,
};
pub use types::{
    BundleTokenClaim, ContinuationCookieClaim, NormalizedSkillBundle, NormalizedSkillFile,
    ParsedSkillBundle, SkillDependencyTool, SkillFileRecord, SkillInterfaceMetadata,
    SkillParseWarning, SkillParseWarningKind, SkillPolicyMetadata, SkillRecord,
    SkillSidecarDependencies, SkillVersionRecord, TenantAliasRecord,
};
pub use validation::{
    is_code_file_path, normalize_skill_bundle_zip, parse_skill_bundle, SkillBundleArchiveError,
    SkillParseError,
};
