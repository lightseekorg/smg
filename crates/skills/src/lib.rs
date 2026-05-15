//! Skills domain types and service scaffolding.
//!
//! This crate intentionally starts small. The first integration step only
//! establishes the stable crate boundary that later PRs will fill in with
//! parsing, storage, CRUD, and execution logic.

pub mod api;
pub mod config;
pub mod memory;
pub mod resolution;
pub mod storage;
pub mod tool_protocol;
pub mod tool_schemas;
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
pub use resolution::{
    resolve_messages_skill_manifest, resolve_responses_skill_manifest, ResolvedSkillManifest,
    ResolvedSkillRef, SkillResolutionError,
};
pub use storage::{
    BundleTokenStore, ContinuationCookieStore, SkillMetadataStore, SkillsStoreError,
    SkillsStoreResult, TenantAliasStore,
};
pub use tool_protocol::{
    messages_skill_tools, response_skill_tools, validate_messages_reserved_skill_tool_names,
    validate_responses_reserved_skill_tool_names, ReservedSkillToolNameError,
};
pub use tool_schemas::{
    execute_skill_tool, is_reserved_skill_tool_name, is_skill_executor_configured,
    list_skill_files_tool, read_only_skill_tools, read_skill_file_tool, read_skill_tool,
    should_register_execute_skill, skill_tools, SkillToolDefinition, EXECUTE_SKILL_TOOL_NAME,
    LIST_SKILL_FILES_TOOL_NAME, READ_SKILL_FILE_TOOL_NAME, READ_SKILL_TOOL_NAME,
    RESERVED_SKILL_TOOL_NAMES,
};
pub use types::{
    BundleTokenClaim, ContinuationCookieClaim, NormalizedSkillBundle, NormalizedSkillFile,
    ParsedSkillBundle, PinnedSkillVersion, SkillDependencyTool, SkillFileRecord,
    SkillInterfaceMetadata, SkillParseWarning, SkillParseWarningKind, SkillPolicyMetadata,
    SkillRecord, SkillSidecarDependencies, SkillVersionRecord, SkillVersionSelector,
    TenantAliasRecord,
};
pub use validation::{
    is_code_file_path, normalize_skill_bundle_zip, parse_skill_bundle, SkillBundleArchiveError,
    SkillParseError,
};
