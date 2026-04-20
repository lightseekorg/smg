use std::{collections::BTreeMap, fmt};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use smg_blob_storage::BlobKey;

/// Top-level skill metadata stored in the control-plane database.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SkillRecord {
    pub tenant_id: String,
    pub skill_id: String,
    pub name: String,
    pub short_description: Option<String>,
    pub description: Option<String>,
    pub source: String,
    pub has_code_files: bool,
    pub latest_version: Option<String>,
    pub default_version: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Metadata for a single immutable skill version.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SkillVersionRecord {
    pub skill_id: String,
    pub version: String,
    pub version_number: u32,
    pub name: String,
    pub short_description: Option<String>,
    pub description: String,
    pub interface: Option<SkillInterfaceMetadata>,
    pub dependencies: Option<SkillSidecarDependencies>,
    pub policy: Option<SkillPolicyMetadata>,
    pub deprecated: bool,
    pub file_manifest: Vec<SkillFileRecord>,
    pub instruction_token_counts: BTreeMap<String, u32>,
    pub created_at: DateTime<Utc>,
}

/// File-level manifest entry stored alongside a normalized skill bundle.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SkillFileRecord {
    pub relative_path: String,
    pub media_type: Option<String>,
    pub size_bytes: u64,
    pub blob_key: Option<BlobKey>,
}

/// Tenant-alias mapping for request-time tenant resolution.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TenantAliasRecord {
    pub alias_tenant_id: String,
    pub canonical_tenant_id: String,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
}

/// Persisted bundle-token claims keyed by a deterministic secret hash.
#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BundleTokenClaim {
    pub token_hash: String,
    pub tenant_id: String,
    pub exec_id: String,
    pub skill_id: String,
    pub skill_version: String,
    pub created_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
}

/// Persisted continuation-cookie claims keyed by a deterministic secret hash.
#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ContinuationCookieClaim {
    pub cookie_hash: String,
    pub tenant_id: String,
    pub exec_id: String,
    pub request_id: String,
    pub created_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
}

/// Canonical in-memory representation of a validated skill bundle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizedSkillBundle {
    pub files: Vec<NormalizedSkillFile>,
    pub skill_md_path: String,
    pub openai_sidecar_path: Option<String>,
    pub has_code_files: bool,
}

impl fmt::Debug for BundleTokenClaim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BundleTokenClaim")
            .field("token_hash", &"<redacted>")
            .field("tenant_id", &self.tenant_id)
            .field("exec_id", &self.exec_id)
            .field("skill_id", &self.skill_id)
            .field("skill_version", &self.skill_version)
            .field("created_at", &self.created_at)
            .field("expires_at", &self.expires_at)
            .finish()
    }
}

impl fmt::Debug for ContinuationCookieClaim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ContinuationCookieClaim")
            .field("cookie_hash", &"<redacted>")
            .field("tenant_id", &self.tenant_id)
            .field("exec_id", &self.exec_id)
            .field("request_id", &self.request_id)
            .field("created_at", &self.created_at)
            .field("expires_at", &self.expires_at)
            .finish()
    }
}

impl NormalizedSkillBundle {
    /// Project the in-memory bundle into a stable file manifest.
    #[must_use]
    pub fn file_manifest(&self) -> Vec<SkillFileRecord> {
        self.files
            .iter()
            .map(|file| SkillFileRecord {
                relative_path: file.relative_path.clone(),
                media_type: None,
                size_bytes: file.size_bytes(),
                blob_key: None,
            })
            .collect()
    }
}

/// Canonical skill-bundle file with a skill-root-relative path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizedSkillFile {
    pub relative_path: String,
    pub contents: Vec<u8>,
}

impl NormalizedSkillFile {
    /// Return the canonical uncompressed size of this file in bytes.
    #[must_use]
    pub fn size_bytes(&self) -> u64 {
        self.contents.len() as u64
    }
}

/// Parsed `SKILL.md` plus any successfully recovered OpenAI sidecar metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ParsedSkillBundle {
    pub name: String,
    pub description: String,
    pub short_description: Option<String>,
    pub instructions_body: String,
    pub interface: Option<SkillInterfaceMetadata>,
    pub dependencies: Option<SkillSidecarDependencies>,
    pub policy: Option<SkillPolicyMetadata>,
    pub warnings: Vec<SkillParseWarning>,
}

/// Optional interface metadata sourced from `agents/openai.yaml`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct SkillInterfaceMetadata {
    pub display_name: Option<String>,
    pub short_description: Option<String>,
    pub icon_small: Option<String>,
    pub icon_large: Option<String>,
    pub brand_color: Option<String>,
    pub default_prompt: Option<String>,
}

impl SkillInterfaceMetadata {
    /// Return whether this interface block contains any usable fields.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.display_name.is_none()
            && self.short_description.is_none()
            && self.icon_small.is_none()
            && self.icon_large.is_none()
            && self.brand_color.is_none()
            && self.default_prompt.is_none()
    }
}

/// Optional dependency metadata sourced from `agents/openai.yaml`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct SkillSidecarDependencies {
    pub tools: Vec<SkillDependencyTool>,
}

impl SkillSidecarDependencies {
    /// Return whether the dependencies block contains any tool declarations.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

/// A single dependency tool declaration from `agents/openai.yaml`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SkillDependencyTool {
    pub tool_type: String,
    pub value: String,
    pub description: Option<String>,
    pub transport: Option<String>,
    pub command: Option<String>,
    pub url: Option<String>,
}

/// Optional invocation-policy metadata sourced from `agents/openai.yaml`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct SkillPolicyMetadata {
    pub allow_implicit_invocation: Option<bool>,
    pub products: Vec<String>,
}

impl SkillPolicyMetadata {
    /// Return whether the policy block contains any usable settings.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.allow_implicit_invocation.is_none() && self.products.is_empty()
    }
}

/// Warning produced while salvaging optional sidecar metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SkillParseWarning {
    pub kind: SkillParseWarningKind,
    pub path: String,
    pub message: String,
}

/// Kinds of non-fatal sidecar parsing warnings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SkillParseWarningKind {
    SidecarFileIgnored,
    SidecarFieldIgnored,
}
