use serde::{Deserialize, Serialize};

/// Top-level skill metadata stored in the control-plane database.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SkillRecord {
    pub tenant_id: String,
    pub skill_id: String,
    pub display_name: String,
    pub default_version: Option<String>,
}

/// Metadata for a single immutable skill version.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SkillVersionRecord {
    pub skill_id: String,
    pub version: String,
    pub version_number: u32,
    pub has_code_files: bool,
}

/// File-level manifest entry stored alongside a normalized skill bundle.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SkillFileRecord {
    pub relative_path: String,
    pub media_type: Option<String>,
    pub size_bytes: u64,
}

/// Canonical in-memory representation of a validated skill bundle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizedSkillBundle {
    pub files: Vec<NormalizedSkillFile>,
    pub skill_md_path: String,
    pub openai_sidecar_path: Option<String>,
    pub has_code_files: bool,
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
