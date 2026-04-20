use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SkillRecord {
    pub tenant_id: String,
    pub skill_id: String,
    pub display_name: String,
    pub default_version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SkillVersionRecord {
    pub skill_id: String,
    pub version: String,
    pub version_number: u32,
    pub has_code_files: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SkillFileRecord {
    pub relative_path: String,
    pub media_type: Option<String>,
    pub size_bytes: u64,
}

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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct SkillSidecarDependencies {
    pub tools: Vec<SkillDependencyTool>,
}

impl SkillSidecarDependencies {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SkillDependencyTool {
    pub tool_type: String,
    pub value: String,
    pub description: Option<String>,
    pub transport: Option<String>,
    pub command: Option<String>,
    pub url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct SkillPolicyMetadata {
    pub allow_implicit_invocation: Option<bool>,
    pub products: Vec<String>,
}

impl SkillPolicyMetadata {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.allow_implicit_invocation.is_none() && self.products.is_empty()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SkillParseWarning {
    pub kind: SkillParseWarningKind,
    pub path: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SkillParseWarningKind {
    SidecarFileIgnored,
    SidecarFieldIgnored,
}
