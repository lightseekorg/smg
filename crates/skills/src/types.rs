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
