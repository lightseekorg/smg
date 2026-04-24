//! Skill protocol types shared across API surfaces.
//!
//! This module keeps wire-facing request fragments separate from the
//! request-local/internal types the router resolves after attach time.

use serde::{
    de::{self, Visitor},
    Deserialize, Deserializer, Serialize, Serializer,
};
use serde_json::{Map, Value};
use validator::Validate;

use crate::validated::Normalizable;

/// Multipart field carrying the explicit admin target tenant id.
pub const SKILLS_MULTIPART_TENANT_ID_FIELD: &str = "tenant_id";
/// Multipart field for zip archive uploads.
pub const SKILLS_MULTIPART_BUNDLE_FIELD: &str = "bundle";
/// Alternate multipart field name accepted for zip archive uploads.
pub const SKILLS_MULTIPART_FILE_FIELD: &str = "file";
/// Multipart field for raw file uploads.
pub const SKILLS_MULTIPART_FILES_FIELD: &str = "files[]";

/// Accepted in `/v1/messages` -> `container.skills[]`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type")]
pub enum MessagesSkillRef {
    /// Anthropic-curated skill that SMG passes through unchanged.
    #[serde(rename = "anthropic")]
    Anthropic {
        skill_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        version: Option<String>,
    },
    /// SMG-managed skill resolved from storage.
    #[serde(rename = "custom")]
    Custom {
        skill_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        version: Option<SkillVersionRef>,
    },
}

/// Accepted in `/v1/responses` -> `tools[].environment.skills[]`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type")]
pub enum ResponsesSkillRef {
    /// Hosted skill reference. Ownership is decided by storage lookup, not id shape.
    #[serde(rename = "skill_reference")]
    Reference {
        skill_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        version: Option<SkillVersionRef>,
    },
    /// Client-local path. SMG never resolves or inspects this payload.
    #[serde(rename = "local")]
    Local {
        name: String,
        description: String,
        path: String,
    },
}

/// Opaque provider-owned Responses skill payload.
///
/// This wrapper keeps the public type aligned with the runtime contract:
/// opaque skill entries must still be JSON objects.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(transparent)]
pub struct OpaqueOpenAIObject(pub Map<String, Value>);

/// One entry in `/v1/responses` -> `tools[].environment.skills[]`.
///
/// SMG interprets the typed subset directly and preserves all other JSON
/// object entries as opaque provider-owned payloads.
#[derive(Debug, Clone, PartialEq, Serialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum ResponsesSkillEntry {
    Typed(ResponsesSkillRef),
    OpaqueOpenAI(OpaqueOpenAIObject),
}

impl<'de> Deserialize<'de> for ResponsesSkillEntry {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        let Value::Object(object) = value else {
            return Err(de::Error::custom(
                "responses skill entries must be JSON objects",
            ));
        };

        if matches_typed_responses_skill_ref(&object) {
            return serde_json::from_value::<ResponsesSkillRef>(Value::Object(object.clone()))
                .map(Self::Typed)
                .map_err(de::Error::custom);
        }

        Ok(Self::OpaqueOpenAI(OpaqueOpenAIObject(object)))
    }
}

/// Version reference accepted by skill attachment surfaces.
///
/// Deserialization is intentionally manual to reject ambiguous numeric strings
/// like `"2"` rather than guessing between integer and timestamp semantics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkillVersionRef {
    /// The literal string `"latest"`.
    Latest,
    /// Integer reference (OpenAI-style).
    Integer(u32),
    /// Timestamp string (Anthropic-style).
    Timestamp(String),
}

/// Public skill object returned by CRUD/read endpoints.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SkillResponse {
    pub id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub short_description: Option<String>,
    pub description: String,
    pub source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latest_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_version: Option<String>,
    pub has_code_files: bool,
    pub created_at: String,
    pub updated_at: String,
}

/// Public immutable skill-version object returned by CRUD/read endpoints.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SkillVersionResponse {
    pub skill_id: String,
    pub version: String,
    pub version_number: u32,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub short_description: Option<String>,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interface: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dependencies: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub policy: Option<Value>,
    pub deprecated: bool,
    pub files: Vec<SkillVersionFileResponse>,
    pub created_at: String,
}

/// Public file-manifest entry for a skill version.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SkillVersionFileResponse {
    pub path: String,
    pub size_bytes: u64,
}

/// Non-fatal warning surfaced when a bundle was accepted with partial metadata salvage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SkillWarningResponse {
    pub kind: String,
    pub path: String,
    pub message: String,
}

/// Success payload for skill create/upload mutations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SkillMutationResponse {
    pub skill: SkillResponse,
    pub version: SkillVersionResponse,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<SkillWarningResponse>,
}

/// JSON body accepted by `PATCH /v1/skills/{skill_id}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Validate, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SkillPatchRequest {
    pub default_version: SkillVersionRef,
}

impl Normalizable for SkillPatchRequest {}

/// JSON body accepted by `PATCH /v1/skills/{skill_id}/versions/{version}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Validate, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SkillVersionPatchRequest {
    pub deprecated: bool,
}

impl Normalizable for SkillVersionPatchRequest {}

/// Query parameters accepted by `GET /v1/skills`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default, schemars::JsonSchema)]
pub struct SkillsListQuery {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Query parameters accepted by `GET /v1/skills/{skill_id}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default, schemars::JsonSchema)]
pub struct SkillGetQuery {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant_id: Option<String>,
}

/// Query parameters accepted by `GET /v1/skills/{skill_id}/versions`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default, schemars::JsonSchema)]
pub struct SkillVersionsListQuery {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after: Option<String>,
    #[serde(default)]
    pub include_deprecated: bool,
}

/// One page of skill records from `GET /v1/skills`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SkillsListResponse {
    pub object: String,
    pub data: Vec<SkillResponse>,
    pub has_more: bool,
    /// Opaque pagination cursor for the first item on this page.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_id: Option<String>,
    /// Opaque pagination cursor for the last item on this page.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_id: Option<String>,
}

/// One page of version records from `GET /v1/skills/{skill_id}/versions`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SkillVersionsListResponse {
    pub object: String,
    pub data: Vec<SkillVersionResponse>,
    pub has_more: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_id: Option<String>,
}

/// Standard error envelope for skills CRUD endpoints.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SkillsErrorEnvelope {
    pub error: SkillsErrorBody,
}

/// Structured error body for skills CRUD endpoints.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SkillsErrorBody {
    pub code: String,
    pub message: String,
}

impl Serialize for SkillVersionRef {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Latest => serializer.serialize_str("latest"),
            Self::Integer(version) => serializer.serialize_u32(*version),
            Self::Timestamp(version) => serializer.serialize_str(version),
        }
    }
}

impl<'de> Deserialize<'de> for SkillVersionRef {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct SkillVersionRefVisitor;

        impl<'de> Visitor<'de> for SkillVersionRefVisitor {
            type Value = SkillVersionRef;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str(
                    "the string `latest`, an integer version, or a 10+ digit timestamp string",
                )
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                let value = u32::try_from(value).map_err(|_| {
                    E::custom(format!(
                        "skill version integer is out of range for u32: {value}"
                    ))
                })?;
                Ok(SkillVersionRef::Integer(value))
            }

            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                if value < 0 {
                    return Err(E::custom("skill version integer must be non-negative"));
                }
                self.visit_u64(value as u64)
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                parse_skill_version_str(value).map_err(E::custom)
            }

            fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                self.visit_str(&value)
            }
        }

        deserializer.deserialize_any(SkillVersionRefVisitor)
    }
}

impl schemars::JsonSchema for SkillVersionRef {
    fn schema_name() -> String {
        "SkillVersionRef".to_string()
    }

    fn json_schema(_gen: &mut schemars::gen::SchemaGenerator) -> schemars::schema::Schema {
        use schemars::schema::*;

        let latest_schema = SchemaObject {
            instance_type: Some(InstanceType::String.into()),
            enum_values: Some(vec!["latest".into()]),
            ..Default::default()
        };
        let integer_schema = SchemaObject {
            instance_type: Some(InstanceType::Integer.into()),
            ..Default::default()
        };
        let timestamp_schema = SchemaObject {
            instance_type: Some(InstanceType::String.into()),
            string: Some(Box::new(StringValidation {
                pattern: Some("^[1-9][0-9]{9,}$".to_string()),
                ..Default::default()
            })),
            ..Default::default()
        };

        SchemaObject {
            subschemas: Some(Box::new(SubschemaValidation {
                one_of: Some(vec![
                    latest_schema.into(),
                    integer_schema.into(),
                    timestamp_schema.into(),
                ]),
                ..Default::default()
            })),
            ..Default::default()
        }
        .into()
    }
}

fn matches_typed_responses_skill_ref(value: &Map<String, Value>) -> bool {
    // Keep this tag/field matching in sync with ResponsesSkillRef's
    // `#[serde(rename = "...")]` values and allowed field sets.
    match value.get("type").and_then(Value::as_str) {
        Some("skill_reference") => has_only_keys(value, &["type", "skill_id", "version"]),
        Some("local") => has_only_keys(value, &["type", "name", "description", "path"]),
        _ => false,
    }
}

fn has_only_keys(value: &Map<String, Value>, allowed_keys: &[&str]) -> bool {
    value.keys().all(|key| allowed_keys.contains(&key.as_str()))
}

fn parse_skill_version_str(value: &str) -> Result<SkillVersionRef, String> {
    if value == "latest" {
        return Ok(SkillVersionRef::Latest);
    }

    if !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()) {
        if value.len() <= 9 {
            return Err(format!(
                "ambiguous skill version string `{value}`: use a JSON number for integer versions or a 10+ digit string for timestamps"
            ));
        }

        if value.starts_with('0') {
            return Err(format!(
                "ambiguous skill version string `{value}`: leading zeros are not allowed in timestamp strings"
            ));
        }

        return Ok(SkillVersionRef::Timestamp(value.to_string()));
    }

    Err(format!(
        "invalid skill version string `{value}`: expected `latest` or a 10+ digit timestamp string without leading zeros"
    ))
}
