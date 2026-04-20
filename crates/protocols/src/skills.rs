//! Skill protocol types shared across API surfaces.
//!
//! This module keeps wire-facing request fragments separate from the
//! request-local/internal types the router resolves after attach time.

use serde::{
    de::{self, Visitor},
    Deserialize, Deserializer, Serialize, Serializer,
};
use serde_json::{Map, Value};

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

#[expect(
    dead_code,
    reason = "Skill resolution lands in follow-up PRs, but these internal protocol-side types stay crate-private now"
)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ResolvedSkillRef {
    pub(crate) public_skill_id: String,
    pub(crate) origin: SkillOrigin,
    pub(crate) requested_version: Option<SkillVersionRef>,
    pub(crate) pinned: Option<PinnedSkillVersion>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PinnedSkillVersion {
    pub(crate) version: String,
    pub(crate) version_number: u32,
}

#[expect(
    dead_code,
    reason = "Skill resolution lands in follow-up PRs, but origin tracking belongs with the other crate-private resolver types"
)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum SkillOrigin {
    AnthropicProvider {
        skill_id: String,
        raw_version: Option<String>,
    },
    OpenAIProvider {
        skill_id: String,
        raw_version: Option<String>,
    },
    OpenAIOpaquePassThrough {
        raw: Value,
    },
    SmgStorage {
        skill_id: String,
    },
    ClientLocalPath {
        name: String,
        description: String,
        path: String,
    },
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
