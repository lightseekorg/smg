use std::fmt::Write as _;

use serde::Deserialize;
use serde_yml::{Mapping, Value};
use thiserror::Error;

use crate::types::{
    ParsedSkillBundle, SkillDependencyTool, SkillInterfaceMetadata, SkillParseWarning,
    SkillParseWarningKind, SkillPolicyMetadata, SkillSidecarDependencies,
};

const MAX_NAME_LEN: usize = 64;
const MAX_DESCRIPTION_LEN: usize = 1024;
const MAX_SIDECAR_STRING_LEN: usize = 1024;
const RESERVED_SKILL_NAMES: [&str; 3] = ["anthropic", "claude", "openai"];

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SkillParseError {
    #[error("SKILL.md must begin with a YAML frontmatter block delimited by --- lines")]
    MissingFrontmatter,
    #[error("SKILL.md frontmatter is missing a closing --- delimiter")]
    MissingFrontmatterTerminator,
    #[error("SKILL.md frontmatter YAML is invalid: {message}")]
    InvalidFrontmatterYaml { message: String },
    #[error("SKILL.md field `{field}` is required")]
    MissingRequiredField { field: &'static str },
    #[error("SKILL.md field `{field}` is invalid: {message}")]
    InvalidField {
        field: &'static str,
        message: String,
    },
}

#[derive(Debug, Deserialize)]
struct SkillFrontmatter {
    name: Option<String>,
    description: Option<String>,
    #[serde(default)]
    metadata: SkillFrontmatterMetadata,
}

#[derive(Debug, Default, Deserialize)]
struct SkillFrontmatterMetadata {
    #[serde(rename = "short-description")]
    short_description: Option<String>,
}

pub fn parse_skill_bundle(
    skill_md: &str,
    openai_yaml: Option<&str>,
) -> Result<ParsedSkillBundle, SkillParseError> {
    let (frontmatter_yaml, instructions_body) = split_frontmatter(skill_md)?;
    let frontmatter: SkillFrontmatter = serde_yml::from_str(frontmatter_yaml).map_err(|error| {
        SkillParseError::InvalidFrontmatterYaml {
            message: error.to_string(),
        }
    })?;

    let name = validate_required_name(frontmatter.name)?;
    let description = validate_required_description(frontmatter.description)?;
    let metadata_short_description = validate_optional_short_description(
        frontmatter.metadata.short_description,
        "metadata.short-description",
    )?;

    let mut warnings = Vec::new();
    let sidecar = parse_openai_sidecar(openai_yaml, &mut warnings);
    let short_description = sidecar
        .interface
        .as_ref()
        .and_then(|interface| interface.short_description.clone())
        .or(metadata_short_description);

    Ok(ParsedSkillBundle {
        name,
        description,
        short_description,
        instructions_body: instructions_body.to_owned(),
        interface: sidecar.interface,
        dependencies: sidecar.dependencies,
        policy: sidecar.policy,
        warnings,
    })
}

fn split_frontmatter(skill_md: &str) -> Result<(&str, &str), SkillParseError> {
    let Some(first_line_end) = find_line_end(skill_md, 0) else {
        return Err(SkillParseError::MissingFrontmatter);
    };

    if trim_line_ending(&skill_md[..first_line_end]) != "---" {
        return Err(SkillParseError::MissingFrontmatter);
    }

    let yaml_start = first_line_end;
    let mut cursor = yaml_start;
    while cursor < skill_md.len() {
        let next_line_end = find_line_end(skill_md, cursor).unwrap_or(skill_md.len());
        let line = trim_line_ending(&skill_md[cursor..next_line_end]);
        if line == "---" {
            let body = &skill_md[next_line_end..];
            let frontmatter = &skill_md[yaml_start..cursor];
            return Ok((frontmatter, body));
        }
        cursor = next_line_end;
    }

    Err(SkillParseError::MissingFrontmatterTerminator)
}

fn find_line_end(content: &str, start: usize) -> Option<usize> {
    if start >= content.len() {
        return None;
    }

    content[start..]
        .find('\n')
        .map(|offset| start + offset + 1)
        .or(Some(content.len()))
}

fn trim_line_ending(line: &str) -> &str {
    line.trim_end_matches(['\n', '\r'])
}

fn validate_required_name(name: Option<String>) -> Result<String, SkillParseError> {
    let name = name.ok_or(SkillParseError::MissingRequiredField { field: "name" })?;
    validate_name(&name)?;
    Ok(name)
}

fn validate_required_description(description: Option<String>) -> Result<String, SkillParseError> {
    let description = description.ok_or(SkillParseError::MissingRequiredField {
        field: "description",
    })?;
    validate_description(&description)?;
    Ok(description)
}

fn validate_optional_short_description(
    value: Option<String>,
    field: &'static str,
) -> Result<Option<String>, SkillParseError> {
    match value {
        Some(short_description) => {
            validate_length(field, &short_description, MAX_DESCRIPTION_LEN)?;
            Ok(Some(short_description))
        }
        None => Ok(None),
    }
}

fn validate_name(name: &str) -> Result<(), SkillParseError> {
    if name.is_empty() {
        return Err(SkillParseError::InvalidField {
            field: "name",
            message: "must not be empty".to_owned(),
        });
    }
    if name.len() > MAX_NAME_LEN {
        return Err(SkillParseError::InvalidField {
            field: "name",
            message: format!("must be at most {MAX_NAME_LEN} characters"),
        });
    }
    if RESERVED_SKILL_NAMES
        .iter()
        .any(|reserved| reserved.eq_ignore_ascii_case(name))
    {
        return Err(SkillParseError::InvalidField {
            field: "name",
            message: "is reserved".to_owned(),
        });
    }

    for segment in name.split(':') {
        if segment.is_empty() {
            return Err(SkillParseError::InvalidField {
                field: "name",
                message: "must not contain empty namespace segments".to_owned(),
            });
        }

        let mut chars = segment.chars();
        let Some(first) = chars.next() else {
            return Err(SkillParseError::InvalidField {
                field: "name",
                message: "must not contain empty namespace segments".to_owned(),
            });
        };

        if !first.is_ascii_lowercase() && !first.is_ascii_digit() {
            return Err(SkillParseError::InvalidField {
                field: "name",
                message: "must start each namespace segment with a lowercase letter or digit"
                    .to_owned(),
            });
        }

        if chars.any(|ch| !ch.is_ascii_lowercase() && !ch.is_ascii_digit() && ch != '-') {
            return Err(SkillParseError::InvalidField {
                field: "name",
                message:
                    "may only contain lowercase letters, digits, hyphens, and namespace colons"
                        .to_owned(),
            });
        }
    }

    Ok(())
}

fn validate_description(description: &str) -> Result<(), SkillParseError> {
    if description.trim().is_empty() {
        return Err(SkillParseError::InvalidField {
            field: "description",
            message: "must not be empty".to_owned(),
        });
    }
    validate_length("description", description, MAX_DESCRIPTION_LEN)?;
    if contains_xml_like_tag(description) {
        return Err(SkillParseError::InvalidField {
            field: "description",
            message: "must not contain XML-like tags".to_owned(),
        });
    }
    Ok(())
}

fn validate_length(
    field: &'static str,
    value: &str,
    max_len: usize,
) -> Result<(), SkillParseError> {
    if value.len() > max_len {
        return Err(SkillParseError::InvalidField {
            field,
            message: format!("must be at most {max_len} characters"),
        });
    }
    Ok(())
}

fn contains_xml_like_tag(value: &str) -> bool {
    let bytes = value.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'<' {
            let next_index = index + 1;
            if next_index < bytes.len()
                && (bytes[next_index].is_ascii_alphabetic() || bytes[next_index] == b'/')
                && bytes[next_index..].contains(&b'>')
            {
                return true;
            }
        }
        index += 1;
    }
    false
}

#[derive(Default)]
struct ParsedOpenAISidecar {
    interface: Option<SkillInterfaceMetadata>,
    dependencies: Option<SkillSidecarDependencies>,
    policy: Option<SkillPolicyMetadata>,
}

fn parse_openai_sidecar(
    openai_yaml: Option<&str>,
    warnings: &mut Vec<SkillParseWarning>,
) -> ParsedOpenAISidecar {
    let Some(openai_yaml) = openai_yaml else {
        return ParsedOpenAISidecar::default();
    };

    let value = match serde_yml::from_str::<Value>(openai_yaml) {
        Ok(value) => value,
        Err(error) => {
            warnings.push(SkillParseWarning {
                kind: SkillParseWarningKind::SidecarFileIgnored,
                path: "agents/openai.yaml".to_owned(),
                message: format!("ignored invalid YAML: {error}"),
            });
            return ParsedOpenAISidecar::default();
        }
    };

    let Some(mapping) = value.as_mapping() else {
        warnings.push(SkillParseWarning {
            kind: SkillParseWarningKind::SidecarFieldIgnored,
            path: "agents/openai.yaml".to_owned(),
            message: "expected a mapping at the YAML document root".to_owned(),
        });
        return ParsedOpenAISidecar::default();
    };

    let interface = parse_interface(mapping_get(mapping, "interface"), warnings);
    let dependencies = parse_dependencies(mapping_get(mapping, "dependencies"), warnings);
    let policy = parse_policy(mapping_get(mapping, "policy"), warnings);

    ParsedOpenAISidecar {
        interface,
        dependencies,
        policy,
    }
}

fn parse_interface(
    value: Option<&Value>,
    warnings: &mut Vec<SkillParseWarning>,
) -> Option<SkillInterfaceMetadata> {
    let value = value?;
    let Some(mapping) = value.as_mapping() else {
        push_field_warning(
            warnings,
            "agents/openai.yaml.interface",
            "expected a mapping",
        );
        return None;
    };

    let interface = SkillInterfaceMetadata {
        display_name: parse_string_field(
            mapping,
            "display_name",
            "agents/openai.yaml.interface.display_name",
            warnings,
            |value| validate_sidecar_string_len("interface.display_name", value, 64),
        ),
        short_description: parse_string_field(
            mapping,
            "short_description",
            "agents/openai.yaml.interface.short_description",
            warnings,
            |value| {
                validate_sidecar_string_len(
                    "interface.short_description",
                    value,
                    MAX_SIDECAR_STRING_LEN,
                )
            },
        ),
        icon_small: parse_string_field(
            mapping,
            "icon_small",
            "agents/openai.yaml.interface.icon_small",
            warnings,
            validate_sidecar_path,
        ),
        icon_large: parse_string_field(
            mapping,
            "icon_large",
            "agents/openai.yaml.interface.icon_large",
            warnings,
            validate_sidecar_path,
        ),
        brand_color: parse_string_field(
            mapping,
            "brand_color",
            "agents/openai.yaml.interface.brand_color",
            warnings,
            validate_brand_color,
        ),
        default_prompt: parse_string_field(
            mapping,
            "default_prompt",
            "agents/openai.yaml.interface.default_prompt",
            warnings,
            |value| {
                validate_sidecar_string_len(
                    "interface.default_prompt",
                    value,
                    MAX_SIDECAR_STRING_LEN,
                )
            },
        ),
    };

    (!interface.is_empty()).then_some(interface)
}

fn parse_dependencies(
    value: Option<&Value>,
    warnings: &mut Vec<SkillParseWarning>,
) -> Option<SkillSidecarDependencies> {
    let value = value?;
    let Some(mapping) = value.as_mapping() else {
        push_field_warning(
            warnings,
            "agents/openai.yaml.dependencies",
            "expected a mapping",
        );
        return None;
    };

    let tools_value = mapping_get(mapping, "tools")?;
    let Some(sequence) = tools_value.as_sequence() else {
        push_field_warning(
            warnings,
            "agents/openai.yaml.dependencies.tools",
            "expected a sequence",
        );
        return None;
    };

    let tools = sequence
        .iter()
        .enumerate()
        .filter_map(|(index, value)| parse_dependency_tool(value, index, warnings))
        .collect::<Vec<_>>();

    let dependencies = SkillSidecarDependencies { tools };
    (!dependencies.is_empty()).then_some(dependencies)
}

fn parse_dependency_tool(
    value: &Value,
    index: usize,
    warnings: &mut Vec<SkillParseWarning>,
) -> Option<SkillDependencyTool> {
    let Some(mapping) = value.as_mapping() else {
        push_field_warning(
            warnings,
            &format!("agents/openai.yaml.dependencies.tools[{index}]"),
            "expected a mapping",
        );
        return None;
    };

    let path = format!("agents/openai.yaml.dependencies.tools[{index}]");
    let tool_type = parse_required_string_field(mapping, "type", &path, warnings, |value| {
        validate_sidecar_string_len("dependencies.tools[].type", value, MAX_SIDECAR_STRING_LEN)
    })?;
    let value = parse_required_string_field(mapping, "value", &path, warnings, |value| {
        validate_sidecar_string_len("dependencies.tools[].value", value, MAX_SIDECAR_STRING_LEN)
    })?;

    Some(SkillDependencyTool {
        tool_type,
        value,
        description: parse_string_field(
            mapping,
            "description",
            &format!("{path}.description"),
            warnings,
            |value| {
                validate_sidecar_string_len(
                    "dependencies.tools[].description",
                    value,
                    MAX_SIDECAR_STRING_LEN,
                )
            },
        ),
        transport: parse_string_field(
            mapping,
            "transport",
            &format!("{path}.transport"),
            warnings,
            |value| {
                validate_sidecar_string_len(
                    "dependencies.tools[].transport",
                    value,
                    MAX_SIDECAR_STRING_LEN,
                )
            },
        ),
        command: parse_string_field(
            mapping,
            "command",
            &format!("{path}.command"),
            warnings,
            |value| {
                validate_sidecar_string_len(
                    "dependencies.tools[].command",
                    value,
                    MAX_SIDECAR_STRING_LEN,
                )
            },
        ),
        url: parse_string_field(mapping, "url", &format!("{path}.url"), warnings, |value| {
            validate_sidecar_string_len("dependencies.tools[].url", value, MAX_SIDECAR_STRING_LEN)
        }),
    })
}

fn parse_policy(
    value: Option<&Value>,
    warnings: &mut Vec<SkillParseWarning>,
) -> Option<SkillPolicyMetadata> {
    let value = value?;
    let Some(mapping) = value.as_mapping() else {
        push_field_warning(warnings, "agents/openai.yaml.policy", "expected a mapping");
        return None;
    };

    let allow_implicit_invocation = parse_bool_field(
        mapping,
        "allow_implicit_invocation",
        "agents/openai.yaml.policy.allow_implicit_invocation",
        warnings,
    );
    let products = parse_string_sequence_field(
        mapping,
        "products",
        "agents/openai.yaml.policy.products",
        warnings,
    );

    let policy = SkillPolicyMetadata {
        allow_implicit_invocation,
        products,
    };
    (!policy.is_empty()).then_some(policy)
}

fn parse_string_field<F>(
    mapping: &Mapping,
    key: &str,
    path: &str,
    warnings: &mut Vec<SkillParseWarning>,
    validator: F,
) -> Option<String>
where
    F: Fn(&str) -> Result<(), String>,
{
    let value = mapping_get(mapping, key)?;
    match value {
        Value::String(string_value) => match validator(string_value) {
            Ok(()) => Some(string_value.clone()),
            Err(message) => {
                push_field_warning(warnings, path, &message);
                None
            }
        },
        _ => {
            push_field_warning(warnings, path, "expected a string");
            None
        }
    }
}

fn parse_required_string_field<F>(
    mapping: &Mapping,
    key: &str,
    parent_path: &str,
    warnings: &mut Vec<SkillParseWarning>,
    validator: F,
) -> Option<String>
where
    F: Fn(&str) -> Result<(), String>,
{
    let mut field_path = String::from(parent_path);
    let _ = write!(&mut field_path, ".{key}");
    let Some(value) = mapping_get(mapping, key) else {
        push_field_warning(warnings, &field_path, "missing required field");
        return None;
    };
    match value {
        Value::String(string_value) => match validator(string_value) {
            Ok(()) => Some(string_value.clone()),
            Err(message) => {
                push_field_warning(warnings, &field_path, &message);
                None
            }
        },
        _ => {
            push_field_warning(warnings, &field_path, "expected a string");
            None
        }
    }
}

fn parse_bool_field(
    mapping: &Mapping,
    key: &str,
    path: &str,
    warnings: &mut Vec<SkillParseWarning>,
) -> Option<bool> {
    let value = mapping_get(mapping, key)?;
    match value {
        Value::Bool(boolean_value) => Some(*boolean_value),
        _ => {
            push_field_warning(warnings, path, "expected a boolean");
            None
        }
    }
}

fn parse_string_sequence_field(
    mapping: &Mapping,
    key: &str,
    path: &str,
    warnings: &mut Vec<SkillParseWarning>,
) -> Vec<String> {
    let Some(value) = mapping_get(mapping, key) else {
        return Vec::new();
    };
    let Some(sequence) = value.as_sequence() else {
        push_field_warning(warnings, path, "expected a sequence");
        return Vec::new();
    };

    sequence
        .iter()
        .enumerate()
        .filter_map(|(index, value)| match value {
            Value::String(string_value) => Some(string_value.clone()),
            _ => {
                push_field_warning(warnings, &format!("{path}[{index}]"), "expected a string");
                None
            }
        })
        .collect()
}

fn mapping_get<'a>(mapping: &'a Mapping, key: &str) -> Option<&'a Value> {
    mapping.iter().find_map(|(candidate, value)| {
        candidate
            .as_str()
            .filter(|candidate_key| *candidate_key == key)
            .map(|_| value)
    })
}

fn push_field_warning(warnings: &mut Vec<SkillParseWarning>, path: &str, message: &str) {
    warnings.push(SkillParseWarning {
        kind: SkillParseWarningKind::SidecarFieldIgnored,
        path: path.to_owned(),
        message: message.to_owned(),
    });
}

fn validate_sidecar_string_len(field: &str, value: &str, max_len: usize) -> Result<(), String> {
    if value.len() > max_len {
        return Err(format!("{field} must be at most {max_len} characters"));
    }
    Ok(())
}

fn validate_brand_color(value: &str) -> Result<(), String> {
    if value.len() != 7 {
        return Err("brand color must be a CSS hex color like #1D4ED8".to_owned());
    }
    let bytes = value.as_bytes();
    if bytes.first() != Some(&b'#') || bytes[1..].iter().any(|byte| !byte.is_ascii_hexdigit()) {
        return Err("brand color must be a CSS hex color like #1D4ED8".to_owned());
    }
    Ok(())
}

fn validate_sidecar_path(value: &str) -> Result<(), String> {
    validate_sidecar_string_len("interface.icon", value, MAX_SIDECAR_STRING_LEN)?;
    if value.is_empty() {
        return Err("path must not be empty".to_owned());
    }
    if value.starts_with('/') || value.starts_with('\\') {
        return Err("path must be relative to the skill root".to_owned());
    }
    if value.contains('\0') {
        return Err("path must not contain NUL bytes".to_owned());
    }
    if value.contains('\\') {
        return Err("path must use forward slashes".to_owned());
    }

    for segment in value.split('/') {
        if segment.is_empty() {
            return Err("path must not contain empty segments".to_owned());
        }
        if segment == "." || segment == ".." {
            return Err("path must not contain traversal segments".to_owned());
        }
    }

    Ok(())
}
