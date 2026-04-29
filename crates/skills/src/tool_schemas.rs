//! Canonical SMG skill tool definitions.
//!
//! These definitions are provider-neutral. Gateway adapters serialize them into
//! provider-specific request shapes, for example Responses `function` tools or
//! Anthropic Messages `custom` tools.

use std::sync::LazyLock;

use serde_json::{json, Value};

use crate::{SkillsConfig, SkillsExecutionConfig};

pub const READ_SKILL_TOOL_NAME: &str = "read_skill";
pub const LIST_SKILL_FILES_TOOL_NAME: &str = "list_skill_files";
pub const READ_SKILL_FILE_TOOL_NAME: &str = "read_skill_file";
pub const EXECUTE_SKILL_TOOL_NAME: &str = "execute_skill";

pub const RESERVED_SKILL_TOOL_NAMES: [&str; 4] = [
    READ_SKILL_TOOL_NAME,
    LIST_SKILL_FILES_TOOL_NAME,
    READ_SKILL_FILE_TOOL_NAME,
    EXECUTE_SKILL_TOOL_NAME,
];

#[derive(Debug, Clone, Copy)]
pub struct SkillToolDefinition {
    pub name: &'static str,
    pub description: &'static str,
    pub input_schema: &'static Value,
    pub output_schema: &'static Value,
}

pub fn read_skill_tool() -> SkillToolDefinition {
    SkillToolDefinition {
        name: READ_SKILL_TOOL_NAME,
        description: "Fetch the full SKILL.md body for a skill attached to this request. Call this when the user's task matches a skill's description in the skills_instructions listing, or when the user explicitly names a skill. The returned body contains YAML frontmatter and Markdown instructions; read only what you need.",
        input_schema: LazyLock::force(&READ_SKILL_INPUT_SCHEMA),
        output_schema: LazyLock::force(&READ_SKILL_OUTPUT_SCHEMA),
    }
}

pub fn list_skill_files_tool() -> SkillToolDefinition {
    SkillToolDefinition {
        name: LIST_SKILL_FILES_TOOL_NAME,
        description: "List the non-SKILL.md files bundled with a skill (scripts, templates, references, assets). Returns a flat list of paths relative to the skill root. Call this when SKILL.md references a file under scripts/ or references/ and you need to know which files exist before reading one.",
        input_schema: LazyLock::force(&SKILL_ID_ONLY_INPUT_SCHEMA),
        output_schema: LazyLock::force(&LIST_SKILL_FILES_OUTPUT_SCHEMA),
    }
}

pub fn read_skill_file_tool() -> SkillToolDefinition {
    SkillToolDefinition {
        name: READ_SKILL_FILE_TOOL_NAME,
        description: "Read the content of a single file in a skill bundle. Use this to load a specific script, template, or reference file mentioned in SKILL.md. Do not bulk-load files - read only what you need for the current task. The response may return text (utf8 encoding) or base64-encoded bytes (binary encoding) depending on the file's type, as reported by list_skill_files.",
        input_schema: LazyLock::force(&READ_SKILL_FILE_INPUT_SCHEMA),
        output_schema: LazyLock::force(&READ_SKILL_FILE_OUTPUT_SCHEMA),
    }
}

pub fn execute_skill_tool() -> SkillToolDefinition {
    SkillToolDefinition {
        name: EXECUTE_SKILL_TOOL_NAME,
        description: "Run a skill's code via the configured external executor. Use this for skills whose SKILL.md describes a workflow to be executed rather than transcribing or re-implementing the code yourself. The input is a free-form JSON object whose shape is defined by the specific skill's SKILL.md. Only small model-consumable files are returned inline in files_produced; larger artifacts are out of scope for this transport.",
        input_schema: LazyLock::force(&EXECUTE_SKILL_INPUT_SCHEMA),
        output_schema: LazyLock::force(&EXECUTE_SKILL_OUTPUT_SCHEMA),
    }
}

pub fn read_only_skill_tools() -> [SkillToolDefinition; 3] {
    [
        read_skill_tool(),
        list_skill_files_tool(),
        read_skill_file_tool(),
    ]
}

pub fn skill_tools(include_execution: bool) -> Vec<SkillToolDefinition> {
    let mut tools = Vec::with_capacity(if include_execution { 4 } else { 3 });
    tools.extend(read_only_skill_tools());
    if include_execution {
        tools.push(execute_skill_tool());
    }
    tools
}

pub fn is_reserved_skill_tool_name(name: &str) -> bool {
    RESERVED_SKILL_TOOL_NAMES.contains(&name)
}

pub fn is_skill_executor_configured(config: &SkillsConfig) -> bool {
    is_execution_configured(&config.execution)
}

pub fn should_register_execute_skill(config: Option<&SkillsConfig>, has_code_files: bool) -> bool {
    has_code_files && config.is_some_and(is_skill_executor_configured)
}

fn is_execution_configured(config: &SkillsExecutionConfig) -> bool {
    let has_url = config
        .executor_url
        .as_deref()
        .is_some_and(|url| !url.trim().is_empty());
    let has_key = config
        .executor_api_key
        .as_deref()
        .is_some_and(|key| !key.trim().is_empty());
    has_url && has_key
}

fn skill_id_only_input_schema(description: &'static str) -> Value {
    json!({
        "type": "object",
        "properties": {
            "skill_id": {
                "type": "string",
                "description": description
            }
        },
        "required": ["skill_id"],
        "additionalProperties": false
    })
}

static READ_SKILL_INPUT_SCHEMA: LazyLock<Value> = LazyLock::new(|| {
    skill_id_only_input_schema("Skill id exactly as it appears in the skills_instructions listing.")
});

static SKILL_ID_ONLY_INPUT_SCHEMA: LazyLock<Value> =
    LazyLock::new(|| skill_id_only_input_schema("Skill id from the skills_instructions listing."));

static READ_SKILL_FILE_INPUT_SCHEMA: LazyLock<Value> = LazyLock::new(|| {
    json!({
        "type": "object",
        "properties": {
            "skill_id": {
                "type": "string",
                "description": "Skill id from the skills_instructions listing."
            },
            "path": {
                "type": "string",
                "description": "Path from list_skill_files output. Must match exactly - relative to skill root, forward-slash delimited. Does not accept absolute paths, '..', or symlink traversal."
            },
            "offset": {
                "type": "integer",
                "minimum": 0,
                "description": "Byte offset into the file. Defaults to 0. Use with length to chunk large files."
            },
            "length": {
                "type": "integer",
                "minimum": 1,
                "maximum": 65536,
                "description": "Maximum number of bytes to return from offset. Defaults to the largest value that still fits within the tool-result budget."
            }
        },
        "required": ["skill_id", "path"],
        "additionalProperties": false
    })
});

static EXECUTE_SKILL_INPUT_SCHEMA: LazyLock<Value> = LazyLock::new(|| {
    json!({
        "type": "object",
        "properties": {
            "skill_id": {
                "type": "string",
                "description": "Skill id from the skills_instructions listing."
            },
            "input": {
                "description": "Free-form JSON payload passed to the skill's entry point. Shape is defined by the skill's SKILL.md. Read SKILL.md first if the expected shape is not obvious.",
                "type": ["object", "array", "string", "number", "boolean", "null"]
            }
        },
        "required": ["skill_id", "input"],
        "additionalProperties": false
    })
});

static READ_SKILL_OUTPUT_SCHEMA: LazyLock<Value> = LazyLock::new(|| {
    json!({
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Skill name from SKILL.md frontmatter."
            },
            "version": {
                "type": "string",
                "description": "Version identifier of the skill."
            },
            "body": {
                "type": "string",
                "description": "Full SKILL.md contents including frontmatter."
            }
        },
        "required": ["name", "version", "body"],
        "additionalProperties": false
    })
});

static LIST_SKILL_FILES_OUTPUT_SCHEMA: LazyLock<Value> = LazyLock::new(|| {
    json!({
        "type": "object",
        "properties": {
            "files": {
                "type": "array",
                "description": "All non-SKILL.md files in the bundle. Capped at upload time, so pagination is not needed.",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path relative to skill root, forward-slash delimited."
                        },
                        "size_bytes": {
                            "type": "integer",
                            "minimum": 0
                        },
                        "mime_type": {
                            "type": "string",
                            "description": "Detected at upload; for example text/x-python or application/octet-stream."
                        },
                        "encoding": {
                            "type": "string",
                            "enum": ["utf8", "binary"],
                            "description": "Whether read_skill_file returns text or base64."
                        }
                    },
                    "required": ["path", "size_bytes", "mime_type", "encoding"],
                    "additionalProperties": false
                }
            }
        },
        "required": ["files"],
        "additionalProperties": false
    })
});

static READ_SKILL_FILE_OUTPUT_SCHEMA: LazyLock<Value> = LazyLock::new(|| {
    json!({
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "File contents. UTF-8 text when encoding=utf8, base64-encoded bytes when encoding=binary."
            },
            "encoding": {
                "type": "string",
                "enum": ["utf8", "base64"]
            },
            "mime_type": {
                "type": "string"
            },
            "offset": {
                "type": "integer",
                "minimum": 0
            },
            "total_size_bytes": {
                "type": "integer",
                "minimum": 0
            },
            "has_more": {
                "type": "boolean",
                "description": "True when unread bytes remain after this chunk."
            }
        },
        "required": ["content", "encoding", "mime_type", "offset", "total_size_bytes", "has_more"],
        "additionalProperties": false
    })
});

static EXECUTE_SKILL_OUTPUT_SCHEMA: LazyLock<Value> = LazyLock::new(|| {
    json!({
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["completed", "paused"],
                "description": "paused means the execution is long-running; resume through the API surface's continuation contract."
            },
            "cookie": {
                "type": ["string", "null"],
                "description": "Continuation cookie, present iff status=paused."
            },
            "summary": {
                "type": ["string", "null"],
                "description": "Textual summary of what the skill did or produced."
            },
            "files_produced": {
                "type": "array",
                "description": "Small files the skill created, delivered inline. Reference the filename when describing results to the user; the client extracts the base64 content from the final response.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Filename, including extension."
                        },
                        "mime_type": {
                            "type": "string",
                            "description": "Detected content type."
                        },
                        "size_bytes": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Size of decoded content."
                        },
                        "content_base64": {
                            "type": "string",
                            "description": "Base64-encoded file content. Subject to the generic tool-result cap and execution output caps."
                        }
                    },
                    "required": ["name", "mime_type", "size_bytes", "content_base64"],
                    "additionalProperties": false
                }
            }
        },
        "required": ["status"],
        "additionalProperties": false
    })
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposes_reserved_skill_tool_names() {
        assert!(is_reserved_skill_tool_name(READ_SKILL_TOOL_NAME));
        assert!(is_reserved_skill_tool_name(LIST_SKILL_FILES_TOOL_NAME));
        assert!(is_reserved_skill_tool_name(READ_SKILL_FILE_TOOL_NAME));
        assert!(is_reserved_skill_tool_name(EXECUTE_SKILL_TOOL_NAME));
        assert!(!is_reserved_skill_tool_name("user_tool"));
    }

    #[test]
    fn read_only_tool_set_omits_execute_skill() {
        let tools = skill_tools(false);
        let names: Vec<&str> = tools.iter().map(|tool| tool.name).collect();

        assert_eq!(
            names,
            vec![
                READ_SKILL_TOOL_NAME,
                LIST_SKILL_FILES_TOOL_NAME,
                READ_SKILL_FILE_TOOL_NAME
            ]
        );
    }

    #[test]
    fn execution_tool_set_includes_execute_skill_last() {
        let tools = skill_tools(true);
        let names: Vec<&str> = tools.iter().map(|tool| tool.name).collect();

        assert_eq!(
            names,
            vec![
                READ_SKILL_TOOL_NAME,
                LIST_SKILL_FILES_TOOL_NAME,
                READ_SKILL_FILE_TOOL_NAME,
                EXECUTE_SKILL_TOOL_NAME
            ]
        );
    }

    #[test]
    fn executor_configuration_requires_url_and_key() {
        let mut config = SkillsConfig::default();
        assert!(!should_register_execute_skill(Some(&config), true));

        config.execution.executor_url = Some("http://executor.internal".to_string());
        assert!(!should_register_execute_skill(Some(&config), true));

        config.execution.executor_api_key = Some("secret".to_string());
        assert!(should_register_execute_skill(Some(&config), true));
        assert!(!should_register_execute_skill(Some(&config), false));
        assert!(!should_register_execute_skill(None, true));
    }

    #[test]
    fn schemas_are_closed_and_have_expected_required_fields() {
        let tool = read_skill_file_tool();

        assert_eq!(tool.input_schema["additionalProperties"], false);
        assert_eq!(tool.input_schema["required"], json!(["skill_id", "path"]));
        assert_eq!(tool.output_schema["additionalProperties"], false);
        assert_eq!(
            tool.output_schema["required"],
            json!([
                "content",
                "encoding",
                "mime_type",
                "offset",
                "total_size_bytes",
                "has_more"
            ])
        );
    }
}
