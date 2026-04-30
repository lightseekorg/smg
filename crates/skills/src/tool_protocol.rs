//! Protocol-facing skill tool adapters and request guardrails.

use std::collections::HashMap;

use openai_protocol::{
    common::Function,
    messages::{self, CustomTool as MessagesCustomTool, InputSchema},
    responses::{FunctionTool, NamespaceTool, ResponseTool},
};
use serde_json::Value;

use crate::{is_reserved_skill_tool_name, skill_tools, SkillToolDefinition};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("tool name '{tool_name}' is reserved for SMG skill tools")]
pub struct ReservedSkillToolNameError {
    tool_name: String,
}

impl ReservedSkillToolNameError {
    #[must_use]
    pub fn tool_name(&self) -> &str {
        &self.tool_name
    }
}

pub fn validate_messages_reserved_skill_tool_names(
    tools: Option<&[messages::Tool]>,
) -> Result<(), ReservedSkillToolNameError> {
    let Some(tools) = tools else {
        return Ok(());
    };

    for tool in tools {
        if let Some(name) = messages_tool_name(tool) {
            validate_user_tool_name(name)?;
        }
    }

    Ok(())
}

pub fn validate_responses_reserved_skill_tool_names(
    tools: Option<&[ResponseTool]>,
) -> Result<(), ReservedSkillToolNameError> {
    let Some(tools) = tools else {
        return Ok(());
    };

    for tool in tools {
        match tool {
            ResponseTool::Function(function_tool) => {
                validate_user_tool_name(&function_tool.function.name)?;
            }
            ResponseTool::Custom(custom_tool) => {
                validate_user_tool_name(&custom_tool.name)?;
            }
            ResponseTool::Namespace(namespace) => {
                validate_user_tool_name(&namespace.name)?;
                for nested_tool in &namespace.tools {
                    validate_namespace_tool_name(nested_tool)?;
                }
            }
            ResponseTool::WebSearchPreview(_)
            | ResponseTool::WebSearch(_)
            | ResponseTool::CodeInterpreter(_)
            | ResponseTool::Mcp(_)
            | ResponseTool::FileSearch(_)
            | ResponseTool::ImageGeneration(_)
            | ResponseTool::Computer
            | ResponseTool::ComputerUsePreview(_)
            | ResponseTool::Shell(_)
            | ResponseTool::ApplyPatch
            | ResponseTool::LocalShell => {}
        }
    }

    Ok(())
}

pub fn response_skill_tools(include_execution: bool) -> Vec<ResponseTool> {
    skill_tools(include_execution)
        .into_iter()
        .map(response_skill_tool)
        .collect()
}

pub fn messages_skill_tools(include_execution: bool) -> Vec<messages::Tool> {
    skill_tools(include_execution)
        .into_iter()
        .map(messages_skill_tool)
        .collect()
}

fn response_skill_tool(tool: SkillToolDefinition) -> ResponseTool {
    ResponseTool::Function(FunctionTool {
        function: Function {
            name: tool.name.to_string(),
            description: Some(tool.description.to_string()),
            parameters: tool.input_schema.clone(),
            strict: Some(false),
        },
    })
}

fn messages_skill_tool(tool: SkillToolDefinition) -> messages::Tool {
    messages::Tool::Custom(MessagesCustomTool {
        name: tool.name.to_string(),
        tool_type: None,
        description: Some(tool.description.to_string()),
        input_schema: input_schema_from_json_schema(tool.input_schema),
        defer_loading: None,
        cache_control: None,
    })
}

fn input_schema_from_json_schema(schema: &Value) -> InputSchema {
    let Some(schema_map) = schema.as_object() else {
        return object_input_schema(HashMap::new());
    };

    let schema_type = schema_map
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("object")
        .to_string();
    let properties = schema_map
        .get("properties")
        .and_then(Value::as_object)
        .map(|obj| {
            obj.iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect()
        });
    let required = schema_map
        .get("required")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        });
    let additional = schema_map
        .iter()
        .filter(|(key, _)| *key != "type" && *key != "properties" && *key != "required")
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();

    InputSchema {
        schema_type,
        properties,
        required,
        additional,
    }
}

fn object_input_schema(additional: HashMap<String, Value>) -> InputSchema {
    InputSchema {
        schema_type: "object".to_string(),
        properties: None,
        required: None,
        additional,
    }
}

fn messages_tool_name(tool: &messages::Tool) -> Option<&str> {
    match tool {
        messages::Tool::McpToolset(_) => None,
        messages::Tool::Custom(tool) => Some(&tool.name),
        messages::Tool::ToolSearch(tool) => Some(&tool.name),
        messages::Tool::Bash(tool) => Some(&tool.name),
        messages::Tool::TextEditor(tool) => Some(&tool.name),
        messages::Tool::WebSearch(tool) => Some(&tool.name),
    }
}

fn validate_namespace_tool_name(tool: &NamespaceTool) -> Result<(), ReservedSkillToolNameError> {
    match tool {
        NamespaceTool::Function(function_tool) => {
            validate_user_tool_name(&function_tool.function.name)
        }
        NamespaceTool::Custom(custom_tool) => validate_user_tool_name(&custom_tool.name),
    }
}

fn validate_user_tool_name(name: &str) -> Result<(), ReservedSkillToolNameError> {
    if is_reserved_skill_tool_name(name) {
        Err(ReservedSkillToolNameError {
            tool_name: name.to_string(),
        })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::{
        common::Function,
        messages::{CustomTool as MessagesCustomTool, InputSchema as MessagesInputSchema},
        responses::{
            CustomTool as ResponseCustomTool, FunctionTool, NamespaceToolDef, ResponseInput,
            ResponsesRequest,
        },
    };
    use serde_json::{json, Value};

    use super::*;
    use crate::EXECUTE_SKILL_TOOL_NAME;

    fn empty_messages_schema() -> MessagesInputSchema {
        MessagesInputSchema {
            schema_type: "object".to_string(),
            properties: None,
            required: None,
            additional: HashMap::new(),
        }
    }

    #[test]
    fn response_skill_tools_serialize_as_function_tools() -> Result<(), serde_json::Error> {
        let tools = response_skill_tools(false);
        let serialized = serde_json::to_value(&tools)?;

        assert_eq!(
            serialized,
            json!([
                {
                    "type": "function",
                    "name": "read_skill",
                    "description": crate::read_skill_tool().description,
                    "parameters": crate::read_skill_tool().input_schema,
                    "strict": false
                },
                {
                    "type": "function",
                    "name": "list_skill_files",
                    "description": crate::list_skill_files_tool().description,
                    "parameters": crate::list_skill_files_tool().input_schema,
                    "strict": false
                },
                {
                    "type": "function",
                    "name": "read_skill_file",
                    "description": crate::read_skill_file_tool().description,
                    "parameters": crate::read_skill_file_tool().input_schema,
                    "strict": false
                }
            ])
        );
        Ok(())
    }

    #[test]
    fn messages_skill_tools_serialize_as_custom_tools() -> Result<(), serde_json::Error> {
        let tools = messages_skill_tools(true);
        let serialized = serde_json::to_value(&tools)?;

        assert_eq!(tools.len(), 4);
        assert_eq!(serialized[0]["name"], "read_skill");
        assert_eq!(serialized[0]["input_schema"]["type"], "object");
        assert_eq!(serialized[0]["input_schema"]["additionalProperties"], false);
        assert_eq!(serialized[3]["name"], EXECUTE_SKILL_TOOL_NAME);
        Ok(())
    }

    #[test]
    fn rejects_reserved_messages_tool_name() {
        let tools = vec![messages::Tool::Custom(MessagesCustomTool {
            name: crate::READ_SKILL_TOOL_NAME.to_string(),
            tool_type: None,
            description: None,
            input_schema: empty_messages_schema(),
            defer_loading: None,
            cache_control: None,
        })];

        assert_eq!(
            validate_messages_reserved_skill_tool_names(Some(&tools)),
            Err(ReservedSkillToolNameError {
                tool_name: crate::READ_SKILL_TOOL_NAME.to_string()
            })
        );
    }

    #[test]
    fn rejects_reserved_responses_function_tool_name() {
        let tools = vec![ResponseTool::Function(FunctionTool {
            function: Function {
                name: crate::READ_SKILL_FILE_TOOL_NAME.to_string(),
                description: None,
                parameters: json!({"type": "object"}),
                strict: None,
            },
        })];

        assert_eq!(
            validate_responses_reserved_skill_tool_names(Some(&tools)),
            Err(ReservedSkillToolNameError {
                tool_name: crate::READ_SKILL_FILE_TOOL_NAME.to_string()
            })
        );
    }

    #[test]
    fn rejects_reserved_responses_namespace_tool_name() {
        let tools = vec![ResponseTool::Namespace(NamespaceToolDef {
            name: "safe_namespace".to_string(),
            description: "safe".to_string(),
            tools: vec![NamespaceTool::Custom(ResponseCustomTool {
                name: crate::LIST_SKILL_FILES_TOOL_NAME.to_string(),
                description: None,
                defer_loading: None,
                format: None,
            })],
        })];

        assert_eq!(
            validate_responses_reserved_skill_tool_names(Some(&tools)),
            Err(ReservedSkillToolNameError {
                tool_name: crate::LIST_SKILL_FILES_TOOL_NAME.to_string()
            })
        );
    }

    #[test]
    fn allows_non_reserved_tool_names() {
        let request = ResponsesRequest {
            model: "gpt-5.4".to_string(),
            input: ResponseInput::Text("hello".to_string()),
            tools: Some(vec![ResponseTool::Function(FunctionTool {
                function: Function {
                    name: "lookup_customer".to_string(),
                    description: None,
                    parameters: Value::Object(Default::default()),
                    strict: None,
                },
            })]),
            ..Default::default()
        };

        assert!(validate_responses_reserved_skill_tool_names(request.tools.as_deref()).is_ok());
    }
}
