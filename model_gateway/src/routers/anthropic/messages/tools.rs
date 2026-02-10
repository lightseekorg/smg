//! MCP tool integration for Messages API

use std::{collections::HashMap, sync::Arc};

use axum::response::Response;
use serde_json::Value;
use tracing::{debug, error, info, warn};

use crate::{
    mcp::{McpToolSession, ToolExecutionInput},
    observability::metrics::{metrics_labels, Metrics},
    protocols::messages::{
        ContentBlock, CreateMessageRequest, CustomTool, InputContentBlock, InputSchema, Message,
        TextBlock, Tool, ToolChoice, ToolResultBlock, ToolResultContent, ToolUseBlock,
    },
    routers::{error as router_error, mcp_utils},
};

/// Tracked MCP tool call for response reconstruction.
pub(super) struct McpToolCall {
    pub original_id: String,
    pub mcp_id: String,
    pub name: String,
    pub server_name: String,
    pub input: Value,
    pub result_content: String,
    pub is_error: bool,
}

pub(super) fn extract_tool_calls(content: &[ContentBlock]) -> Vec<ToolUseBlock> {
    content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::ToolUse { id, name, input } => Some(ToolUseBlock {
                id: id.clone(),
                name: name.clone(),
                input: input.clone(),
                cache_control: None,
            }),
            _ => None,
        })
        .collect()
}

/// Connect MCP servers, inject tools into the request, and return connected servers.
pub(crate) async fn ensure_mcp_connection(
    request: &mut CreateMessageRequest,
    orchestrator: &Arc<crate::mcp::McpOrchestrator>,
) -> Result<Vec<(String, String)>, Response> {
    let mcp_server_configs = match &request.mcp_servers {
        Some(servers) if !servers.is_empty() => servers.clone(),
        _ => {
            return Err(router_error::bad_request(
                "invalid_request",
                "mcp_servers field is empty or missing",
            ));
        }
    };

    let inputs: Vec<mcp_utils::McpServerInput> = mcp_server_configs
        .iter()
        .map(|server| mcp_utils::McpServerInput {
            label: server.name.clone(),
            url: server.url.clone(),
            authorization: server.authorization_token.clone(),
            headers: HashMap::new(),
        })
        .collect();

    let mcp_servers = match mcp_utils::connect_mcp_servers(orchestrator, &inputs).await {
        Some(servers) => servers,
        None => {
            error!("Failed to connect to any MCP servers");
            return Err(router_error::bad_gateway(
                "mcp_connection_failed",
                "Failed to connect to MCP servers. Check server URLs and authorization.",
            ));
        }
    };

    info!(
        server_count = mcp_servers.len(),
        "MCP: connected to MCP servers"
    );

    let allowed_tools = collect_allowed_tools_from_toolsets(&request.tools);

    let request_id = format!("msg_{}", uuid::Uuid::new_v4());
    let session = McpToolSession::new(orchestrator, mcp_servers.clone(), &request_id);

    inject_mcp_tools_into_request(request, &session, &allowed_tools);

    Ok(mcp_servers)
}

/// Strip `mcp_servers` from request and inject MCP tools as regular tools.
fn inject_mcp_tools_into_request(
    request: &mut CreateMessageRequest,
    session: &McpToolSession<'_>,
    allowed_tools_filter: &Option<Vec<String>>,
) {
    request.mcp_servers = None;

    let mut tools: Vec<Tool> = request
        .tools
        .take()
        .unwrap_or_default()
        .into_iter()
        .filter(|t| !matches!(t, Tool::McpToolset(_)))
        .collect();

    for entry in session.mcp_tools() {
        let tool_name = entry.tool.name.to_string();

        if let Some(allowed) = allowed_tools_filter {
            if !allowed.contains(&tool_name) {
                continue;
            }
        }

        tools.push(Tool::Custom(convert_tool_entry_to_anthropic_tool(entry)));
    }

    if !tools.is_empty() {
        request.tools = Some(tools);
        if request.tool_choice.is_none() {
            request.tool_choice = Some(ToolChoice::Auto {
                disable_parallel_tool_use: None,
            });
        }
    }
}

/// Execute MCP tool calls and return `(mcp_calls, assistant_blocks, tool_result_blocks)`.
pub(super) async fn execute_mcp_tool_calls(
    content: &[ContentBlock],
    tool_calls: &[ToolUseBlock],
    session: &McpToolSession<'_>,
    model_id: &str,
) -> Result<
    (
        Vec<McpToolCall>,
        Vec<InputContentBlock>,
        Vec<InputContentBlock>,
    ),
    String,
> {
    let assistant_content_blocks = build_assistant_content_blocks(content);

    let mut mcp_calls = Vec::new();
    let mut tool_result_blocks: Vec<InputContentBlock> = Vec::new();

    for tool_call in tool_calls {
        let server_name = session.resolve_tool_server_label(&tool_call.name);

        debug!(
            tool = %tool_call.name,
            server = %server_name,
            "Executing MCP tool call"
        );

        let input = ToolExecutionInput {
            call_id: tool_call.id.clone(),
            tool_name: tool_call.name.clone(),
            arguments: tool_call.input.clone(),
        };

        let output = session.execute_tool(input).await;

        Metrics::record_mcp_tool_duration(model_id, &output.tool_name, output.duration);
        Metrics::record_mcp_tool_call(
            model_id,
            &output.tool_name,
            if output.is_error {
                metrics_labels::RESULT_ERROR
            } else {
                metrics_labels::RESULT_SUCCESS
            },
        );

        let result_content = extract_output_from_value(&output.output);
        let is_error = output.is_error;

        if is_error {
            let err_msg = output.error_message.as_deref().unwrap_or("unknown error");
            warn!(tool = %tool_call.name, error = %err_msg, "MCP tool execution failed");
        }

        let mcp_id = format!("mcptoolu_{}", tool_call.id.trim_start_matches("toolu_"));

        mcp_calls.push(McpToolCall {
            original_id: tool_call.id.clone(),
            mcp_id: mcp_id.clone(),
            name: tool_call.name.clone(),
            server_name,
            input: tool_call.input.clone(),
            result_content: result_content.clone(),
            is_error,
        });

        tool_result_blocks.push(InputContentBlock::ToolResult(ToolResultBlock {
            tool_use_id: tool_call.id.clone(),
            content: Some(ToolResultContent::String(result_content)),
            is_error: if is_error { Some(true) } else { None },
            cache_control: None,
        }));
    }

    Ok((mcp_calls, assistant_content_blocks, tool_result_blocks))
}

/// Collect allowed tools filter from `McpToolset` entries in the tools array.
///
/// Returns `None` (no filtering) if any toolset allows all tools.
fn collect_allowed_tools_from_toolsets(tools: &Option<Vec<Tool>>) -> Option<Vec<String>> {
    let tools = tools.as_ref()?;

    let mut all_allowed = Vec::new();
    for tool in tools {
        if let Tool::McpToolset(toolset) = tool {
            let default_enabled = toolset
                .default_config
                .as_ref()
                .and_then(|c| c.enabled)
                .unwrap_or(true);

            match &toolset.configs {
                Some(configs) => {
                    for (tool_name, config) in configs {
                        let enabled = config.enabled.unwrap_or(default_enabled);
                        if enabled {
                            all_allowed.push(tool_name.clone());
                        }
                    }
                }
                None if default_enabled => {
                    // This server allows all tools — skip global filtering
                    return None;
                }
                None => {}
            }
        }
    }
    if all_allowed.is_empty() {
        None
    } else {
        Some(all_allowed)
    }
}

/// Convert an MCP `ToolEntry` to an Anthropic `CustomTool`.
fn convert_tool_entry_to_anthropic_tool(entry: &crate::mcp::ToolEntry) -> CustomTool {
    let schema_map = (*entry.tool.input_schema).clone();
    let schema_type = schema_map
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("object")
        .to_string();

    let properties = schema_map
        .get("properties")
        .and_then(|v| v.as_object())
        .map(|obj| obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect());

    let required = schema_map
        .get("required")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        });

    let additional: HashMap<String, Value> = schema_map
        .into_iter()
        .filter(|(k, _)| k != "type" && k != "properties" && k != "required")
        .collect();

    let input_schema = InputSchema {
        schema_type,
        properties,
        required,
        additional,
    };

    CustomTool {
        name: entry.tool.name.to_string(),
        tool_type: None,
        description: entry.tool.description.as_ref().map(|d| d.to_string()),
        input_schema,
        cache_control: None,
    }
}

fn build_assistant_content_blocks(content: &[ContentBlock]) -> Vec<InputContentBlock> {
    let mut blocks = Vec::new();
    for block in content {
        match block {
            ContentBlock::Text { text, .. } => {
                blocks.push(InputContentBlock::Text(TextBlock {
                    text: text.clone(),
                    cache_control: None,
                    citations: None,
                }));
            }
            ContentBlock::ToolUse { id, name, input } => {
                blocks.push(InputContentBlock::ToolUse(ToolUseBlock {
                    id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                    cache_control: None,
                }));
            }
            _ => {}
        }
    }
    blocks
}

/// Replace tool_use blocks with mcp_tool_use/mcp_tool_result pairs in the response.
pub(super) fn rebuild_response_with_mcp_blocks(
    mut message: Message,
    mcp_calls: &[McpToolCall],
) -> Message {
    if mcp_calls.is_empty() {
        return message;
    }

    let call_lookup: HashMap<&str, &McpToolCall> = mcp_calls
        .iter()
        .map(|c| (c.original_id.as_str(), c))
        .collect();

    let mut new_content: Vec<ContentBlock> = Vec::new();

    for call in mcp_calls {
        new_content.push(ContentBlock::McpToolUse {
            id: call.mcp_id.clone(),
            name: call.name.clone(),
            server_name: call.server_name.clone(),
            input: call.input.clone(),
        });
        new_content.push(ContentBlock::McpToolResult {
            tool_use_id: call.mcp_id.clone(),
            content: Some(ToolResultContent::Blocks(vec![
                crate::protocols::messages::ToolResultContentBlock::Text(TextBlock {
                    text: call.result_content.clone(),
                    cache_control: None,
                    citations: None,
                }),
            ])),
            is_error: if call.is_error {
                Some(true)
            } else {
                Some(false)
            },
        });
    }

    for block in message.content {
        match &block {
            ContentBlock::ToolUse { id, .. } => {
                if call_lookup.contains_key(id.as_str()) {
                    continue;
                }
                new_content.push(block);
            }
            _ => new_content.push(block),
        }
    }

    message.content = new_content;
    message
}

/// Extract output content string from a `ToolExecutionOutput` value.
fn extract_output_from_value(output: &Value) -> String {
    if let Some(text) = output.as_str() {
        text.to_string()
    } else if let Some(arr) = output.as_array() {
        arr.iter()
            .filter_map(|item| item.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        output.to_string()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use serde_json::json;

    use super::*;
    use crate::protocols::messages::StopReason;

    #[test]
    fn test_extract_tool_calls() {
        let content = vec![
            ContentBlock::Text {
                text: "Let me check that.".to_string(),
                citations: None,
            },
            ContentBlock::ToolUse {
                id: "toolu_01".to_string(),
                name: "get_weather".to_string(),
                input: json!({"location": "SF"}),
            },
        ];

        let calls = extract_tool_calls(&content);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].id, "toolu_01");
    }

    #[test]
    fn test_rebuild_response_with_mcp_blocks() {
        let message = Message {
            id: "msg_01".to_string(),
            message_type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![ContentBlock::Text {
                text: "Here is the answer.".to_string(),
                citations: None,
            }],
            model: "claude-3-haiku".to_string(),
            stop_reason: Some(StopReason::EndTurn),
            stop_sequence: None,
            usage: crate::protocols::messages::Usage {
                input_tokens: 100,
                output_tokens: 50,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
                cache_creation: None,
                server_tool_use: None,
                service_tier: None,
            },
        };

        let mcp_calls = vec![McpToolCall {
            original_id: "toolu_01".to_string(),
            mcp_id: "mcptoolu_01".to_string(),
            name: "ask_question".to_string(),
            server_name: "deepwiki".to_string(),
            input: json!({"question": "What is MCP?"}),
            result_content: "MCP is the Model Context Protocol.".to_string(),
            is_error: false,
        }];

        let result = rebuild_response_with_mcp_blocks(message, &mcp_calls);

        assert_eq!(result.content.len(), 3);
        assert!(matches!(result.content[0], ContentBlock::McpToolUse { .. }));
        assert!(matches!(
            result.content[1],
            ContentBlock::McpToolResult { .. }
        ));
        assert!(matches!(result.content[2], ContentBlock::Text { .. }));

        if let ContentBlock::McpToolUse {
            id,
            name,
            server_name,
            ..
        } = &result.content[0]
        {
            assert_eq!(id, "mcptoolu_01");
            assert_eq!(name, "ask_question");
            assert_eq!(server_name, "deepwiki");
        }

        if let ContentBlock::McpToolResult {
            tool_use_id,
            is_error,
            ..
        } = &result.content[1]
        {
            assert_eq!(tool_use_id, "mcptoolu_01");
            assert_eq!(*is_error, Some(false));
        }
    }

    #[test]
    fn test_rebuild_response_no_mcp_calls() {
        let message = Message {
            id: "msg_01".to_string(),
            message_type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![ContentBlock::Text {
                text: "Hello".to_string(),
                citations: None,
            }],
            model: "claude-3-haiku".to_string(),
            stop_reason: Some(StopReason::EndTurn),
            stop_sequence: None,
            usage: crate::protocols::messages::Usage {
                input_tokens: 10,
                output_tokens: 5,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
                cache_creation: None,
                server_tool_use: None,
                service_tier: None,
            },
        };

        let result = rebuild_response_with_mcp_blocks(message, &[]);
        assert_eq!(result.content.len(), 1);
        assert!(matches!(result.content[0], ContentBlock::Text { .. }));
    }

    #[test]
    fn test_extract_output_from_value_string() {
        let value = Value::String("The answer is 42".to_string());
        assert_eq!(extract_output_from_value(&value), "The answer is 42");
    }

    #[test]
    fn test_extract_output_from_value_array() {
        let value = json!([
            {"type": "text", "text": "Line 1"},
            {"type": "text", "text": "Line 2"}
        ]);
        assert_eq!(extract_output_from_value(&value), "Line 1\nLine 2");
    }

    #[test]
    fn test_extract_output_from_value_object() {
        let value = json!({"result": 42});
        assert_eq!(extract_output_from_value(&value), "{\"result\":42}");
    }

    #[test]
    fn test_build_assistant_content_blocks() {
        let content = vec![
            ContentBlock::Text {
                text: "Let me help.".to_string(),
                citations: None,
            },
            ContentBlock::ToolUse {
                id: "toolu_01".to_string(),
                name: "search".to_string(),
                input: json!({"q": "test"}),
            },
        ];

        let blocks = build_assistant_content_blocks(&content);
        assert_eq!(blocks.len(), 2);
        assert!(matches!(blocks[0], InputContentBlock::Text(_)));
        assert!(matches!(blocks[1], InputContentBlock::ToolUse(_)));
    }

    #[test]
    fn test_mcp_toolset_deserializes_as_tool() {
        let json_str = r#"{"type": "mcp_toolset", "mcp_server_name": "brave"}"#;
        let tool: Tool = serde_json::from_str(json_str).unwrap();
        assert!(matches!(tool, Tool::McpToolset(_)));
        if let Tool::McpToolset(ts) = &tool {
            assert_eq!(ts.mcp_server_name, "brave");
            assert_eq!(ts.toolset_type, "mcp_toolset");
        }
    }

    #[test]
    fn test_custom_tool_still_deserializes() {
        let json_str = r#"{
            "name": "get_weather",
            "description": "Get weather",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }"#;
        let tool: Tool = serde_json::from_str(json_str).unwrap();
        assert!(matches!(tool, Tool::Custom(_)));
        if let Tool::Custom(ct) = &tool {
            assert_eq!(ct.name, "get_weather");
        }
    }

    #[test]
    fn test_collect_allowed_tools_with_configs() {
        use crate::protocols::messages::{McpToolConfig, McpToolDefaultConfig, McpToolset};

        let tools = vec![Tool::McpToolset(McpToolset {
            toolset_type: "mcp_toolset".to_string(),
            mcp_server_name: "brave".to_string(),
            default_config: Some(McpToolDefaultConfig {
                enabled: Some(false),
                defer_loading: None,
            }),
            configs: Some(HashMap::from([
                (
                    "brave_search".to_string(),
                    McpToolConfig {
                        enabled: Some(true),
                        defer_loading: None,
                    },
                ),
                (
                    "brave_local".to_string(),
                    McpToolConfig {
                        enabled: None,
                        defer_loading: None,
                    },
                ),
            ])),
            cache_control: None,
        })];

        let allowed = collect_allowed_tools_from_toolsets(&Some(tools));
        let allowed = allowed.unwrap();
        assert!(allowed.contains(&"brave_search".to_string()));
        assert!(!allowed.contains(&"brave_local".to_string()));
    }

    #[test]
    fn test_collect_allowed_tools_no_configs() {
        use crate::protocols::messages::McpToolset;

        let tools = vec![Tool::McpToolset(McpToolset {
            toolset_type: "mcp_toolset".to_string(),
            mcp_server_name: "brave".to_string(),
            default_config: None,
            configs: None,
            cache_control: None,
        })];

        // No configs and default enabled → no filtering (allow all)
        let allowed = collect_allowed_tools_from_toolsets(&Some(tools));
        assert!(allowed.is_none());
    }

    #[test]
    fn test_collect_allowed_tools_multi_server_allow_all_wins() {
        use crate::protocols::messages::{McpToolConfig, McpToolDefaultConfig, McpToolset};

        // Server A has explicit configs, Server B allows all
        let tools = vec![
            Tool::McpToolset(McpToolset {
                toolset_type: "mcp_toolset".to_string(),
                mcp_server_name: "server_a".to_string(),
                default_config: Some(McpToolDefaultConfig {
                    enabled: Some(false),
                    defer_loading: None,
                }),
                configs: Some(HashMap::from([(
                    "tool_a".to_string(),
                    McpToolConfig {
                        enabled: Some(true),
                        defer_loading: None,
                    },
                )])),
                cache_control: None,
            }),
            Tool::McpToolset(McpToolset {
                toolset_type: "mcp_toolset".to_string(),
                mcp_server_name: "server_b".to_string(),
                default_config: None,
                configs: None,
                cache_control: None,
            }),
        ];

        // Server B allows all → no global filtering
        let allowed = collect_allowed_tools_from_toolsets(&Some(tools));
        assert!(allowed.is_none());
    }
}
