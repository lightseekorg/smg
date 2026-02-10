//! MCP tool integration for Messages API

use std::collections::HashMap;

use axum::response::Response;
use serde_json::Value;
use tracing::{debug, error, info, warn};

use crate::{
    mcp::{McpToolSession, ToolExecutionInput},
    observability::metrics::{metrics_labels, Metrics},
    protocols::{
        messages::{
            ContentBlock, CreateMessageRequest, CustomTool, InputContentBlock, InputSchema,
            Message, TextBlock, Tool, ToolChoice, ToolResultBlock, ToolResultContent, ToolUseBlock,
        },
        responses::{ResponseTool, ResponseToolType},
    },
    routers::{
        anthropic::context::MessagesContext, error as router_error,
        mcp_utils::ensure_request_mcp_client,
    },
};

/// Tracked MCP tool call for response reconstruction.
pub struct McpToolCall {
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

/// Convert Anthropic `McpServerConfig` to `ResponseTool` for shared MCP connection utility.
fn to_response_tools(
    mcp_servers: &[crate::protocols::messages::McpServerConfig],
) -> Vec<ResponseTool> {
    mcp_servers
        .iter()
        .map(|server| ResponseTool {
            r#type: ResponseToolType::Mcp,
            server_url: Some(server.url.clone()),
            server_label: Some(server.name.clone()),
            authorization: server.authorization_token.clone(),
            ..Default::default()
        })
        .collect()
}

/// Connect MCP servers, store in context, and inject tools into the request.
pub(crate) async fn ensure_mcp_connection(
    request: &mut CreateMessageRequest,
    messages_ctx: &MessagesContext,
) -> Result<(), Response> {
    let mcp_server_configs = match &request.mcp_servers {
        Some(servers) if !servers.is_empty() => servers.clone(),
        _ => {
            return Err(router_error::bad_request(
                "invalid_request",
                "mcp_servers field is empty or missing",
            ));
        }
    };

    let orchestrator = &messages_ctx.mcp_orchestrator;

    let response_tools = to_response_tools(&mcp_server_configs);
    let mcp_servers = match ensure_request_mcp_client(orchestrator, &response_tools).await {
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

    *messages_ctx.requested_servers.write().unwrap() = mcp_servers.clone();

    let allowed_tools = collect_allowed_tools_from_toolsets(&request.tools);

    let request_id = format!("msg_{}", uuid::Uuid::new_v4());
    let session = McpToolSession::new(orchestrator, mcp_servers, &request_id);

    inject_mcp_tools_into_request(request, &session, &allowed_tools);

    Ok(())
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

/// Execute MCP tool calls from a response message and return
/// `(mcp_calls, assistant_blocks, tool_result_blocks)`.
pub async fn execute_mcp_tool_calls(
    message: &Message,
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
    let tool_calls = extract_tool_calls(&message.content);

    let assistant_content_blocks = build_assistant_content_blocks(&message.content);

    let mut mcp_calls = Vec::new();
    let mut tool_result_blocks: Vec<InputContentBlock> = Vec::new();

    for tool_call in &tool_calls {
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
fn collect_allowed_tools_from_toolsets(tools: &Option<Vec<Tool>>) -> Option<Vec<String>> {
    let tools = match tools {
        Some(t) => t,
        None => return None,
    };

    let mut all_allowed = Vec::new();
    for tool in tools {
        if let Tool::McpToolset(toolset) = tool {
            let default_enabled = toolset
                .default_config
                .as_ref()
                .and_then(|c| c.enabled)
                .unwrap_or(true);

            if let Some(ref configs) = toolset.configs {
                for (tool_name, config) in configs {
                    let enabled = config.enabled.unwrap_or(default_enabled);
                    if enabled {
                        all_allowed.push(tool_name.clone());
                    }
                }
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
