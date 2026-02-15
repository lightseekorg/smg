//! MCP tool processing layer for Anthropic Messages API
//!
//! Provides MCP server connection, tool injection, tool execution,
//! and the standard `process_iteration` interface used by both
//! streaming and non-streaming processors.

use std::{collections::HashMap, sync::Arc};

use axum::response::Response;
use openai_protocol::messages::{
    ContentBlock, CreateMessageRequest, CustomTool, InputContentBlock, InputSchema, Message,
    RedactedThinkingBlock, ServerToolUseBlock, StopReason, TextBlock, ThinkingBlock, Tool,
    ToolChoice, ToolResultBlock, ToolResultContent, ToolResultContentBlock, ToolUseBlock,
    WebSearchToolResultBlock,
};
use serde_json::Value;
use smg_mcp::{McpToolSession, ToolEntry, ToolExecutionInput};
use tracing::{debug, error, info, warn};

use crate::{
    observability::metrics::{metrics_labels, Metrics},
    routers::{error as router_error, mcp_utils},
};

// ============================================================================
// Standard I/O types for processor ↔ MCP layer communication
// ============================================================================

/// What processors extract from an LLM response (streaming or non-streaming).
pub(crate) struct IterationResult {
    pub content_blocks: Vec<ContentBlock>,
    pub tool_use_blocks: Vec<ToolUseBlock>,
    pub stop_reason: Option<StopReason>,
}

impl IterationResult {
    /// Construct from a non-streaming `Message`.
    pub(crate) fn from_message(message: &Message) -> Self {
        let tool_use_blocks = extract_tool_calls(&message.content);
        Self {
            content_blocks: message.content.clone(),
            tool_use_blocks,
            stop_reason: message.stop_reason.clone(),
        }
    }
}

/// Continuation data when the tool loop should proceed.
pub(crate) struct ToolLoopContinuation {
    pub mcp_calls: Vec<McpToolCall>,
    pub assistant_blocks: Vec<InputContentBlock>,
    pub tool_result_blocks: Vec<InputContentBlock>,
}

/// Decision from `process_iteration`: stop, continue, or error.
pub(crate) enum ToolLoopAction {
    Done,
    Continue(ToolLoopContinuation),
    Error(String),
}

// ============================================================================
// Tracked MCP tool call
// ============================================================================

/// Tracked MCP tool call for response reconstruction and SSE emission.
pub(crate) struct McpToolCall {
    pub original_id: String,
    pub mcp_id: String,
    pub name: String,
    pub server_name: String,
    pub input: Value,
    pub result_content: String,
    pub is_error: bool,
}

// ============================================================================
// Standard process_iteration interface
// ============================================================================

/// Evaluate one iteration of the MCP tool loop.
///
/// Returns `Done` if there are no tool calls or the stop reason is not `ToolUse`.
/// Returns `Continue` with executed tool results when the loop should proceed.
pub(crate) async fn process_iteration(
    result: &IterationResult,
    session: &McpToolSession<'_>,
    model_id: &str,
) -> ToolLoopAction {
    if result.tool_use_blocks.is_empty() {
        return ToolLoopAction::Done;
    }

    if result.stop_reason != Some(StopReason::ToolUse) {
        let msg = format!(
            "Model returned {} tool_use block(s) but stop_reason is {:?}; tool calls will not be executed",
            result.tool_use_blocks.len(),
            result.stop_reason
        );
        warn!(
            tool_count = result.tool_use_blocks.len(),
            stop_reason = ?result.stop_reason,
            "{}", &msg
        );
        return ToolLoopAction::Error(msg);
    }

    info!(
        tool_count = result.tool_use_blocks.len(),
        "Executing MCP tool calls"
    );

    let (mcp_calls, assistant_blocks, tool_result_blocks) = execute_mcp_tool_calls(
        &result.content_blocks,
        &result.tool_use_blocks,
        session,
        model_id,
    )
    .await;

    ToolLoopAction::Continue(ToolLoopContinuation {
        mcp_calls,
        assistant_blocks,
        tool_result_blocks,
    })
}

// ============================================================================
// MCP connection and tool injection
// ============================================================================

pub(crate) fn extract_tool_calls(content: &[ContentBlock]) -> Vec<ToolUseBlock> {
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
///
/// Request is validated by `ValidatedJson` before reaching the router,
/// so `mcp_server_configs()` is guaranteed to be `Some` here.
pub(crate) async fn ensure_connection(
    request: &mut CreateMessageRequest,
    orchestrator: &Arc<smg_mcp::McpOrchestrator>,
) -> Result<Vec<(String, String)>, Response> {
    let inputs: Vec<mcp_utils::McpServerInput> = request
        .mcp_server_configs()
        .unwrap_or_default()
        .iter()
        .map(|server| mcp_utils::McpServerInput {
            label: server.name.clone(),
            url: Some(server.url.clone()),
            authorization: server.authorization_token.clone(),
            headers: HashMap::new(),
        })
        .collect();

    let mcp_servers = mcp_utils::connect_mcp_servers(orchestrator, &inputs).await;
    if mcp_servers.is_empty() {
        error!("Failed to connect to any MCP servers");
        return Err(router_error::bad_gateway(
            "mcp_connection_failed",
            "Failed to connect to MCP servers. Check server URLs and authorization.",
        ));
    }

    info!(
        server_count = mcp_servers.len(),
        "MCP: connected to MCP servers"
    );

    let allowed_tools = collect_allowed_tools_from_toolsets(&request.tools);

    let session_id = format!("msg_{}", uuid::Uuid::new_v4());
    let session = McpToolSession::new(orchestrator, mcp_servers.clone(), &session_id);

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

// ============================================================================
// Tool execution
// ============================================================================

/// Execute MCP tool calls and return `(mcp_calls, assistant_blocks, tool_result_blocks)`.
async fn execute_mcp_tool_calls(
    content: &[ContentBlock],
    tool_calls: &[ToolUseBlock],
    session: &McpToolSession<'_>,
    model_id: &str,
) -> (
    Vec<McpToolCall>,
    Vec<InputContentBlock>,
    Vec<InputContentBlock>,
) {
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
            is_error: is_error.then_some(true),
            cache_control: None,
        }));
    }

    (mcp_calls, assistant_content_blocks, tool_result_blocks)
}

// ============================================================================
// Response reconstruction
// ============================================================================

/// Replace tool_use blocks with mcp_tool_use/mcp_tool_result pairs in the response.
///
/// All MCP blocks are emitted first, then the original content blocks follow (with any
/// matched ToolUse blocks skipped to avoid duplication). This ensures the final text
/// block appears after all MCP tool activity.
pub(crate) fn rebuild_response_with_mcp_blocks(
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

    // First: emit all MCP tool_use/tool_result pairs
    for call in mcp_calls {
        push_mcp_blocks(&mut new_content, call);
    }

    // Then: append original content, skipping ToolUse blocks already covered by MCP pairs
    for block in std::mem::take(&mut message.content) {
        match &block {
            ContentBlock::ToolUse { id, .. } if call_lookup.contains_key(id.as_str()) => {
                continue;
            }
            _ => new_content.push(block),
        }
    }

    message.content = new_content;
    message
}

// ============================================================================
// Private helpers
// ============================================================================

/// Collect allowed tools filter from `McpToolset` entries in the tools array.
///
/// Returns `None` (no filtering) if any toolset allows all tools.
fn collect_allowed_tools_from_toolsets(tools: &Option<Vec<Tool>>) -> Option<Vec<String>> {
    let tools = tools.as_ref()?;

    let mut all_allowed = Vec::new();
    let mut saw_mcp_toolset = false;
    for tool in tools {
        if let Tool::McpToolset(toolset) = tool {
            saw_mcp_toolset = true;
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
    if saw_mcp_toolset {
        Some(all_allowed)
    } else {
        None
    }
}

/// Convert an MCP `ToolEntry` to an Anthropic `CustomTool`.
fn convert_tool_entry_to_anthropic_tool(entry: &ToolEntry) -> CustomTool {
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
            ContentBlock::Text { text, citations } => {
                blocks.push(InputContentBlock::Text(TextBlock {
                    text: text.clone(),
                    cache_control: None,
                    citations: citations.clone(),
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
            ContentBlock::Thinking {
                thinking,
                signature,
            } => {
                blocks.push(InputContentBlock::Thinking(ThinkingBlock {
                    thinking: thinking.clone(),
                    signature: signature.clone(),
                }));
            }
            ContentBlock::RedactedThinking { data } => {
                blocks.push(InputContentBlock::RedactedThinking(RedactedThinkingBlock {
                    data: data.clone(),
                }));
            }
            ContentBlock::ServerToolUse { id, name, input } => {
                blocks.push(InputContentBlock::ServerToolUse(ServerToolUseBlock {
                    id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                    cache_control: None,
                }));
            }
            ContentBlock::WebSearchToolResult {
                tool_use_id,
                content,
            } => {
                blocks.push(InputContentBlock::WebSearchToolResult(
                    WebSearchToolResultBlock {
                        tool_use_id: tool_use_id.clone(),
                        content: content.clone(),
                        cache_control: None,
                    },
                ));
            }
            // MCP blocks are handled separately by rebuild_response_with_mcp_blocks
            ContentBlock::McpToolUse { .. } | ContentBlock::McpToolResult { .. } => {}
        }
    }
    blocks
}

fn push_mcp_blocks(content: &mut Vec<ContentBlock>, call: &McpToolCall) {
    content.push(ContentBlock::McpToolUse {
        id: call.mcp_id.clone(),
        name: call.name.clone(),
        server_name: call.server_name.clone(),
        input: call.input.clone(),
    });
    content.push(ContentBlock::McpToolResult {
        tool_use_id: call.mcp_id.clone(),
        content: Some(ToolResultContent::Blocks(vec![
            ToolResultContentBlock::Text(TextBlock {
                text: call.result_content.clone(),
                cache_control: None,
                citations: None,
            }),
        ])),
        is_error: call.is_error.then_some(true),
    });
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
