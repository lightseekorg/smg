//! Utility functions for /v1/responses endpoint

use std::{collections::HashSet, sync::Arc};

use axum::response::Response;
use bytes::Bytes;
use openai_protocol::{
    common::{Tool, ToolReference},
    responses::{
        BuiltInToolChoiceType, ResponseOutputItem, ResponseTool, ResponsesRequest,
        ResponsesResponse, ResponsesToolChoice, ToolChoiceOptions,
    },
};
use serde_json::to_value;
use smg_data_connector::{
    ConversationItemStorage, ConversationStorage, RequestContext as StorageRequestContext,
    ResponseStorage,
};
use smg_mcp::{McpOrchestrator, McpServerBinding, McpToolSession};
use tokio::sync::mpsc;
use tracing::{debug, error, warn};

use super::streaming::ResponseStreamEventEmitter;
use crate::{
    routers::{
        common::{
            mcp_utils::ensure_request_mcp_client, persistence_utils::persist_conversation_items,
        },
        error,
    },
    worker::WorkerRegistry,
};

/// Ensure MCP connection succeeds if MCP tools or builtin tools are declared
///
/// Checks if request declares MCP tools or builtin tool types (web_search_preview,
/// code_interpreter), and if so, validates that the MCP clients can be created
/// and connected.
///
/// Returns Ok((has_mcp_tools, mcp_servers)) on success.
pub(crate) async fn ensure_mcp_connection(
    mcp_orchestrator: &Arc<McpOrchestrator>,
    tools: Option<&[ResponseTool]>,
) -> Result<(bool, Vec<McpServerBinding>), Response> {
    // Check for explicit MCP tools (must error if connection fails)
    let has_explicit_mcp_tools = tools
        .map(|t| t.iter().any(|tool| matches!(tool, ResponseTool::Mcp(_))))
        .unwrap_or(false);

    // Check for builtin tools that MAY have MCP routing configured
    let has_builtin_tools = tools
        .map(|t| {
            t.iter().any(|tool| {
                matches!(
                    tool,
                    ResponseTool::WebSearchPreview(_) | ResponseTool::CodeInterpreter(_)
                )
            })
        })
        .unwrap_or(false);

    // Only process if we have MCP or builtin tools
    if !has_explicit_mcp_tools && !has_builtin_tools {
        return Ok((false, Vec::new()));
    }

    if let Some(tools) = tools {
        // TODO: Thread real request headers through the gRPC responses path if/when
        // gRPC MCP flows need the same forwarded-header preservation contract.
        match ensure_request_mcp_client(mcp_orchestrator, tools).await {
            Some(mcp_servers) => {
                return Ok((true, mcp_servers));
            }
            None => {
                // No MCP servers available
                if has_explicit_mcp_tools {
                    // Explicit MCP tools MUST have working connections
                    error!(
                        function = "ensure_mcp_connection",
                        "Failed to connect to MCP servers"
                    );
                    return Err(error::failed_dependency(
                        "connect_mcp_server_failed",
                        "Failed to connect to MCP servers. Check server_url and authorization.",
                    ));
                }
                // Builtin tools without MCP routing - pass through to model
                debug!(
                    function = "ensure_mcp_connection",
                    "No MCP routing configured for builtin tools, passing through to model"
                );
                return Ok((false, Vec::new()));
            }
        }
    }

    Ok((false, Vec::new()))
}

/// Validate that workers are available for the requested model
pub(crate) fn validate_worker_availability(
    worker_registry: &Arc<WorkerRegistry>,
    model: &str,
) -> Option<Response> {
    let available_models = worker_registry.get_models();

    if !available_models.contains(&model.to_string()) {
        return Some(error::model_not_found(model));
    }

    None
}

/// Extract function tools from ResponseTools
///
/// This utility consolidates the logic for extracting tools with schemas from ResponseTools.
/// It's used by both Harmony and Regular routers for different purposes:
///
/// - **Harmony router**: Extracts function tools because MCP tools are exposed to the model as
///   function tools (via `convert_mcp_tools_to_response_tools()`), and those are used to
///   generate structural constraints in the Harmony preparation stage.
///
/// - **Regular router**: Extracts function tools during the initial conversion from
///   ResponsesRequest to ChatCompletionRequest. MCP tools are merged later by the tool loop.
pub(crate) fn extract_tools_from_response_tools(
    response_tools: Option<&[ResponseTool]>,
) -> Vec<Tool> {
    let Some(tools) = response_tools else {
        return Vec::new();
    };

    tools
        .iter()
        .filter_map(|rt| match rt {
            ResponseTool::Function(ft) => Some(Tool {
                tool_type: "function".to_string(),
                function: ft.function.clone(),
            }),
            _ => None,
        })
        .collect()
}

/// Persist response to storage if store=true
///
/// Common helper function to avoid duplication across sync and streaming paths
/// in both harmony and regular responses implementations.
pub(crate) async fn persist_response_if_needed(
    conversation_storage: Arc<dyn ConversationStorage>,
    conversation_item_storage: Arc<dyn ConversationItemStorage>,
    response_storage: Arc<dyn ResponseStorage>,
    response: &ResponsesResponse,
    original_request: &ResponsesRequest,
    request_context: Option<StorageRequestContext>,
) {
    if !original_request.store.unwrap_or(true) {
        return;
    }

    if let Ok(response_json) = to_value(response) {
        if let Err(e) = persist_conversation_items(
            conversation_storage,
            conversation_item_storage,
            response_storage,
            &response_json,
            original_request,
            request_context,
        )
        .await
        {
            warn!("Failed to persist response: {}", e);
        } else {
            debug!("Persisted response: {}", response.id);
        }
    }
}

/// Retain only client-visible output items based on session hide policy.
///
/// This keeps non-streaming redaction behavior consistent across regular and
/// harmony response paths.
pub(crate) fn retain_client_visible_output_items(
    output: &mut Vec<ResponseOutputItem>,
    session: &McpToolSession<'_>,
    user_function_names: &HashSet<String>,
) {
    output.retain(|item| {
        let Ok(json) = to_value(item) else {
            return true;
        };
        !session.should_hide_output_item_json(&json, user_function_names)
    });
}

fn retain_client_visible_tools_in_place(
    tools: &mut Vec<ResponseTool>,
    session: &McpToolSession<'_>,
    user_function_names: &HashSet<String>,
) {
    tools.retain(|tool| {
        let Ok(json) = to_value(tool) else {
            return true;
        };
        !session.should_hide_tool_json(&json, user_function_names)
    });
}

fn has_function_tool(tools: &[ResponseTool], tool_name: &str) -> bool {
    tools.iter().any(|tool| {
        matches!(
            tool,
            ResponseTool::Function(function_tool) if function_tool.function.name == tool_name
        )
    })
}

fn has_builtin_tool(tools: &[ResponseTool], tool_type: BuiltInToolChoiceType) -> bool {
    tools.iter().any(|tool| match tool_type {
        BuiltInToolChoiceType::FileSearch => matches!(tool, ResponseTool::FileSearch(_)),
        BuiltInToolChoiceType::WebSearch => matches!(tool, ResponseTool::WebSearch(_)),
        BuiltInToolChoiceType::WebSearchPreview
        | BuiltInToolChoiceType::WebSearchPreview20250311 => {
            matches!(tool, ResponseTool::WebSearchPreview(_))
        }
        BuiltInToolChoiceType::ImageGeneration => {
            matches!(tool, ResponseTool::ImageGeneration(_))
        }
        BuiltInToolChoiceType::ComputerUsePreview => {
            matches!(tool, ResponseTool::ComputerUsePreview(_))
        }
        BuiltInToolChoiceType::CodeInterpreter => {
            matches!(tool, ResponseTool::CodeInterpreter(_))
        }
    })
}

fn has_mcp_tool(tools: &[ResponseTool], server_label: &str, name: Option<&str>) -> bool {
    tools.iter().any(|tool| match tool {
        ResponseTool::Mcp(mcp_tool) if mcp_tool.server_label == server_label => {
            if let Some(name) = name {
                match &mcp_tool.allowed_tools {
                    Some(allowed_tools) => allowed_tools.iter().any(|tool_name| tool_name == name),
                    None => true,
                }
            } else {
                true
            }
        }
        _ => false,
    })
}

fn has_custom_tool(tools: &[ResponseTool], custom_name: &str) -> bool {
    tools.iter().any(
        |tool| matches!(tool, ResponseTool::Custom(custom_tool) if custom_tool.name == custom_name),
    )
}

fn has_allowed_tool_reference(tools: &[ResponseTool], tool_reference: &ToolReference) -> bool {
    match tool_reference {
        ToolReference::Function { name } => has_function_tool(tools, name),
        ToolReference::Mcp { server_label, name } => {
            has_mcp_tool(tools, server_label, name.as_deref())
        }
        ToolReference::FileSearch => has_builtin_tool(tools, BuiltInToolChoiceType::FileSearch),
        ToolReference::WebSearchPreview => {
            has_builtin_tool(tools, BuiltInToolChoiceType::WebSearchPreview)
        }
        ToolReference::ComputerUsePreview => {
            has_builtin_tool(tools, BuiltInToolChoiceType::ComputerUsePreview)
        }
        ToolReference::CodeInterpreter => {
            has_builtin_tool(tools, BuiltInToolChoiceType::CodeInterpreter)
        }
        ToolReference::ImageGeneration => {
            has_builtin_tool(tools, BuiltInToolChoiceType::ImageGeneration)
        }
    }
}

fn is_tool_choice_compatible(
    choice: &ResponsesToolChoice,
    available_tools: &[ResponseTool],
) -> bool {
    match choice {
        ResponsesToolChoice::Options(_) => true,
        ResponsesToolChoice::Types { tool_type } => has_builtin_tool(available_tools, *tool_type),
        ResponsesToolChoice::Function(function_choice) => {
            has_function_tool(available_tools, &function_choice.name)
        }
        ResponsesToolChoice::AllowedTools {
            tools: allowed_tools,
            ..
        } => {
            !allowed_tools.is_empty()
                && allowed_tools.iter().all(|tool_reference| {
                    has_allowed_tool_reference(available_tools, tool_reference)
                })
        }
        ResponsesToolChoice::Mcp {
            server_label, name, ..
        } => has_mcp_tool(available_tools, server_label, name.as_deref()),
        ResponsesToolChoice::Custom { name, .. } => has_custom_tool(available_tools, name),
        ResponsesToolChoice::ApplyPatch { .. } => available_tools
            .iter()
            .any(|tool| matches!(tool, ResponseTool::ApplyPatch)),
        ResponsesToolChoice::Shell { .. } => available_tools
            .iter()
            .any(|tool| matches!(tool, ResponseTool::Shell(_))),
    }
}

fn normalize_request_tool_choice(
    tool_choice: &mut Option<ResponsesToolChoice>,
    tools: &[ResponseTool],
) {
    if tools.is_empty() {
        *tool_choice = None;
        return;
    }

    if let Some(choice) = tool_choice.as_ref() {
        if !is_tool_choice_compatible(choice, tools) {
            *tool_choice = Some(ResponsesToolChoice::Options(ToolChoiceOptions::Auto));
        }
    }
}

fn normalize_response_tool_choice(tool_choice: &mut String, tools: &[ResponseTool]) {
    if tools.is_empty() {
        *tool_choice = "auto".to_string();
        return;
    }

    let Ok(choice) = serde_json::from_str::<ResponsesToolChoice>(tool_choice) else {
        *tool_choice = "auto".to_string();
        return;
    };

    if !is_tool_choice_compatible(&choice, tools) {
        *tool_choice = "auto".to_string();
    }
}

/// Retain only client-visible tools and normalize request `tool_choice`.
///
/// Used on streaming paths before copying original request fields into
/// `response.completed` payloads.
pub(crate) fn retain_client_visible_request_tools(
    request: &mut ResponsesRequest,
    session: &McpToolSession<'_>,
    user_function_names: &HashSet<String>,
) {
    if let Some(tools) = request.tools.as_mut() {
        retain_client_visible_tools_in_place(tools, session, user_function_names);
        normalize_request_tool_choice(&mut request.tool_choice, tools);
        if tools.is_empty() {
            request.tools = None;
        }
    } else {
        request.tool_choice = None;
    }
}

/// Retain only client-visible tools and normalize response `tool_choice`.
///
/// Keeps gRPC non-streaming responses aligned with session hide policy.
pub(crate) fn retain_client_visible_response_tools(
    response: &mut ResponsesResponse,
    session: &McpToolSession<'_>,
    user_function_names: &HashSet<String>,
) {
    retain_client_visible_tools_in_place(&mut response.tools, session, user_function_names);
    normalize_response_tool_choice(&mut response.tool_choice, &response.tools);
}

/// Emit the visible `mcp_list_tools` streaming sequence for all server bindings
/// in the session.
///
/// Visibility is gated by `session.should_emit_streaming_mcp_list_tools` so
/// hidden bindings do not appear in client-facing streams.
pub(crate) fn emit_visible_mcp_list_tools_sequence(
    session: &McpToolSession<'_>,
    emitter: &mut ResponseStreamEventEmitter,
    tx: &mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
) -> Result<(), String> {
    for binding in session.mcp_servers() {
        if !session.should_emit_streaming_mcp_list_tools(&binding.label) {
            continue;
        }
        let tools_for_server = session.list_tools_for_server(&binding.server_key);
        emitter.emit_mcp_list_tools_sequence(&binding.label, &tools_for_server, tx)?;
    }
    Ok(())
}
