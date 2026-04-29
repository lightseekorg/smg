//! Utility functions for /v1/responses endpoint

use std::{collections::HashSet, sync::Arc};

use axum::response::Response;
use openai_protocol::{
    common::Tool,
    responses::{ResponseTool, ResponsesRequest, ResponsesResponse},
};
use serde::Serialize;
use serde_json::{to_value, Value};
use smg_data_connector::{
    ConversationItemStorage, ConversationStorage, RequestContext as StorageRequestContext,
    ResponseStorage,
};
use smg_mcp::{McpOrchestrator, McpServerBinding, ResponseFormat};
use tracing::{debug, error, warn};

use crate::{
    routers::{
        common::{
            mcp_utils::{collect_user_function_names, ensure_request_mcp_client},
            persistence_utils::persist_conversation_items,
        },
        error,
    },
    worker::WorkerRegistry,
};

/// Ensure MCP connection succeeds if MCP tools or builtin tools are declared.
///
/// Checks if the request declares MCP tools or builtin tool types
/// (`web_search_preview`, `code_interpreter`, `image_generation`) and,
/// if so, validates that the MCP clients can be created and connected.
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

    // Check for builtin tools that MAY have MCP routing configured.
    //
    // `ImageGeneration` is included here because gpt-oss via the
    // harmony pipeline, and Qwen/Llama via the regular pipeline, both
    // dispatch hosted `image_generation` calls through the same MCP
    // routing path — the only difference is how the tool is advertised in
    // the prompt. Without this arm, the short-circuit below would return
    // `(false, Vec::new())`, the MCP loop would never be entered, and the
    // registered `image_generation` MCP server would receive zero
    // dispatches.
    let has_builtin_tools = tools
        .map(|t| {
            t.iter().any(|tool| {
                matches!(
                    tool,
                    ResponseTool::WebSearchPreview(_)
                        | ResponseTool::CodeInterpreter(_)
                        | ResponseTool::ImageGeneration(_)
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

pub(crate) fn redact_response_for_client(
    response: &mut ResponsesResponse,
    original_request: &ResponsesRequest,
    session: Option<&smg_mcp::McpToolSession<'_>>,
) {
    if let Some(redaction) = ClientRedaction::new(original_request, session) {
        redaction.redact_response(response);
    }
}

pub(crate) fn redact_response_completed_event(
    event: &mut Value,
    original_request: &ResponsesRequest,
    session: Option<&smg_mcp::McpToolSession<'_>>,
) {
    if let (Some(response), Some(redaction)) = (
        event.get_mut("response"),
        ClientRedaction::new(original_request, session),
    ) {
        redaction.redact_response_json(response);
    }
}

pub(crate) fn should_hide_mcp_streaming_tool(
    tool_name: &str,
    response_format: Option<&ResponseFormat>,
    session: &smg_mcp::McpToolSession<'_>,
    user_function_names: &HashSet<String>,
) -> bool {
    matches!(response_format, None | Some(ResponseFormat::Passthrough))
        && session.is_internal_non_builtin_tool(tool_name)
        && !user_function_names.contains(tool_name)
}

struct ClientRedaction<'a, 'session> {
    session: &'a smg_mcp::McpToolSession<'session>,
    user_function_names: HashSet<String>,
}

impl<'a, 'session> ClientRedaction<'a, 'session> {
    fn new(
        original_request: &ResponsesRequest,
        session: Option<&'a smg_mcp::McpToolSession<'session>>,
    ) -> Option<Self> {
        session.map(|session| Self {
            session,
            user_function_names: collect_user_function_names(original_request),
        })
    }

    fn redact_response(&self, response: &mut ResponsesResponse) {
        response
            .output
            .retain(|item| !self.should_hide_serialized_output_item(item));
        response
            .tools
            .retain(|tool| !self.should_hide_serialized_tool(tool));

        let should_hide_tool_choice = serde_json::from_str::<Value>(&response.tool_choice)
            .ok()
            .is_some_and(|tool_choice| self.should_hide_tool_json(&tool_choice));
        if should_hide_tool_choice {
            response.tool_choice = "auto".to_string();
        }
    }

    fn redact_response_json(&self, response: &mut Value) {
        let Some(obj) = response.as_object_mut() else {
            return;
        };

        if let Some(output) = obj.get_mut("output").and_then(Value::as_array_mut) {
            output.retain(|item| !self.should_hide_output_item_json(item));
        }

        if let Some(tools) = obj.get_mut("tools").and_then(Value::as_array_mut) {
            tools.retain(|tool| !self.should_hide_tool_json(tool));
        }

        let should_hide_tool_choice = obj
            .get("tool_choice")
            .is_some_and(|tool_choice| self.should_hide_tool_json(tool_choice));
        if should_hide_tool_choice {
            obj.insert("tool_choice".to_string(), Value::String("auto".to_string()));
        }
    }

    fn should_hide_serialized_output_item(&self, item: &impl Serialize) -> bool {
        to_value(item)
            .ok()
            .is_some_and(|item| self.should_hide_output_item_json(&item))
    }

    fn should_hide_serialized_tool(&self, tool: &impl Serialize) -> bool {
        to_value(tool)
            .ok()
            .is_some_and(|tool| self.should_hide_tool_json(&tool))
    }

    fn should_hide_output_item_json(&self, item: &Value) -> bool {
        self.session
            .should_hide_output_item_json(item, &self.user_function_names)
    }

    fn should_hide_tool_json(&self, tool: &Value) -> bool {
        self.session
            .should_hide_tool_json(tool, &self.user_function_names)
    }
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

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use openai_protocol::responses::{
        McpTool, ResponseInput, ResponseOutputItem, ResponseTool, ResponsesRequest,
        ResponsesResponse,
    };
    use serde_json::{json, Value};
    use smg_mcp::{
        McpConfig, McpOrchestrator, McpServerBinding, McpServerConfig, McpToolSession,
        McpTransport, ResponseFormat, Tool, ToolEntry,
    };

    use super::{
        redact_response_completed_event, redact_response_for_client, should_hide_mcp_streaming_tool,
    };

    fn test_tool(name: &str) -> Tool {
        Tool {
            name: name.to_string().into(),
            title: None,
            description: Some("internal".into()),
            input_schema: json!({"type": "object"})
                .as_object()
                .expect("schema object")
                .clone()
                .into(),
            output_schema: None,
            icons: None,
            annotations: None,
        }
    }

    async fn session_with_server(
        server_name: &str,
        server_label: &str,
        internal: bool,
        tool_entry: ToolEntry,
    ) -> (McpOrchestrator, ResponsesRequest) {
        let orchestrator = McpOrchestrator::new(McpConfig {
            servers: vec![McpServerConfig {
                name: server_name.to_string(),
                transport: McpTransport::Sse {
                    url: "http://localhost:3000/sse".to_string(),
                    token: None,
                    headers: Default::default(),
                },
                proxy: None,
                required: false,
                tools: None,
                builtin_type: None,
                builtin_tool_name: None,
                internal,
            }],
            ..Default::default()
        })
        .await
        .expect("orchestrator");

        orchestrator.tool_inventory().insert_entry(tool_entry);

        let request = ResponsesRequest {
            model: "gpt-5.4".to_string(),
            input: ResponseInput::Text("hello".to_string()),
            tools: Some(vec![ResponseTool::Mcp(McpTool {
                server_url: Some("http://localhost:3000/sse".to_string()),
                authorization: None,
                headers: None,
                server_label: server_label.to_string(),
                server_description: None,
                require_approval: None,
                allowed_tools: None,
                connector_id: None,
                defer_loading: None,
            })]),
            ..Default::default()
        };

        (orchestrator, request)
    }

    async fn session_with_internal_server() -> (McpOrchestrator, ResponsesRequest) {
        session_with_server(
            "internal-server",
            "internal-label",
            true,
            ToolEntry::from_server_tool("internal-server", test_tool("internal_search")),
        )
        .await
    }

    fn binding(server_key: &str, server_label: &str) -> Vec<McpServerBinding> {
        vec![McpServerBinding {
            label: server_label.to_string(),
            server_key: server_key.to_string(),
            allowed_tools: None,
        }]
    }

    #[tokio::test]
    async fn grpc_response_redaction_strips_internal_mcp_artifacts() {
        let (orchestrator, request) = session_with_internal_server().await;
        let session = McpToolSession::new(
            &orchestrator,
            binding("internal-server", "internal-label"),
            "resp_test",
        );
        let mut response = ResponsesResponse::builder("resp_test", &request.model)
            .copy_from_request(&request)
            .output(vec![
                ResponseOutputItem::McpCall {
                    id: "mcp_internal".to_string(),
                    status: "completed".to_string(),
                    approval_request_id: None,
                    arguments: "{\"query\":\"secret args\"}".to_string(),
                    error: None,
                    name: "internal_search".to_string(),
                    output: "{\"trace\":\"secret output\"}".to_string(),
                    server_label: "internal-label".to_string(),
                },
                ResponseOutputItem::FunctionToolCall {
                    id: "fc_public".to_string(),
                    call_id: "call_public".to_string(),
                    name: "public_function".to_string(),
                    arguments: "{\"city\":\"Paris\"}".to_string(),
                    output: None,
                    status: "completed".to_string(),
                },
            ])
            .build();

        redact_response_for_client(&mut response, &request, Some(&session));

        let redacted = serde_json::to_value(&response).expect("response json");
        let serialized = redacted.to_string();
        assert!(!serialized.contains("internal_search"));
        assert!(!serialized.contains("internal-label"));
        assert!(!serialized.contains("secret args"));
        assert!(!serialized.contains("secret output"));
        assert!(serialized.contains("public_function"));

        let mut event = json!({
            "type": "response.completed",
            "response": {
                "id": "resp_test",
                "object": "response",
                "status": "completed",
                "model": request.model.clone(),
                "tools": request.tools,
                "tool_choice": {
                    "type": "mcp",
                    "server_label": "internal-label",
                    "name": "internal_search"
                },
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc_internal",
                        "call_id": "call_internal",
                        "name": "internal_search",
                        "arguments": "{\"query\":\"secret args\"}",
                        "output": "{\"trace\":\"secret output\"}",
                        "status": "completed"
                    },
                    {
                        "type": "function_call",
                        "id": "fc_public",
                        "call_id": "call_public",
                        "name": "public_function",
                        "arguments": "{\"city\":\"Paris\"}",
                        "status": "completed"
                    }
                ]
            }
        });

        redact_response_completed_event(&mut event, &request, Some(&session));

        let serialized = event.to_string();
        assert!(!serialized.contains("internal_search"));
        assert!(!serialized.contains("internal-label"));
        assert!(!serialized.contains("secret args"));
        assert!(!serialized.contains("secret output"));
        assert!(!serialized.contains("localhost:3000"));
        assert!(serialized.contains("public_function"));
        assert_eq!(
            event["response"]["tool_choice"],
            Value::String("auto".to_string())
        );
    }

    #[tokio::test]
    async fn grpc_streaming_visibility_keeps_public_and_hosted_formats_visible() {
        let user_function_names = HashSet::new();
        let (internal_orchestrator, _internal_request) = session_with_internal_server().await;
        let internal_session = McpToolSession::new(
            &internal_orchestrator,
            binding("internal-server", "internal-label"),
            "resp_test",
        );
        assert!(should_hide_mcp_streaming_tool(
            "internal_search",
            Some(&internal_session.tool_response_format("internal_search")),
            &internal_session,
            &user_function_names
        ));

        let (public_orchestrator, _public_request) = session_with_server(
            "public-server",
            "public-label",
            false,
            ToolEntry::from_server_tool("public-server", test_tool("public_search")),
        )
        .await;
        let public_session = McpToolSession::new(
            &public_orchestrator,
            binding("public-server", "public-label"),
            "resp_test",
        );
        assert!(!should_hide_mcp_streaming_tool(
            "public_search",
            Some(&public_session.tool_response_format("public_search")),
            &public_session,
            &user_function_names
        ));

        let (image_orchestrator, _image_request) = session_with_server(
            "image-server",
            "image-label",
            true,
            ToolEntry::from_server_tool("image-server", test_tool("image_generation"))
                .with_response_format(ResponseFormat::ImageGenerationCall),
        )
        .await;
        let image_session = McpToolSession::new(
            &image_orchestrator,
            binding("image-server", "image-label"),
            "resp_test",
        );
        assert_eq!(
            image_session.tool_response_format("image_generation"),
            ResponseFormat::ImageGenerationCall
        );
        assert!(!should_hide_mcp_streaming_tool(
            "image_generation",
            Some(&image_session.tool_response_format("image_generation")),
            &image_session,
            &user_function_names
        ));
    }
}
