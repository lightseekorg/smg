// OpenAI Realtime Conversation API types
// https://platform.openai.com/docs/api-reference/realtime
//
// Session configuration and audio types live in `realtime_session`.
// Event type constants live in `event_types`.
// This module covers conversation items, content parts, the realtime response
// object, usage, errors, rate limits, and MCP approval types.

use serde::{Deserialize, Serialize};

// ============================================================================
// Conversation Item
// ============================================================================
/// A conversation item in the Realtime API.
///
/// Discriminated by the `type` field:
/// - `message`               — a text/audio/image message (system/user/assistant)
/// - `function_call`         — a function call issued by the model
/// - `function_call_output`  — the result supplied by the client
/// - `mcp_call`              — an MCP tool call issued by the model
/// - `mcp_list_tools`        — MCP list-tools result
/// - `mcp_approval_request`  — server asks client to approve an MCP call
/// - `mcp_approval_response` — client approves/denies an MCP call
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RealtimeConversationItem {
    Message {
        id: Option<String>,
        object: Option<ConversationItemObject>,
        status: Option<ConversationItemStatus>,
        role: ConversationItemRole,
        content: Vec<RealtimeContentPart>,
    },
    FunctionCall {
        id: Option<String>,
        object: Option<ConversationItemObject>,
        status: Option<ConversationItemStatus>,
        call_id: Option<String>,
        name: String,
        arguments: String,
    },
    FunctionCallOutput {
        id: Option<String>,
        object: Option<ConversationItemObject>,
        status: Option<ConversationItemStatus>,
        call_id: String,
        output: String,
    },
    McpCall {
        id: String,
        server_label: String,
        name: String,
        arguments: String,
        approval_request_id: Option<String>,
        output: Option<String>,
        error: Option<McpCallError>,
    },
    McpListTools {
        id: Option<String>,
        server_label: String,
        tools: Vec<McpListToolEntry>,
    },
    McpApprovalRequest {
        id: String,
        server_label: String,
        name: String,
        arguments: String,
    },
    McpApprovalResponse {
        id: String,
        approval_request_id: String,
        approve: bool,
        reason: Option<String>,
    },
}

/// Object type for conversation items. Always `"realtime.item"`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ConversationItemObject {
    #[serde(rename = "realtime.item")]
    RealtimeItem,
}

/// Status of a conversation item.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConversationItemStatus {
    Completed,
    Incomplete,
    InProgress,
}

/// Role for a conversation item.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConversationItemRole {
    User,
    Assistant,
    System,
}

// ============================================================================
// Content Parts (Realtime-specific)
// ============================================================================

/// Content part inside a `RealtimeConversationItem::Message`.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RealtimeContentPart {
    InputText { text: String },
    InputAudio {
        audio: Option<String>,
        transcript: Option<String>,
    },
    InputImage {
        image_url: Option<String>,
        detail: Option<ImageDetail>,
    },
    OutputText { text: String },
    OutputAudio {
        audio: Option<String>,
        transcript: Option<String>,
    },
}

/// Detail level for image processing.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ImageDetail {
    Low,
    High,
    #[default]
    Auto,
}

// ============================================================================
// MCP Types
// ============================================================================

/// A tool entry returned in `mcp_list_tools` conversation items.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpListToolEntry {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
    pub annotations: Option<serde_json::Value>,
}

/// Error from an MCP tool call.
///
/// One of: protocol error, tool execution error, or HTTP error.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum McpCallError {
    /// MCP protocol-level error.
    ProtocolError { code: i32, message: String },
    /// Error during tool execution on the MCP server.
    ToolExecutionError { message: String },
    /// HTTP-level error communicating with the MCP server.
    HttpError { code: i32, message: String },
}