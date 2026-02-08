// OpenAI Realtime Conversation API types
// https://platform.openai.com/docs/api-reference/realtime
//
// Session configuration and audio types live in `realtime_session`.
// Event type constants live in `event_types`.
// This module covers conversation items, content parts, the realtime response
// object, usage, errors, rate limits, and MCP approval types.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::realtime_session::{
    MaxOutputTokens, RealtimeSessionConfig, RealtimeTool, RealtimeToolChoice, Truncation, Voice,
};

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
    InputText {
        text: String,
    },
    InputAudio {
        audio: Option<String>,
        transcript: Option<String>,
    },
    InputImage {
        image_url: Option<String>,
        detail: Option<ImageDetail>,
    },
    OutputText {
        text: String,
    },
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
// Logprobs
// ============================================================================

/// A single logprob entry for output text or audio transcription.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogprobEntry {
    pub token: String,
    pub logprob: f64,
    pub bytes: Option<Vec<u8>>,
    pub top_logprobs: Option<Vec<TopLogprob>>,
}

/// A top-logprob alternative token.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopLogprob {
    pub token: String,
    pub logprob: f64,
    pub bytes: Option<Vec<u8>>,
}

// ============================================================================
// Realtime Response
// ============================================================================

/// Status of a realtime response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RealtimeResponseStatus {
    InProgress,
    Completed,
    Cancelled,
    Incomplete,
    Failed,
}

/// Reason the response was cancelled.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CancelledReason {
    TurnDetected,
    ClientCancelled,
}

/// Reason the response is incomplete.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IncompleteReason {
    MaxOutputTokens,
    ContentFilter,
    Interruption,
}

/// Status details for a realtime response (discriminated by `type`).
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RealtimeResponseStatusDetails {
    Completed {},
    Cancelled { reason: Option<CancelledReason> },
    Incomplete { reason: Option<IncompleteReason> },
    Failed { error: Option<RealtimeError> },
}

/// The realtime response object returned by the server.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeResponse {
    pub id: String,
    pub object: Option<String>,
    pub status: RealtimeResponseStatus,
    pub status_details: Option<RealtimeResponseStatusDetails>,
    pub output: Vec<RealtimeConversationItem>,
    pub usage: Option<RealtimeUsage>,
    pub metadata: Option<HashMap<String, String>>,
}

// ============================================================================
// Response Create Parameters
// ============================================================================

/// Parameters for the `response.create` client event.
///
/// These allow per-response overrides of session-level settings.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseCreateParams {
    pub modalities: Option<Vec<String>>,
    pub instructions: Option<String>,
    pub voice: Option<Voice>,
    pub output_audio_format: Option<String>,
    pub tools: Option<Vec<RealtimeTool>>,
    pub tool_choice: Option<RealtimeToolChoice>,
    pub temperature: Option<f64>,
    pub max_output_tokens: Option<MaxOutputTokens>,
    pub conversation: Option<ResponseConversation>,
    pub metadata: Option<HashMap<String, String>>,
    pub input: Option<Vec<RealtimeConversationItem>>,
    pub truncation: Option<Truncation>,
}

/// Conversation mode for `response.create`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseConversation {
    /// Use the default session conversation.
    Auto,
    /// No conversation context — ephemeral response.
    None,
}

// ============================================================================
// Usage
// ============================================================================

/// Token usage for a realtime response.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeUsage {
    pub total_tokens: u32,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub input_token_details: Option<RealtimeInputTokenDetails>,
    pub output_token_details: Option<RealtimeOutputTokenDetails>,
}

/// Breakdown of input tokens.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeInputTokenDetails {
    pub cached_tokens: Option<u32>,
    pub text_tokens: Option<u32>,
    pub audio_tokens: Option<u32>,
}

/// Breakdown of output tokens.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeOutputTokenDetails {
    pub text_tokens: Option<u32>,
    pub audio_tokens: Option<u32>,
}

// ============================================================================
// Error
// ============================================================================

/// An error returned in a realtime server event.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeError {
    #[serde(rename = "type")]
    pub r#type: Option<String>,
    pub code: Option<String>,
    pub message: String,
    pub param: Option<String>,
    pub event_id: Option<String>,
}

// ============================================================================
// Rate Limits
// ============================================================================

/// A single rate limit entry from `rate_limits.updated`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimit {
    pub name: String,
    pub limit: u64,
    pub remaining: u64,
    pub reset_seconds: f64,
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
    pub input_schema: Option<JsonValue>,
    pub annotations: Option<JsonValue>,
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

// ============================================================================
// Conversation
// ============================================================================

/// The conversation object returned by `conversation.created`.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeConversation {
    pub id: String,
    pub object: Option<String>,
}

// ============================================================================
// Transcription Events
// ============================================================================

/// Audio transcription result.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResult {
    pub item_id: String,
    pub content_index: u32,
    pub transcript: String,
    pub logprobs: Option<Vec<LogprobEntry>>,
}

/// Audio transcription segment from transcription session mode.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionSegment {
    pub item_id: String,
    pub content_index: u32,
    pub text: String,
    pub start_ms: Option<u64>,
    pub end_ms: Option<u64>,
    pub logprobs: Option<Vec<LogprobEntry>>,
}

// ============================================================================
// Session Update Payloads
// ============================================================================

/// Payload for the `session.update` client event.
///
/// Wraps the full session config. The server merges the provided
/// fields into the current session state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionUpdatePayload {
    pub session: RealtimeSessionConfig,
}
