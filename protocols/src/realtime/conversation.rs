//! Conversation types for the Realtime API.
//!
//! This module contains types representing conversation items, including
//! messages, function calls, and function call outputs.

use serde::{Deserialize, Serialize};

// ============================================================================
// Conversation Item (Main Type)
// ============================================================================

/// A conversation item (message, function call, or function call output).
///
/// This is the main type for items in the conversation history.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ConversationItem {
    /// A message (user, assistant, or system)
    Message {
        /// Unique identifier for this item
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// Object type (always "realtime.item")
        #[serde(skip_serializing_if = "Option::is_none")]
        object: Option<String>,
        /// Status of the item
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<ItemStatus>,
        /// Role of the message author
        role: Role,
        /// Content of the message
        content: Vec<ContentPart>,
    },
    /// A function call made by the assistant
    FunctionCall {
        /// Unique identifier for this item
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// Object type (always "realtime.item")
        #[serde(skip_serializing_if = "Option::is_none")]
        object: Option<String>,
        /// Status of the item
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<ItemStatus>,
        /// ID for matching call with output
        #[serde(skip_serializing_if = "Option::is_none")]
        call_id: Option<String>,
        /// Name of the function being called
        name: String,
        /// JSON-encoded function arguments
        arguments: String,
    },
    /// Output from a function call
    FunctionCallOutput {
        /// Unique identifier for this item
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// Object type (always "realtime.item")
        #[serde(skip_serializing_if = "Option::is_none")]
        object: Option<String>,
        /// Status of the item (has no effect on the conversation)
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<ItemStatus>,
        /// ID matching the function call
        call_id: String,
        /// Output from the function (JSON string)
        output: String,
    },

    // === MCP Events ===
    /// List of tools available on an MCP server
    McpListTools {
        /// Unique identifier for this item
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// The label of the MCP server
        server_label: String,
        /// The tools available on the server
        tools: Vec<McpListToolsTool>,
    },

    /// An invocation of a tool on an MCP server
    McpCall {
        /// Unique identifier for this item
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// The label of the MCP server running the tool
        server_label: String,
        /// The name of the tool being called
        name: String,
        /// JSON-encoded arguments passed to the tool
        arguments: String,
        /// ID of an associated approval request, if any
        #[serde(skip_serializing_if = "Option::is_none")]
        approval_request_id: Option<String>,
        /// Output from the tool call
        #[serde(skip_serializing_if = "Option::is_none")]
        output: Option<String>,
        /// Error from the tool call, if any
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<McpError>,
    },

    /// A request for human approval of an MCP tool invocation
    McpApprovalRequest {
        /// Unique identifier for this item
        id: String,
        /// The label of the MCP server making the request
        server_label: String,
        /// The name of the tool to run
        name: String,
        /// JSON-encoded arguments for the tool
        arguments: String,
    },

    /// A response to an MCP approval request
    McpApprovalResponse {
        /// Unique identifier for this item
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// The ID of the approval request being answered
        approval_request_id: String,
        /// Whether the request was approved
        approve: bool,
        /// Optional reason for the decision
        #[serde(skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },

    /// Unknown item type (for forward compatibility)
    #[serde(other)]
    Unknown,
}

impl ConversationItem {
    /// Create a new user message with text content
    pub fn user_message(content: Vec<ContentPart>) -> Self {
        Self::Message {
            id: None,
            object: None,
            status: None,
            role: Role::User,
            content,
        }
    }

    /// Create a new user message with simple text
    pub fn user_text(text: impl Into<String>) -> Self {
        Self::user_message(vec![ContentPart::input_text(text)])
    }

    /// Create a new assistant message
    pub fn assistant_message(content: Vec<ContentPart>) -> Self {
        Self::Message {
            id: None,
            object: None,
            status: None,
            role: Role::Assistant,
            content,
        }
    }

    /// Create a new system message
    pub fn system_message(text: impl Into<String>) -> Self {
        Self::Message {
            id: None,
            object: None,
            status: None,
            role: Role::System,
            content: vec![ContentPart::input_text(text)],
        }
    }

    /// Create a new function call output
    pub fn function_output(call_id: impl Into<String>, output: impl Into<String>) -> Self {
        Self::FunctionCallOutput {
            id: None,
            object: None,
            status: None,
            call_id: call_id.into(),
            output: output.into(),
        }
    }

    /// Create an MCP approval response
    pub fn mcp_approval(
        approval_request_id: impl Into<String>,
        approve: bool,
        reason: Option<String>,
    ) -> Self {
        Self::McpApprovalResponse {
            id: None,
            approval_request_id: approval_request_id.into(),
            approve,
            reason,
        }
    }

    /// Get the item ID if present
    pub fn id(&self) -> Option<&str> {
        match self {
            Self::Message { id, .. }
            | Self::FunctionCall { id, .. }
            | Self::FunctionCallOutput { id, .. }
            | Self::McpListTools { id, .. }
            | Self::McpCall { id, .. }
            | Self::McpApprovalResponse { id, .. } => id.as_deref(),
            Self::McpApprovalRequest { id, .. } => Some(id),
            Self::Unknown => None,
        }
    }

    /// Get the role if this is a message
    pub fn role(&self) -> Option<Role> {
        match self {
            Self::Message { role, .. } => Some(*role),
            _ => None,
        }
    }

    /// Check if this is a function call
    pub fn is_function_call(&self) -> bool {
        matches!(self, Self::FunctionCall { .. })
    }

    /// Check if this is an MCP tool call
    pub fn is_mcp_call(&self) -> bool {
        matches!(self, Self::McpCall { .. })
    }

    /// Check if this is an unknown item type
    pub fn is_unknown(&self) -> bool {
        matches!(self, Self::Unknown)
    }

    /// Get function call details if this is a function call
    ///
    /// Returns (call_id, name, arguments) tuple. call_id may be None.
    pub fn as_function_call(&self) -> Option<(Option<&str>, &str, &str)> {
        match self {
            Self::FunctionCall {
                call_id,
                name,
                arguments,
                ..
            } => Some((call_id.as_deref(), name.as_str(), arguments.as_str())),
            _ => None,
        }
    }

    /// Get MCP call details if this is an MCP tool call
    pub fn as_mcp_call(&self) -> Option<(&str, &str, &str)> {
        match self {
            Self::McpCall {
                server_label,
                name,
                arguments,
                ..
            } => Some((server_label.as_str(), name.as_str(), arguments.as_str())),
            _ => None,
        }
    }
}

// ============================================================================
// Content Parts
// ============================================================================

/// Content part types for conversation items.
///
/// Content can be text, audio, or images, for both input and output.
/// - System messages: only `InputText`
/// - User messages: `InputText`, `InputAudio`, `InputImage`
/// - Assistant messages: `OutputText`, `OutputAudio`
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// Text input (for system/user messages)
    InputText {
        /// The text content
        text: String,
    },
    /// Audio input (for user messages)
    InputAudio {
        /// Base64-encoded audio data
        audio: String,
        /// Transcript of the audio (client-provided, not sent to model)
        transcript: String,
    },
    /// Image input (for user messages)
    InputImage {
        /// Base64-encoded image as data URI (e.g., "data:image/png;base64,...")
        /// Supported formats: PNG, JPEG
        image_url: String,
        /// Detail level for image processing
        detail: ImageDetail,
    },
    /// Text output (for assistant messages)
    OutputText {
        /// The text content
        text: String,
    },
    /// Audio output (for assistant messages)
    OutputAudio {
        /// Base64-encoded audio data
        audio: String,
        /// Transcript of the audio (always present when output type is audio)
        transcript: String,
    },

    /// Unknown content part type (for forward compatibility)
    #[serde(other)]
    Unknown,
}

impl ContentPart {
    /// Create a new input text content part
    pub fn input_text(text: impl Into<String>) -> Self {
        Self::InputText { text: text.into() }
    }

    /// Create a new input audio content part
    pub fn input_audio(audio: impl Into<String>, transcript: impl Into<String>) -> Self {
        Self::InputAudio {
            audio: audio.into(),
            transcript: transcript.into(),
        }
    }

    /// Create a new input image content part
    pub fn input_image(image_url: impl Into<String>, detail: ImageDetail) -> Self {
        Self::InputImage {
            image_url: image_url.into(),
            detail,
        }
    }

    /// Create a new output text content part
    pub fn output_text(text: impl Into<String>) -> Self {
        Self::OutputText { text: text.into() }
    }

    /// Create a new output audio content part
    pub fn output_audio(audio: impl Into<String>, transcript: impl Into<String>) -> Self {
        Self::OutputAudio {
            audio: audio.into(),
            transcript: transcript.into(),
        }
    }

    /// Get the text content if this is a text part
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::InputText { text } | Self::OutputText { text } => Some(text),
            _ => None,
        }
    }

    /// Get the transcript if this part has one
    pub fn transcript(&self) -> Option<&str> {
        match self {
            Self::InputAudio { transcript, .. } | Self::OutputAudio { transcript, .. } => {
                Some(transcript.as_str())
            }
            _ => None,
        }
    }

    /// Check if this is an unknown content part type
    pub fn is_unknown(&self) -> bool {
        matches!(self, Self::Unknown)
    }
}

// ============================================================================
// Supporting Types
// ============================================================================

// --- Role ---

/// Conversation item roles.
///
/// Represents who authored a message in the conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
    System,
}

// --- Item Status ---

/// Status of a conversation item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ItemStatus {
    /// Item is complete
    Completed,
    /// Item is still being generated
    InProgress,
    /// Item generation was interrupted
    Incomplete,
}

// --- Image Detail ---

/// Detail level for image processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageDetail {
    /// Automatically choose detail level (defaults to high)
    #[default]
    Auto,
    /// Low detail - faster processing
    Low,
    /// High detail - better accuracy
    High,
}

// --- MCP Types ---

/// Error from an MCP tool call.
///
/// Represents one of three error types that can occur during MCP tool execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum McpError {
    /// HTTP-level error from the MCP server
    #[serde(rename = "http_error")]
    HttpError {
        /// HTTP status code
        code: i32,
        /// Error message
        message: String,
    },
    /// Protocol-level error from the MCP server
    #[serde(rename = "protocol_error")]
    ProtocolError {
        /// Error code
        code: i32,
        /// Error message
        message: String,
    },
    /// Error during tool execution
    #[serde(rename = "tool_execution_error")]
    ToolExecutionError {
        /// Error message
        message: String,
    },
}

/// A tool available on an MCP server.
///
/// Returned in `mcp_list_tools` items and `mcp.list_tools.completed` events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpListToolsTool {
    /// The name of the tool
    pub name: String,
    /// Description of the tool
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON schema describing the tool's input (required)
    pub input_schema: serde_json::Value,
    /// Additional annotations about the tool
    #[serde(skip_serializing_if = "Option::is_none")]
    pub annotations: Option<serde_json::Value>,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_role_serialization() {
        assert_eq!(serde_json::to_string(&Role::User).unwrap(), "\"user\"");
        assert_eq!(
            serde_json::to_string(&Role::Assistant).unwrap(),
            "\"assistant\""
        );
        assert_eq!(serde_json::to_string(&Role::System).unwrap(), "\"system\"");
    }

    #[test]
    fn test_content_part_serialization() {
        let text = ContentPart::input_text("hello");
        let json = serde_json::to_string(&text).unwrap();
        assert!(json.contains("\"type\":\"input_text\""));
        assert!(json.contains("\"text\":\"hello\""));

        let audio = ContentPart::input_audio("base64data", "transcript");
        let json = serde_json::to_string(&audio).unwrap();
        assert!(json.contains("\"type\":\"input_audio\""));

        let image = ContentPart::input_image("data:image/png;base64,abc123", ImageDetail::High);
        let json = serde_json::to_string(&image).unwrap();
        assert!(json.contains("\"type\":\"input_image\""));
        assert!(json.contains("\"detail\":\"high\""));

        let output_text = ContentPart::output_text("response");
        let json = serde_json::to_string(&output_text).unwrap();
        assert!(json.contains("\"type\":\"output_text\""));

        let output_audio = ContentPart::output_audio("audio_data", "transcript");
        let json = serde_json::to_string(&output_audio).unwrap();
        assert!(json.contains("\"type\":\"output_audio\""));
        assert!(json.contains("\"transcript\":\"transcript\""));
    }

    #[test]
    fn test_conversation_item_message() {
        let item = ConversationItem::user_text("Hello!");
        let json = serde_json::to_string(&item).unwrap();
        assert!(json.contains("\"type\":\"message\""));
        assert!(json.contains("\"role\":\"user\""));
    }

    #[test]
    fn test_conversation_item_function_call() {
        let item = ConversationItem::FunctionCall {
            id: Some("item_123".to_string()),
            object: Some("realtime.item".to_string()),
            status: Some(ItemStatus::Completed),
            call_id: Some("call_456".to_string()),
            name: "get_weather".to_string(),
            arguments: r#"{"location":"NYC"}"#.to_string(),
        };
        let json = serde_json::to_string(&item).unwrap();
        assert!(json.contains("\"type\":\"function_call\""));
        assert!(json.contains("\"name\":\"get_weather\""));
    }

    #[test]
    fn test_conversation_item_function_output() {
        let item = ConversationItem::function_output("call_123", r#"{"temp": 72}"#);
        let json = serde_json::to_string(&item).unwrap();
        assert!(json.contains("\"type\":\"function_call_output\""));
        assert!(json.contains("\"call_id\":\"call_123\""));
    }

    #[test]
    fn test_mcp_list_tools_serialization() {
        let item = ConversationItem::McpListTools {
            id: Some("item_001".to_string()),
            server_label: "my_server".to_string(),
            tools: vec![McpListToolsTool {
                name: "get_weather".to_string(),
                description: Some("Get weather info".to_string()),
                input_schema: serde_json::json!({"type": "object"}),
                annotations: None,
            }],
        };
        let json = serde_json::to_string(&item).unwrap();
        assert!(json.contains("\"type\":\"mcp_list_tools\""));
        assert!(json.contains("\"server_label\":\"my_server\""));
        assert!(json.contains("\"get_weather\""));
    }

    #[test]
    fn test_mcp_call_serialization() {
        let item = ConversationItem::McpCall {
            id: Some("item_002".to_string()),
            server_label: "my_server".to_string(),
            name: "get_weather".to_string(),
            arguments: r#"{"city":"NYC"}"#.to_string(),
            approval_request_id: None,
            output: Some("Sunny, 72F".to_string()),
            error: None,
        };
        let json = serde_json::to_string(&item).unwrap();
        assert!(json.contains("\"type\":\"mcp_call\""));
        assert!(json.contains("\"server_label\":\"my_server\""));
        assert!(json.contains("\"name\":\"get_weather\""));
        assert!(json.contains("\"output\":\"Sunny, 72F\""));

        assert!(item.is_mcp_call());
        let (label, name, args) = item.as_mcp_call().unwrap();
        assert_eq!(label, "my_server");
        assert_eq!(name, "get_weather");
        assert_eq!(args, r#"{"city":"NYC"}"#);
    }

    #[test]
    fn test_mcp_call_with_error() {
        let item = ConversationItem::McpCall {
            id: Some("item_003".to_string()),
            server_label: "my_server".to_string(),
            name: "bad_tool".to_string(),
            arguments: "{}".to_string(),
            approval_request_id: None,
            output: None,
            error: Some(McpError::HttpError {
                code: 500,
                message: "Internal server error".to_string(),
            }),
        };
        let json = serde_json::to_string(&item).unwrap();
        assert!(json.contains("\"type\":\"http_error\""));
        assert!(json.contains("\"code\":500"));
    }

    #[test]
    fn test_mcp_approval_request_serialization() {
        let item = ConversationItem::McpApprovalRequest {
            id: "item_004".to_string(),
            server_label: "my_server".to_string(),
            name: "delete_file".to_string(),
            arguments: r#"{"path":"/tmp/foo"}"#.to_string(),
        };
        let json = serde_json::to_string(&item).unwrap();
        assert!(json.contains("\"type\":\"mcp_approval_request\""));
        assert!(json.contains("\"name\":\"delete_file\""));
    }

    #[test]
    fn test_mcp_approval_response_serialization() {
        let item = ConversationItem::mcp_approval("req_123", true, None);
        let json = serde_json::to_string(&item).unwrap();
        assert!(json.contains("\"type\":\"mcp_approval_response\""));
        assert!(json.contains("\"approval_request_id\":\"req_123\""));
        assert!(json.contains("\"approve\":true"));
        assert!(!json.contains("\"reason\""));
    }

    #[test]
    fn test_mcp_approval_response_with_reason() {
        let item =
            ConversationItem::mcp_approval("req_456", false, Some("Not authorized".to_string()));
        let json = serde_json::to_string(&item).unwrap();
        assert!(json.contains("\"approve\":false"));
        assert!(json.contains("\"reason\":\"Not authorized\""));
    }

    #[test]
    fn test_mcp_error_types() {
        let http_err = McpError::HttpError {
            code: 404,
            message: "Not found".to_string(),
        };
        let json = serde_json::to_string(&http_err).unwrap();
        assert!(json.contains("\"type\":\"http_error\""));

        let proto_err = McpError::ProtocolError {
            code: -32600,
            message: "Invalid request".to_string(),
        };
        let json = serde_json::to_string(&proto_err).unwrap();
        assert!(json.contains("\"type\":\"protocol_error\""));

        let exec_err = McpError::ToolExecutionError {
            message: "Tool crashed".to_string(),
        };
        let json = serde_json::to_string(&exec_err).unwrap();
        assert!(json.contains("\"type\":\"tool_execution_error\""));
        assert!(!json.contains("\"code\""));
    }

    #[test]
    fn test_mcp_call_deserialization() {
        let json = r#"{
            "type": "mcp_call",
            "id": "item_005",
            "server_label": "weather_server",
            "name": "get_forecast",
            "arguments": "{\"city\":\"SF\"}",
            "approval_request_id": "req_789",
            "error": {
                "type": "tool_execution_error",
                "message": "Timeout"
            }
        }"#;
        let item: ConversationItem = serde_json::from_str(json).unwrap();
        assert!(item.is_mcp_call());
        assert_eq!(item.id(), Some("item_005"));
    }

    #[test]
    fn test_unknown_content_part_handling() {
        // Unknown content types should deserialize to Unknown variant
        let json = r#"{"type": "future_content_type", "data": "something"}"#;
        let part: ContentPart = serde_json::from_str(json).unwrap();
        assert!(part.is_unknown());
        assert!(part.as_text().is_none());
        assert!(part.transcript().is_none());
    }

    #[test]
    fn test_unknown_conversation_item_handling() {
        // Unknown item types should deserialize to Unknown variant
        let json = r#"{"type": "future_item_type", "id": "item_999", "data": "something"}"#;
        let item: ConversationItem = serde_json::from_str(json).unwrap();
        assert!(item.is_unknown());
        assert!(item.id().is_none());
        assert!(item.role().is_none());
        assert!(!item.is_function_call());
        assert!(!item.is_mcp_call());
    }
}
