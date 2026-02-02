//! Conversation types for the Realtime API.
//!
//! This module contains types representing conversation items, including
//! messages, function calls, and function call outputs.

use serde::{Deserialize, Serialize};

// ============================================================================
// Role
// ============================================================================

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

// ============================================================================
// Item Status
// ============================================================================

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

// ============================================================================
// Content Parts
// ============================================================================

/// Content part types for conversation items.
///
/// Content can be text or audio, for both input and output.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// Text input from user
    InputText {
        text: String,
    },
    /// Audio input from user
    InputAudio {
        /// Base64-encoded audio data
        #[serde(skip_serializing_if = "Option::is_none")]
        audio: Option<String>,
        /// Transcript of the audio (if transcription enabled)
        #[serde(skip_serializing_if = "Option::is_none")]
        transcript: Option<String>,
    },
    /// Text output from assistant
    Text {
        text: String,
    },
    /// Audio output from assistant
    Audio {
        /// Base64-encoded audio data
        #[serde(skip_serializing_if = "Option::is_none")]
        audio: Option<String>,
        /// Transcript of the audio
        #[serde(skip_serializing_if = "Option::is_none")]
        transcript: Option<String>,
    },
}

impl ContentPart {
    /// Create a new input text content part
    pub fn input_text(text: impl Into<String>) -> Self {
        Self::InputText { text: text.into() }
    }

    /// Create a new input audio content part
    pub fn input_audio(audio: Option<String>, transcript: Option<String>) -> Self {
        Self::InputAudio { audio, transcript }
    }

    /// Create a new text content part
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    /// Create a new audio content part
    pub fn audio(audio: Option<String>, transcript: Option<String>) -> Self {
        Self::Audio { audio, transcript }
    }

    /// Get the text content if this is a text part
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::InputText { text } | Self::Text { text } => Some(text),
            _ => None,
        }
    }

    /// Get the transcript if this part has one
    pub fn transcript(&self) -> Option<&str> {
        match self {
            Self::InputAudio { transcript, .. } | Self::Audio { transcript, .. } => {
                transcript.as_deref()
            }
            _ => None,
        }
    }
}

// ============================================================================
// Conversation Items
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
        call_id: String,
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
        /// ID matching the function call
        call_id: String,
        /// Output from the function (JSON string)
        output: String,
    },
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
            call_id: call_id.into(),
            output: output.into(),
        }
    }

    /// Get the item ID if present
    pub fn id(&self) -> Option<&str> {
        match self {
            Self::Message { id, .. }
            | Self::FunctionCall { id, .. }
            | Self::FunctionCallOutput { id, .. } => id.as_deref(),
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

    /// Get function call details if this is a function call
    pub fn as_function_call(&self) -> Option<(&str, &str, &str)> {
        match self {
            Self::FunctionCall {
                call_id,
                name,
                arguments,
                ..
            } => Some((call_id.as_str(), name.as_str(), arguments.as_str())),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_role_serialization() {
        assert_eq!(serde_json::to_string(&Role::User).unwrap(), "\"user\"");
        assert_eq!(serde_json::to_string(&Role::Assistant).unwrap(), "\"assistant\"");
        assert_eq!(serde_json::to_string(&Role::System).unwrap(), "\"system\"");
    }

    #[test]
    fn test_content_part_serialization() {
        let text = ContentPart::input_text("hello");
        let json = serde_json::to_string(&text).unwrap();
        assert!(json.contains("\"type\":\"input_text\""));
        assert!(json.contains("\"text\":\"hello\""));

        let audio = ContentPart::input_audio(Some("base64data".to_string()), None);
        let json = serde_json::to_string(&audio).unwrap();
        assert!(json.contains("\"type\":\"input_audio\""));
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
            call_id: "call_456".to_string(),
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
}
