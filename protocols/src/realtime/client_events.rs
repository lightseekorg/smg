//! Client events for the Realtime API.
//!
//! This module contains all event types that can be sent from the client
//! to the Realtime API server. There are 9 client event types organized
//! into four categories:
//!
//! - Session events: `session.update`
//! - Input audio buffer events: `append`, `commit`, `clear`
//! - Conversation events: `item.create`, `item.truncate`, `item.delete`
//! - Response events: `response.create`, `response.cancel`

use serde::{Deserialize, Serialize};

use super::conversation::ConversationItem;
use super::response::ResponseConfig;
use super::session::SessionConfig;

// ============================================================================
// Client Event Enum
// ============================================================================

/// All client events that can be sent to the Realtime API.
///
/// Each event has an optional `event_id` that can be used for tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RealtimeClientEvent {
    // === Session Events ===
    /// Update the session configuration
    #[serde(rename = "session.update")]
    SessionUpdate {
        /// Optional client-generated event ID
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
        /// New session configuration
        session: SessionConfig,
    },

    // === Input Audio Buffer Events ===
    /// Append audio data to the input buffer
    #[serde(rename = "input_audio_buffer.append")]
    InputAudioBufferAppend {
        /// Optional client-generated event ID
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
        /// Base64-encoded audio data to append
        audio: String,
    },

    /// Commit the input audio buffer
    ///
    /// Creates a user message item from the buffered audio.
    #[serde(rename = "input_audio_buffer.commit")]
    InputAudioBufferCommit {
        /// Optional client-generated event ID
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
    },

    /// Clear the input audio buffer
    ///
    /// Discards any buffered audio without creating a message.
    #[serde(rename = "input_audio_buffer.clear")]
    InputAudioBufferClear {
        /// Optional client-generated event ID
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
    },

    // === Conversation Events ===
    /// Create a new conversation item
    ///
    /// Can be used to add user messages, assistant messages, or function
    /// call outputs to the conversation.
    #[serde(rename = "conversation.item.create")]
    ConversationItemCreate {
        /// Optional client-generated event ID
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
        /// ID of the item to insert after (None for end of conversation)
        #[serde(skip_serializing_if = "Option::is_none")]
        previous_item_id: Option<String>,
        /// The conversation item to create
        item: ConversationItem,
    },

    /// Truncate a conversation item
    ///
    /// Removes audio from a user or assistant message up to a specified point.
    #[serde(rename = "conversation.item.truncate")]
    ConversationItemTruncate {
        /// Optional client-generated event ID
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
        /// ID of the item to truncate
        item_id: String,
        /// Index of the content part to truncate
        content_index: u32,
        /// Audio end position in milliseconds
        audio_end_ms: u32,
    },

    /// Delete a conversation item
    ///
    /// Removes an item from the conversation history.
    #[serde(rename = "conversation.item.delete")]
    ConversationItemDelete {
        /// Optional client-generated event ID
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
        /// ID of the item to delete
        item_id: String,
    },

    // === Response Events ===
    /// Create a new response
    ///
    /// Triggers the model to generate a response. If VAD is enabled and
    /// `create_response` is true, this happens automatically on speech end.
    #[serde(rename = "response.create")]
    ResponseCreate {
        /// Optional client-generated event ID
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
        /// Optional response configuration
        #[serde(skip_serializing_if = "Option::is_none")]
        response: Option<ResponseConfig>,
    },

    /// Cancel an in-progress response
    ///
    /// Stops the current response generation.
    #[serde(rename = "response.cancel")]
    ResponseCancel {
        /// Optional client-generated event ID
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
    },
}

impl RealtimeClientEvent {
    /// Get the event type as a string (e.g., "session.update")
    pub fn event_type(&self) -> &'static str {
        match self {
            Self::SessionUpdate { .. } => "session.update",
            Self::InputAudioBufferAppend { .. } => "input_audio_buffer.append",
            Self::InputAudioBufferCommit { .. } => "input_audio_buffer.commit",
            Self::InputAudioBufferClear { .. } => "input_audio_buffer.clear",
            Self::ConversationItemCreate { .. } => "conversation.item.create",
            Self::ConversationItemTruncate { .. } => "conversation.item.truncate",
            Self::ConversationItemDelete { .. } => "conversation.item.delete",
            Self::ResponseCreate { .. } => "response.create",
            Self::ResponseCancel { .. } => "response.cancel",
        }
    }

    /// Get the event ID if one was specified
    pub fn event_id(&self) -> Option<&str> {
        match self {
            Self::SessionUpdate { event_id, .. }
            | Self::InputAudioBufferAppend { event_id, .. }
            | Self::InputAudioBufferCommit { event_id, .. }
            | Self::InputAudioBufferClear { event_id, .. }
            | Self::ConversationItemCreate { event_id, .. }
            | Self::ConversationItemTruncate { event_id, .. }
            | Self::ConversationItemDelete { event_id, .. }
            | Self::ResponseCreate { event_id, .. }
            | Self::ResponseCancel { event_id, .. } => event_id.as_deref(),
        }
    }
}

// ============================================================================
// Builder methods for common operations
// ============================================================================

impl RealtimeClientEvent {
    /// Create a session update event
    pub fn session_update(session: SessionConfig) -> Self {
        Self::SessionUpdate {
            event_id: None,
            session,
        }
    }

    /// Create an audio buffer append event
    pub fn audio_append(audio: impl Into<String>) -> Self {
        Self::InputAudioBufferAppend {
            event_id: None,
            audio: audio.into(),
        }
    }

    /// Create an audio buffer commit event
    pub fn audio_commit() -> Self {
        Self::InputAudioBufferCommit { event_id: None }
    }

    /// Create an audio buffer clear event
    pub fn audio_clear() -> Self {
        Self::InputAudioBufferClear { event_id: None }
    }

    /// Create a conversation item create event
    pub fn item_create(item: ConversationItem) -> Self {
        Self::ConversationItemCreate {
            event_id: None,
            previous_item_id: None,
            item,
        }
    }

    /// Create a conversation item create event with positioning
    pub fn item_create_after(item: ConversationItem, previous_item_id: impl Into<String>) -> Self {
        Self::ConversationItemCreate {
            event_id: None,
            previous_item_id: Some(previous_item_id.into()),
            item,
        }
    }

    /// Create a conversation item delete event
    pub fn item_delete(item_id: impl Into<String>) -> Self {
        Self::ConversationItemDelete {
            event_id: None,
            item_id: item_id.into(),
        }
    }

    /// Create a response create event with default config
    pub fn response_create() -> Self {
        Self::ResponseCreate {
            event_id: None,
            response: None,
        }
    }

    /// Create a response create event with custom config
    pub fn response_create_with(config: ResponseConfig) -> Self {
        Self::ResponseCreate {
            event_id: None,
            response: Some(config),
        }
    }

    /// Create a response cancel event
    pub fn response_cancel() -> Self {
        Self::ResponseCancel { event_id: None }
    }

    /// Add an event ID to this event
    pub fn with_event_id(mut self, id: impl Into<String>) -> Self {
        let id = Some(id.into());
        match &mut self {
            Self::SessionUpdate { event_id, .. }
            | Self::InputAudioBufferAppend { event_id, .. }
            | Self::InputAudioBufferCommit { event_id, .. }
            | Self::InputAudioBufferClear { event_id, .. }
            | Self::ConversationItemCreate { event_id, .. }
            | Self::ConversationItemTruncate { event_id, .. }
            | Self::ConversationItemDelete { event_id, .. }
            | Self::ResponseCreate { event_id, .. }
            | Self::ResponseCancel { event_id, .. } => *event_id = id,
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::realtime::conversation::ContentPart;

    #[test]
    fn test_session_update_serialization() {
        let event = RealtimeClientEvent::session_update(SessionConfig {
            instructions: Some("Be helpful".to_string()),
            ..Default::default()
        });

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"session.update\""));
        assert!(json.contains("\"instructions\":\"Be helpful\""));
    }

    #[test]
    fn test_audio_append_serialization() {
        let event = RealtimeClientEvent::audio_append("SGVsbG8gV29ybGQ=");
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"input_audio_buffer.append\""));
        assert!(json.contains("\"audio\":\"SGVsbG8gV29ybGQ=\""));
    }

    #[test]
    fn test_item_create_serialization() {
        let item = ConversationItem::user_message(vec![ContentPart::input_text("Hello!")]);
        let event = RealtimeClientEvent::item_create(item);
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"conversation.item.create\""));
        assert!(json.contains("\"role\":\"user\""));
    }

    #[test]
    fn test_response_create_serialization() {
        let event = RealtimeClientEvent::response_create();
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"response.create\""));
    }

    #[test]
    fn test_event_with_id() {
        let event = RealtimeClientEvent::response_create().with_event_id("evt_123");
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"event_id\":\"evt_123\""));
    }

    #[test]
    fn test_event_type_method() {
        assert_eq!(
            RealtimeClientEvent::response_create().event_type(),
            "response.create"
        );
        assert_eq!(
            RealtimeClientEvent::audio_commit().event_type(),
            "input_audio_buffer.commit"
        );
    }

    #[test]
    fn test_item_truncate_serialization() {
        let event = RealtimeClientEvent::ConversationItemTruncate {
            event_id: None,
            item_id: "item_123".to_string(),
            content_index: 0,
            audio_end_ms: 5000,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"conversation.item.truncate\""));
        assert!(json.contains("\"item_id\":\"item_123\""));
        assert!(json.contains("\"audio_end_ms\":5000"));
    }

    #[test]
    fn test_deserialize_client_event() {
        let json = r#"{
            "type": "response.create",
            "event_id": "evt_456"
        }"#;
        let event: RealtimeClientEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.event_type(), "response.create");
        assert_eq!(event.event_id(), Some("evt_456"));
    }
}
