//! Client events for the Realtime API.
//!
//! This module contains all event types that can be sent from the client
//! to the Realtime API server. There are 12 client event types organized
//! into six categories:
//!
//! - Session events: `session.update`
//! - Input audio buffer events: `append`, `commit`, `clear`
//! - Output audio buffer events: `clear`
//! - Conversation events: `item.create`, `item.truncate`, `item.delete`, `item.retrieve`
//! - Response events: `response.create`, `response.cancel`
//! - Transcription session events: `transcription_session.update`

use serde::{Deserialize, Deserializer, Serialize};

use super::{
    conversation::ConversationItem, response::ResponseConfig, session::SessionConfig,
    transcription::TranscriptionSessionConfig,
};

// ============================================================================
// Client Event Enum (Main Type)
// ============================================================================

/// All client events that can be sent to the Realtime API.
///
/// Each event has an optional `event_id` that can be used for tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RealtimeClientEvent {
    // === Session Events ===
    /// Update the session configuration
    ///
    /// The session can be either a realtime session (type: "realtime") or
    /// a transcription session (type: "transcription").
    #[serde(rename = "session.update")]
    SessionUpdate {
        /// Optional client-generated event ID
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
        /// New session configuration (realtime or transcription)
        session: Box<SessionUpdateConfig>,
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

    /// Retrieve a conversation item
    ///
    /// Requests the server to return a specific conversation item.
    #[serde(rename = "conversation.item.retrieve")]
    ConversationItemRetrieve {
        /// Optional client-generated event ID
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
        /// ID of the item to retrieve
        item_id: String,
    },

    // === Output Audio Buffer Events ===
    /// Clear the output audio buffer
    ///
    /// Discards any audio that has been generated but not yet played.
    #[serde(rename = "output_audio_buffer.clear")]
    OutputAudioBufferClear {
        /// Optional client-generated event ID
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
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
        response: Option<Box<ResponseConfig>>,
    },

    /// Cancel an in-progress response
    ///
    /// Stops the current response generation. If no `response_id` is provided,
    /// cancels the in-progress response in the default conversation.
    #[serde(rename = "response.cancel")]
    ResponseCancel {
        /// Optional client-generated event ID
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
        /// Specific response ID to cancel (if not provided, cancels in-progress response)
        #[serde(skip_serializing_if = "Option::is_none")]
        response_id: Option<String>,
    },

    // === Transcription Session Events ===
    /// Update the transcription session configuration
    ///
    /// Used for transcription-only sessions (speech-to-text without conversation).
    #[serde(rename = "transcription_session.update")]
    TranscriptionSessionUpdate {
        /// Optional client-generated event ID
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
        /// New transcription session configuration
        session: TranscriptionSessionConfig,
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
            Self::ConversationItemRetrieve { .. } => "conversation.item.retrieve",
            Self::OutputAudioBufferClear { .. } => "output_audio_buffer.clear",
            Self::ResponseCreate { .. } => "response.create",
            Self::ResponseCancel { .. } => "response.cancel",
            Self::TranscriptionSessionUpdate { .. } => "transcription_session.update",
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
            | Self::ConversationItemRetrieve { event_id, .. }
            | Self::OutputAudioBufferClear { event_id, .. }
            | Self::ResponseCreate { event_id, .. }
            | Self::ResponseCancel { event_id, .. }
            | Self::TranscriptionSessionUpdate { event_id, .. } => event_id.as_deref(),
        }
    }
}

// ============================================================================
// Supporting Types
// ============================================================================

/// Configuration for `session.update` events.
///
/// Can be either a realtime session config or a transcription session config,
/// distinguished by the `type` field ("realtime" or "transcription").
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum SessionUpdateConfig {
    /// Realtime session configuration (type: "realtime")
    Realtime(Box<SessionConfig>),
    /// Transcription session configuration (type: "transcription")
    Transcription(TranscriptionSessionConfig),
}

impl<'de> Deserialize<'de> for SessionUpdateConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Deserialize to a raw value first to inspect the type field
        let value = serde_json::Value::deserialize(deserializer)?;

        // Check the type field to determine which variant to use
        let session_type = value
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("realtime");

        match session_type {
            "transcription" => {
                let config: TranscriptionSessionConfig =
                    serde_json::from_value(value).map_err(serde::de::Error::custom)?;
                Ok(SessionUpdateConfig::Transcription(config))
            }
            _ => {
                // Default to realtime for "realtime" or any other value
                let config: SessionConfig =
                    serde_json::from_value(value).map_err(serde::de::Error::custom)?;
                Ok(SessionUpdateConfig::Realtime(Box::new(config)))
            }
        }
    }
}

impl From<SessionConfig> for SessionUpdateConfig {
    fn from(config: SessionConfig) -> Self {
        Self::Realtime(Box::new(config))
    }
}

impl From<TranscriptionSessionConfig> for SessionUpdateConfig {
    fn from(config: TranscriptionSessionConfig) -> Self {
        Self::Transcription(config)
    }
}

// ============================================================================
// Builder Methods
// ============================================================================

impl RealtimeClientEvent {
    /// Create a session update event
    ///
    /// Accepts either `SessionConfig` (realtime) or `TranscriptionSessionConfig`.
    pub fn session_update(session: impl Into<SessionUpdateConfig>) -> Self {
        Self::SessionUpdate {
            event_id: None,
            session: Box::new(session.into()),
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

    /// Create a conversation item retrieve event
    pub fn item_retrieve(item_id: impl Into<String>) -> Self {
        Self::ConversationItemRetrieve {
            event_id: None,
            item_id: item_id.into(),
        }
    }

    /// Create an output audio buffer clear event
    pub fn output_audio_clear() -> Self {
        Self::OutputAudioBufferClear { event_id: None }
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
            response: Some(Box::new(config)),
        }
    }

    /// Create a response cancel event (cancels in-progress response in default conversation)
    pub fn response_cancel() -> Self {
        Self::ResponseCancel {
            event_id: None,
            response_id: None,
        }
    }

    /// Create a response cancel event for a specific response ID
    pub fn response_cancel_with_id(response_id: impl Into<String>) -> Self {
        Self::ResponseCancel {
            event_id: None,
            response_id: Some(response_id.into()),
        }
    }

    /// Create a transcription session update event
    pub fn transcription_session_update(session: TranscriptionSessionConfig) -> Self {
        Self::TranscriptionSessionUpdate {
            event_id: None,
            session,
        }
    }

    /// Add an event ID to this event
    pub fn with_event_id(mut self, id: impl Into<String>) -> Self {
        let event_id_ref = match &mut self {
            Self::SessionUpdate { event_id, .. } => event_id,
            Self::InputAudioBufferAppend { event_id, .. } => event_id,
            Self::InputAudioBufferCommit { event_id, .. } => event_id,
            Self::InputAudioBufferClear { event_id, .. } => event_id,
            Self::ConversationItemCreate { event_id, .. } => event_id,
            Self::ConversationItemTruncate { event_id, .. } => event_id,
            Self::ConversationItemDelete { event_id, .. } => event_id,
            Self::ConversationItemRetrieve { event_id, .. } => event_id,
            Self::OutputAudioBufferClear { event_id, .. } => event_id,
            Self::ResponseCreate { event_id, .. } => event_id,
            Self::ResponseCancel { event_id, .. } => event_id,
            Self::TranscriptionSessionUpdate { event_id, .. } => event_id,
        };
        *event_id_ref = Some(id.into());
        self
    }
}

// ============================================================================
// Tests
// ============================================================================

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

    #[test]
    fn test_item_retrieve_serialization() {
        let event = RealtimeClientEvent::item_retrieve("item_abc");
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"conversation.item.retrieve\""));
        assert!(json.contains("\"item_id\":\"item_abc\""));
        assert_eq!(event.event_type(), "conversation.item.retrieve");
    }

    #[test]
    fn test_output_audio_buffer_clear_serialization() {
        let event = RealtimeClientEvent::output_audio_clear();
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"output_audio_buffer.clear\""));
        assert_eq!(event.event_type(), "output_audio_buffer.clear");
    }

    #[test]
    fn test_transcription_session_update_serialization() {
        use crate::realtime::transcription::TranscriptionSessionConfig;

        let event = RealtimeClientEvent::transcription_session_update(
            TranscriptionSessionConfig::default(),
        );
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"transcription_session.update\""));
        assert!(json.contains("\"session\""));
        assert_eq!(event.event_type(), "transcription_session.update");
    }

    #[test]
    fn test_session_update_with_realtime_config() {
        let event = RealtimeClientEvent::session_update(SessionConfig::default());
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"session.update\""));
        // The session contains type: "realtime"
        assert!(json.contains("\"realtime\""));
    }

    #[test]
    fn test_session_update_with_transcription_config() {
        use crate::realtime::transcription::TranscriptionSessionConfig;

        let event = RealtimeClientEvent::session_update(TranscriptionSessionConfig::default());
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"session.update\""));
        // The session contains type: "transcription"
        assert!(json.contains("\"transcription\""));
    }

    #[test]
    fn test_session_update_config_deserialization() {
        // Realtime config
        let json = r#"{"type": "realtime"}"#;
        let config: SessionUpdateConfig = serde_json::from_str(json).unwrap();
        assert!(matches!(config, SessionUpdateConfig::Realtime(_)));

        // Transcription config
        let json = r#"{"type": "transcription"}"#;
        let config: SessionUpdateConfig = serde_json::from_str(json).unwrap();
        assert!(matches!(config, SessionUpdateConfig::Transcription(_)));
    }
}
