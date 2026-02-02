//! Server events for the Realtime API.
//!
//! This module contains all event types that can be received from the
//! Realtime API server. There are 28 server event types organized into
//! several categories:
//!
//! - Error events
//! - Session events: `created`, `updated`
//! - Conversation events: `created`, `item.created`, `item.truncated`, etc.
//! - Input audio buffer events: `committed`, `cleared`, `speech_started`, `speech_stopped`
//! - Response events: `created`, `done`, `output_item.added`, `output_item.done`, etc.
//! - Streaming delta events: `text.delta`, `audio.delta`, `function_call_arguments.delta`, etc.
//! - Rate limit events

use serde::{Deserialize, Serialize};

use super::conversation::{ContentPart, ConversationItem};
use super::response::Response;
use super::session::Session;

// ============================================================================
// Supporting Types
// ============================================================================

/// API error returned in error events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiError {
    /// Error type (e.g., "invalid_request_error")
    #[serde(rename = "type")]
    pub error_type: String,
    /// Error code (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    /// Human-readable error message
    pub message: String,
    /// Parameter that caused the error (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
    /// Event ID that caused the error (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event_id: Option<String>,
}

/// Conversation object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    /// Unique identifier for the conversation
    pub id: String,
    /// Object type (always "realtime.conversation")
    pub object: String,
}

/// Rate limit information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimit {
    /// Name of the rate limit (e.g., "requests", "tokens")
    pub name: String,
    /// Maximum allowed
    pub limit: u32,
    /// Remaining in current window
    pub remaining: u32,
    /// Seconds until reset
    pub reset_seconds: f32,
}

// ============================================================================
// Server Event Enum
// ============================================================================

/// All server events received from the Realtime API.
///
/// Each event includes a server-generated `event_id`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RealtimeServerEvent {
    // === Error Events ===
    /// An error occurred
    #[serde(rename = "error")]
    Error {
        /// Server-generated event ID
        event_id: String,
        /// Error details
        error: ApiError,
    },

    // === Session Events ===
    /// Session was created
    #[serde(rename = "session.created")]
    SessionCreated {
        /// Server-generated event ID
        event_id: String,
        /// The created session
        session: Session,
    },

    /// Session was updated
    #[serde(rename = "session.updated")]
    SessionUpdated {
        /// Server-generated event ID
        event_id: String,
        /// The updated session
        session: Session,
    },

    // === Conversation Events ===
    /// Conversation was created
    #[serde(rename = "conversation.created")]
    ConversationCreated {
        /// Server-generated event ID
        event_id: String,
        /// The created conversation
        conversation: Conversation,
    },

    /// Conversation item was created
    #[serde(rename = "conversation.item.created")]
    ConversationItemCreated {
        /// Server-generated event ID
        event_id: String,
        /// ID of the previous item (for ordering)
        #[serde(skip_serializing_if = "Option::is_none")]
        previous_item_id: Option<String>,
        /// The created item
        item: ConversationItem,
    },

    /// Input audio transcription completed
    #[serde(rename = "conversation.item.input_audio_transcription.completed")]
    ConversationItemInputAudioTranscriptionCompleted {
        /// Server-generated event ID
        event_id: String,
        /// ID of the item containing the audio
        item_id: String,
        /// Index of the content part
        content_index: u32,
        /// The transcription text
        transcript: String,
    },

    /// Input audio transcription failed
    #[serde(rename = "conversation.item.input_audio_transcription.failed")]
    ConversationItemInputAudioTranscriptionFailed {
        /// Server-generated event ID
        event_id: String,
        /// ID of the item containing the audio
        item_id: String,
        /// Index of the content part
        content_index: u32,
        /// Error details
        error: ApiError,
    },

    /// Conversation item was truncated
    #[serde(rename = "conversation.item.truncated")]
    ConversationItemTruncated {
        /// Server-generated event ID
        event_id: String,
        /// ID of the truncated item
        item_id: String,
        /// Index of the content part
        content_index: u32,
        /// Audio end position in milliseconds
        audio_end_ms: u32,
    },

    /// Conversation item was deleted
    #[serde(rename = "conversation.item.deleted")]
    ConversationItemDeleted {
        /// Server-generated event ID
        event_id: String,
        /// ID of the deleted item
        item_id: String,
    },

    // === Input Audio Buffer Events ===
    /// Input audio buffer was committed
    #[serde(rename = "input_audio_buffer.committed")]
    InputAudioBufferCommitted {
        /// Server-generated event ID
        event_id: String,
        /// ID of the previous item
        #[serde(skip_serializing_if = "Option::is_none")]
        previous_item_id: Option<String>,
        /// ID of the created item
        item_id: String,
    },

    /// Input audio buffer was cleared
    #[serde(rename = "input_audio_buffer.cleared")]
    InputAudioBufferCleared {
        /// Server-generated event ID
        event_id: String,
    },

    /// Speech started in input audio
    #[serde(rename = "input_audio_buffer.speech_started")]
    InputAudioBufferSpeechStarted {
        /// Server-generated event ID
        event_id: String,
        /// Audio start position in milliseconds
        audio_start_ms: u32,
        /// ID of the item being created
        item_id: String,
    },

    /// Speech stopped in input audio
    #[serde(rename = "input_audio_buffer.speech_stopped")]
    InputAudioBufferSpeechStopped {
        /// Server-generated event ID
        event_id: String,
        /// Audio end position in milliseconds
        audio_end_ms: u32,
        /// ID of the item being created
        item_id: String,
    },

    // === Response Events ===
    /// Response was created
    #[serde(rename = "response.created")]
    ResponseCreated {
        /// Server-generated event ID
        event_id: String,
        /// The created response
        response: Response,
    },

    /// Response is done
    #[serde(rename = "response.done")]
    ResponseDone {
        /// Server-generated event ID
        event_id: String,
        /// The completed response
        response: Response,
    },

    /// Output item was added to response
    #[serde(rename = "response.output_item.added")]
    ResponseOutputItemAdded {
        /// Server-generated event ID
        event_id: String,
        /// ID of the response
        response_id: String,
        /// Index of the output item
        output_index: u32,
        /// The added item
        item: ConversationItem,
    },

    /// Output item is done
    #[serde(rename = "response.output_item.done")]
    ResponseOutputItemDone {
        /// Server-generated event ID
        event_id: String,
        /// ID of the response
        response_id: String,
        /// Index of the output item
        output_index: u32,
        /// The completed item
        item: ConversationItem,
    },

    /// Content part was added to output item
    #[serde(rename = "response.content_part.added")]
    ResponseContentPartAdded {
        /// Server-generated event ID
        event_id: String,
        /// ID of the response
        response_id: String,
        /// ID of the output item
        item_id: String,
        /// Index of the output item
        output_index: u32,
        /// Index of the content part
        content_index: u32,
        /// The added content part
        part: ContentPart,
    },

    /// Content part is done
    #[serde(rename = "response.content_part.done")]
    ResponseContentPartDone {
        /// Server-generated event ID
        event_id: String,
        /// ID of the response
        response_id: String,
        /// ID of the output item
        item_id: String,
        /// Index of the output item
        output_index: u32,
        /// Index of the content part
        content_index: u32,
        /// The completed content part
        part: ContentPart,
    },

    // === Response Streaming Delta Events ===
    /// Text delta in response
    #[serde(rename = "response.text.delta")]
    ResponseTextDelta {
        /// Server-generated event ID
        event_id: String,
        /// ID of the response
        response_id: String,
        /// ID of the output item
        item_id: String,
        /// Index of the output item
        output_index: u32,
        /// Index of the content part
        content_index: u32,
        /// The text delta
        delta: String,
    },

    /// Text output is done
    #[serde(rename = "response.text.done")]
    ResponseTextDone {
        /// Server-generated event ID
        event_id: String,
        /// ID of the response
        response_id: String,
        /// ID of the output item
        item_id: String,
        /// Index of the output item
        output_index: u32,
        /// Index of the content part
        content_index: u32,
        /// The complete text
        text: String,
    },

    /// Audio transcript delta in response
    #[serde(rename = "response.audio_transcript.delta")]
    ResponseAudioTranscriptDelta {
        /// Server-generated event ID
        event_id: String,
        /// ID of the response
        response_id: String,
        /// ID of the output item
        item_id: String,
        /// Index of the output item
        output_index: u32,
        /// Index of the content part
        content_index: u32,
        /// The transcript delta
        delta: String,
    },

    /// Audio transcript is done
    #[serde(rename = "response.audio_transcript.done")]
    ResponseAudioTranscriptDone {
        /// Server-generated event ID
        event_id: String,
        /// ID of the response
        response_id: String,
        /// ID of the output item
        item_id: String,
        /// Index of the output item
        output_index: u32,
        /// Index of the content part
        content_index: u32,
        /// The complete transcript
        transcript: String,
    },

    /// Audio delta in response
    #[serde(rename = "response.audio.delta")]
    ResponseAudioDelta {
        /// Server-generated event ID
        event_id: String,
        /// ID of the response
        response_id: String,
        /// ID of the output item
        item_id: String,
        /// Index of the output item
        output_index: u32,
        /// Index of the content part
        content_index: u32,
        /// Base64-encoded audio delta
        delta: String,
    },

    /// Audio output is done
    #[serde(rename = "response.audio.done")]
    ResponseAudioDone {
        /// Server-generated event ID
        event_id: String,
        /// ID of the response
        response_id: String,
        /// ID of the output item
        item_id: String,
        /// Index of the output item
        output_index: u32,
        /// Index of the content part
        content_index: u32,
    },

    // === Function Call Events ===
    /// Function call arguments delta
    #[serde(rename = "response.function_call_arguments.delta")]
    ResponseFunctionCallArgumentsDelta {
        /// Server-generated event ID
        event_id: String,
        /// ID of the response
        response_id: String,
        /// ID of the output item
        item_id: String,
        /// Index of the output item
        output_index: u32,
        /// ID of the function call
        call_id: String,
        /// The arguments delta (JSON fragment)
        delta: String,
    },

    /// Function call arguments are done
    #[serde(rename = "response.function_call_arguments.done")]
    ResponseFunctionCallArgumentsDone {
        /// Server-generated event ID
        event_id: String,
        /// ID of the response
        response_id: String,
        /// ID of the output item
        item_id: String,
        /// Index of the output item
        output_index: u32,
        /// ID of the function call
        call_id: String,
        /// The complete arguments (JSON string)
        arguments: String,
    },

    // === Rate Limit Events ===
    /// Rate limits were updated
    #[serde(rename = "rate_limits.updated")]
    RateLimitsUpdated {
        /// Server-generated event ID
        event_id: String,
        /// Current rate limits
        rate_limits: Vec<RateLimit>,
    },

    // === Catch-all for unknown events ===
    /// Unknown event type (for forward compatibility)
    #[serde(other)]
    Unknown,
}

impl RealtimeServerEvent {
    /// Get the event type as a string
    pub fn event_type(&self) -> &'static str {
        match self {
            Self::Error { .. } => "error",
            Self::SessionCreated { .. } => "session.created",
            Self::SessionUpdated { .. } => "session.updated",
            Self::ConversationCreated { .. } => "conversation.created",
            Self::ConversationItemCreated { .. } => "conversation.item.created",
            Self::ConversationItemInputAudioTranscriptionCompleted { .. } => {
                "conversation.item.input_audio_transcription.completed"
            }
            Self::ConversationItemInputAudioTranscriptionFailed { .. } => {
                "conversation.item.input_audio_transcription.failed"
            }
            Self::ConversationItemTruncated { .. } => "conversation.item.truncated",
            Self::ConversationItemDeleted { .. } => "conversation.item.deleted",
            Self::InputAudioBufferCommitted { .. } => "input_audio_buffer.committed",
            Self::InputAudioBufferCleared { .. } => "input_audio_buffer.cleared",
            Self::InputAudioBufferSpeechStarted { .. } => "input_audio_buffer.speech_started",
            Self::InputAudioBufferSpeechStopped { .. } => "input_audio_buffer.speech_stopped",
            Self::ResponseCreated { .. } => "response.created",
            Self::ResponseDone { .. } => "response.done",
            Self::ResponseOutputItemAdded { .. } => "response.output_item.added",
            Self::ResponseOutputItemDone { .. } => "response.output_item.done",
            Self::ResponseContentPartAdded { .. } => "response.content_part.added",
            Self::ResponseContentPartDone { .. } => "response.content_part.done",
            Self::ResponseTextDelta { .. } => "response.text.delta",
            Self::ResponseTextDone { .. } => "response.text.done",
            Self::ResponseAudioTranscriptDelta { .. } => "response.audio_transcript.delta",
            Self::ResponseAudioTranscriptDone { .. } => "response.audio_transcript.done",
            Self::ResponseAudioDelta { .. } => "response.audio.delta",
            Self::ResponseAudioDone { .. } => "response.audio.done",
            Self::ResponseFunctionCallArgumentsDelta { .. } => {
                "response.function_call_arguments.delta"
            }
            Self::ResponseFunctionCallArgumentsDone { .. } => {
                "response.function_call_arguments.done"
            }
            Self::RateLimitsUpdated { .. } => "rate_limits.updated",
            Self::Unknown => "unknown",
        }
    }

    /// Get the event ID if present
    pub fn event_id(&self) -> Option<&str> {
        match self {
            Self::Error { event_id, .. }
            | Self::SessionCreated { event_id, .. }
            | Self::SessionUpdated { event_id, .. }
            | Self::ConversationCreated { event_id, .. }
            | Self::ConversationItemCreated { event_id, .. }
            | Self::ConversationItemInputAudioTranscriptionCompleted { event_id, .. }
            | Self::ConversationItemInputAudioTranscriptionFailed { event_id, .. }
            | Self::ConversationItemTruncated { event_id, .. }
            | Self::ConversationItemDeleted { event_id, .. }
            | Self::InputAudioBufferCommitted { event_id, .. }
            | Self::InputAudioBufferCleared { event_id, .. }
            | Self::InputAudioBufferSpeechStarted { event_id, .. }
            | Self::InputAudioBufferSpeechStopped { event_id, .. }
            | Self::ResponseCreated { event_id, .. }
            | Self::ResponseDone { event_id, .. }
            | Self::ResponseOutputItemAdded { event_id, .. }
            | Self::ResponseOutputItemDone { event_id, .. }
            | Self::ResponseContentPartAdded { event_id, .. }
            | Self::ResponseContentPartDone { event_id, .. }
            | Self::ResponseTextDelta { event_id, .. }
            | Self::ResponseTextDone { event_id, .. }
            | Self::ResponseAudioTranscriptDelta { event_id, .. }
            | Self::ResponseAudioTranscriptDone { event_id, .. }
            | Self::ResponseAudioDelta { event_id, .. }
            | Self::ResponseAudioDone { event_id, .. }
            | Self::ResponseFunctionCallArgumentsDelta { event_id, .. }
            | Self::ResponseFunctionCallArgumentsDone { event_id, .. }
            | Self::RateLimitsUpdated { event_id, .. } => Some(event_id.as_str()),
            Self::Unknown => None,
        }
    }

    /// Check if this is a function call completion event (for MCP interception)
    pub fn is_function_call_done(&self) -> bool {
        matches!(self, Self::ResponseFunctionCallArgumentsDone { .. })
    }

    /// Extract function call details if this is a function call done event
    ///
    /// Returns (call_id, item_id, arguments) tuple
    pub fn get_function_call(&self) -> Option<(&str, &str, &str)> {
        match self {
            Self::ResponseFunctionCallArgumentsDone {
                call_id,
                item_id,
                arguments,
                ..
            } => Some((call_id.as_str(), item_id.as_str(), arguments.as_str())),
            _ => None,
        }
    }

    /// Check if this is an error event
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error { .. })
    }

    /// Get error details if this is an error event
    pub fn as_error(&self) -> Option<&ApiError> {
        match self {
            Self::Error { error, .. } => Some(error),
            _ => None,
        }
    }

    /// Check if this is a session event
    pub fn is_session_event(&self) -> bool {
        matches!(self, Self::SessionCreated { .. } | Self::SessionUpdated { .. })
    }

    /// Check if this is a response-related event
    pub fn is_response_event(&self) -> bool {
        matches!(
            self,
            Self::ResponseCreated { .. }
                | Self::ResponseDone { .. }
                | Self::ResponseOutputItemAdded { .. }
                | Self::ResponseOutputItemDone { .. }
                | Self::ResponseContentPartAdded { .. }
                | Self::ResponseContentPartDone { .. }
                | Self::ResponseTextDelta { .. }
                | Self::ResponseTextDone { .. }
                | Self::ResponseAudioTranscriptDelta { .. }
                | Self::ResponseAudioTranscriptDone { .. }
                | Self::ResponseAudioDelta { .. }
                | Self::ResponseAudioDone { .. }
                | Self::ResponseFunctionCallArgumentsDelta { .. }
                | Self::ResponseFunctionCallArgumentsDone { .. }
        )
    }

    /// Check if this is a streaming delta event
    pub fn is_delta_event(&self) -> bool {
        matches!(
            self,
            Self::ResponseTextDelta { .. }
                | Self::ResponseAudioTranscriptDelta { .. }
                | Self::ResponseAudioDelta { .. }
                | Self::ResponseFunctionCallArgumentsDelta { .. }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_event_serialization() {
        let event = RealtimeServerEvent::Error {
            event_id: "evt_123".to_string(),
            error: ApiError {
                error_type: "invalid_request_error".to_string(),
                code: Some("invalid_value".to_string()),
                message: "Invalid parameter".to_string(),
                param: Some("temperature".to_string()),
                event_id: None,
            },
        };

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"error\""));
        assert!(json.contains("\"event_id\":\"evt_123\""));
    }

    #[test]
    fn test_session_created_deserialization() {
        let json = r#"{
            "type": "session.created",
            "event_id": "evt_001",
            "session": {
                "id": "sess_123",
                "object": "realtime.session",
                "model": "gpt-4o-realtime-preview",
                "expires_at": 1699999999,
                "modalities": ["text", "audio"],
                "instructions": "",
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "tools": [],
                "tool_choice": "auto",
                "temperature": 0.8,
                "max_response_output_tokens": "inf"
            }
        }"#;

        let event: RealtimeServerEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, RealtimeServerEvent::SessionCreated { .. }));
        assert_eq!(event.event_id(), Some("evt_001"));
    }

    #[test]
    fn test_function_call_done_extraction() {
        let event = RealtimeServerEvent::ResponseFunctionCallArgumentsDone {
            event_id: "evt_456".to_string(),
            response_id: "resp_123".to_string(),
            item_id: "item_789".to_string(),
            output_index: 0,
            call_id: "call_abc".to_string(),
            arguments: r#"{"location":"NYC"}"#.to_string(),
        };

        assert!(event.is_function_call_done());
        let (call_id, item_id, args) = event.get_function_call().unwrap();
        assert_eq!(call_id, "call_abc");
        assert_eq!(item_id, "item_789");
        assert_eq!(args, r#"{"location":"NYC"}"#);
    }

    #[test]
    fn test_audio_delta_serialization() {
        let event = RealtimeServerEvent::ResponseAudioDelta {
            event_id: "evt_audio".to_string(),
            response_id: "resp_123".to_string(),
            item_id: "item_456".to_string(),
            output_index: 0,
            content_index: 0,
            delta: "SGVsbG8gV29ybGQ=".to_string(),
        };

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"response.audio.delta\""));
        assert!(json.contains("\"delta\":\"SGVsbG8gV29ybGQ=\""));
        assert!(event.is_delta_event());
    }

    #[test]
    fn test_rate_limits_updated() {
        let event = RealtimeServerEvent::RateLimitsUpdated {
            event_id: "evt_rate".to_string(),
            rate_limits: vec![
                RateLimit {
                    name: "requests".to_string(),
                    limit: 100,
                    remaining: 95,
                    reset_seconds: 60.0,
                },
                RateLimit {
                    name: "tokens".to_string(),
                    limit: 100000,
                    remaining: 50000,
                    reset_seconds: 60.0,
                },
            ],
        };

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"rate_limits.updated\""));
        assert!(json.contains("\"name\":\"requests\""));
    }

    #[test]
    fn test_unknown_event_handling() {
        let json = r#"{"type": "some.future.event", "event_id": "evt_999"}"#;
        let event: RealtimeServerEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, RealtimeServerEvent::Unknown));
        assert_eq!(event.event_type(), "unknown");
    }

    #[test]
    fn test_event_type_categorization() {
        let session = RealtimeServerEvent::SessionCreated {
            event_id: "evt".to_string(),
            session: serde_json::from_str(
                r#"{
                    "id": "sess", "object": "realtime.session", "model": "gpt-4o",
                    "expires_at": 0, "modalities": [], "instructions": "", "voice": "alloy",
                    "input_audio_format": "pcm16", "output_audio_format": "pcm16",
                    "tools": [], "tool_choice": "auto", "temperature": 0.8,
                    "max_response_output_tokens": "inf"
                }"#,
            )
            .unwrap(),
        };
        assert!(session.is_session_event());
        assert!(!session.is_response_event());
        assert!(!session.is_delta_event());

        let text_delta = RealtimeServerEvent::ResponseTextDelta {
            event_id: "evt".to_string(),
            response_id: "resp".to_string(),
            item_id: "item".to_string(),
            output_index: 0,
            content_index: 0,
            delta: "Hello".to_string(),
        };
        assert!(text_delta.is_response_event());
        assert!(text_delta.is_delta_event());
    }
}
