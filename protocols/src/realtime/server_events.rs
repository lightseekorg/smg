//! Server events for the Realtime API.
//!
//! This module contains all event types that can be received from the
//! Realtime API server. There are 46 server event types organized into
//! several categories:
//!
//! - Error events
//! - Session events: `created`, `updated`
//! - Transcription session events: `created`, `updated`
//! - Conversation events: `created`, `item.created`, `item.retrieved`, `item.added`, etc.
//! - Input audio buffer events: `committed`, `cleared`, `speech_started`, `speech_stopped`, etc.
//! - Output audio buffer events: `started`, `stopped`, `cleared`
//! - Response events: `created`, `done`, `output_item.added`, `output_item.done`, etc.
//! - Streaming delta events: `text.delta`, `audio.delta`, `function_call_arguments.delta`, etc.
//! - MCP events: `mcp_call` streaming, `list_tools` lifecycle
//! - Rate limit events

use serde::{Deserialize, Serialize};

use super::{
    conversation::ConversationItem,
    response::{Response, ResponseContentPart},
    session::Session,
    transcription::TranscriptionSession,
};

// ============================================================================
// Server Event Enum (Main Type)
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
    ///
    /// Can be either a realtime session or a transcription session.
    #[serde(rename = "session.created")]
    SessionCreated {
        /// Server-generated event ID
        event_id: String,
        /// The created session (realtime or transcription)
        session: SessionOrTranscription,
    },

    /// Session was updated
    ///
    /// Can be either a realtime session or a transcription session.
    #[serde(rename = "session.updated")]
    SessionUpdated {
        /// Server-generated event ID
        event_id: String,
        /// The updated session (realtime or transcription)
        session: SessionOrTranscription,
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

    /// Conversation item was retrieved
    #[serde(rename = "conversation.item.retrieved")]
    ConversationItemRetrieved {
        /// Server-generated event ID
        event_id: String,
        /// The retrieved item
        item: ConversationItem,
    },

    /// Conversation item was added
    #[serde(rename = "conversation.item.added")]
    ConversationItemAdded {
        /// Server-generated event ID
        event_id: String,
        /// ID of the previous item (for ordering)
        #[serde(skip_serializing_if = "Option::is_none")]
        previous_item_id: Option<String>,
        /// The added item
        item: ConversationItem,
    },

    /// Conversation item is done (fully populated)
    #[serde(rename = "conversation.item.done")]
    ConversationItemDone {
        /// Server-generated event ID
        event_id: String,
        /// ID of the previous item (for ordering)
        #[serde(skip_serializing_if = "Option::is_none")]
        previous_item_id: Option<String>,
        /// The completed item
        item: ConversationItem,
    },

    /// Input audio transcription completed
    ///
    /// Transcription begins when the input audio buffer is committed.
    /// Transcription runs asynchronously with Response creation.
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
        /// Log probabilities of the transcription (if requested via `include`)
        #[serde(skip_serializing_if = "Option::is_none")]
        logprobs: Option<Vec<LogProbProperties>>,
        /// Usage statistics for the transcription (billed per ASR model pricing)
        usage: TranscriptTextUsage,
    },

    /// Input audio transcription delta
    #[serde(rename = "conversation.item.input_audio_transcription.delta")]
    ConversationItemInputAudioTranscriptionDelta {
        /// Server-generated event ID
        event_id: String,
        /// ID of the item containing the audio
        item_id: String,
        /// Index of the content part
        #[serde(skip_serializing_if = "Option::is_none")]
        content_index: Option<u32>,
        /// The transcription delta
        #[serde(skip_serializing_if = "Option::is_none")]
        delta: Option<String>,
        /// Log probabilities of the transcription (if requested via `include`)
        #[serde(skip_serializing_if = "Option::is_none")]
        logprobs: Option<Vec<LogProbProperties>>,
    },

    /// Input audio transcription segment
    #[serde(rename = "conversation.item.input_audio_transcription.segment")]
    ConversationItemInputAudioTranscriptionSegment {
        /// Server-generated event ID
        event_id: String,
        /// ID of the item containing the audio
        item_id: String,
        /// Index of the content part
        content_index: u32,
        /// The text for this segment
        text: String,
        /// The segment identifier
        id: String,
        /// The detected speaker label for this segment
        speaker: String,
        /// Start time of the segment in seconds
        start: f32,
        /// End time of the segment in seconds
        end: f32,
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

    /// Input audio buffer timeout triggered
    ///
    /// Returned when the Server VAD timeout is triggered for the input audio buffer.
    /// This is configured with `idle_timeout_ms` in the `turn_detection` settings.
    #[serde(rename = "input_audio_buffer.timeout_triggered")]
    InputAudioBufferTimeoutTriggered {
        /// Server-generated event ID
        event_id: String,
        /// Millisecond offset of audio after the last model response playback time
        audio_start_ms: u32,
        /// Millisecond offset of audio at the time the timeout was triggered
        audio_end_ms: u32,
        /// The ID of the item associated with this segment
        item_id: String,
    },

    /// DTMF event received in input audio buffer
    ///
    /// **SIP Only:** Returned when a DTMF event is received. A DTMF event
    /// represents a telephone keypad press (0-9, *, #, A-D).
    #[serde(rename = "input_audio_buffer.dtmf_event_received")]
    InputAudioBufferDtmfEventReceived {
        /// The telephone keypad that was pressed by the user (0-9, *, #, A-D)
        event: String,
        /// UTC Unix Timestamp when DTMF event was received by server
        received_at: u64,
    },

    // === Output Audio Buffer Events ===
    /// Output audio buffer started playing
    #[serde(rename = "output_audio_buffer.started")]
    OutputAudioBufferStarted {
        /// Server-generated event ID
        event_id: String,
        /// ID of the response
        response_id: String,
    },

    /// Output audio buffer stopped playing
    #[serde(rename = "output_audio_buffer.stopped")]
    OutputAudioBufferStopped {
        /// Server-generated event ID
        event_id: String,
        /// ID of the response
        response_id: String,
    },

    /// Output audio buffer was cleared
    #[serde(rename = "output_audio_buffer.cleared")]
    OutputAudioBufferCleared {
        /// Server-generated event ID
        event_id: String,
        /// ID of the response
        response_id: String,
    },

    // === Transcription Session Events ===
    /// Transcription session was created
    #[serde(rename = "transcription_session.created")]
    TranscriptionSessionCreated {
        /// Server-generated event ID
        event_id: String,
        /// The created transcription session
        session: TranscriptionSession,
    },

    /// Transcription session was updated
    #[serde(rename = "transcription_session.updated")]
    TranscriptionSessionUpdated {
        /// Server-generated event ID
        event_id: String,
        /// The updated transcription session
        session: TranscriptionSession,
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
        part: ResponseContentPart,
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
        part: ResponseContentPart,
    },

    // === Response Streaming Delta Events ===
    /// Text delta in response
    #[serde(rename = "response.output_text.delta")]
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
    #[serde(rename = "response.output_text.done")]
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
    #[serde(rename = "response.output_audio_transcript.delta")]
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
    #[serde(rename = "response.output_audio_transcript.done")]
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
    #[serde(rename = "response.output_audio.delta")]
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
    #[serde(rename = "response.output_audio.done")]
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
    ///
    /// Also emitted when a Response is interrupted, incomplete, or cancelled.
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
        /// The name of the function that was called
        name: String,
        /// The final arguments (JSON string)
        arguments: String,
    },

    // === MCP Call Events ===
    /// MCP call arguments delta
    #[serde(rename = "response.mcp_call_arguments.delta")]
    ResponseMcpCallArgumentsDelta {
        /// Server-generated event ID
        event_id: String,
        /// ID of the response
        response_id: String,
        /// ID of the output item
        item_id: String,
        /// Index of the output item
        output_index: u32,
        /// The arguments delta (JSON fragment)
        delta: String,
        /// Obfuscation details (if applicable)
        #[serde(skip_serializing_if = "Option::is_none")]
        obfuscation: Option<String>,
    },

    /// MCP call arguments are done
    #[serde(rename = "response.mcp_call_arguments.done")]
    ResponseMcpCallArgumentsDone {
        /// Server-generated event ID
        event_id: String,
        /// ID of the response
        response_id: String,
        /// ID of the MCP tool call item
        item_id: String,
        /// Index of the output item in the response
        output_index: u32,
        /// The final JSON-encoded arguments string
        arguments: String,
    },

    /// MCP call is in progress
    #[serde(rename = "response.mcp_call.in_progress")]
    ResponseMcpCallInProgress {
        /// Server-generated event ID
        event_id: String,
        /// ID of the MCP tool call item
        item_id: String,
        /// Index of the output item in the response
        output_index: u32,
    },

    /// MCP call completed successfully
    #[serde(rename = "response.mcp_call.completed")]
    ResponseMcpCallCompleted {
        /// Server-generated event ID
        event_id: String,
        /// ID of the MCP tool call item
        item_id: String,
        /// Index of the output item in the response
        output_index: u32,
    },

    /// MCP call failed
    #[serde(rename = "response.mcp_call.failed")]
    ResponseMcpCallFailed {
        /// Server-generated event ID
        event_id: String,
        /// ID of the MCP tool call item
        item_id: String,
        /// Index of the output item in the response
        output_index: u32,
    },

    // === MCP List Tools Events ===
    /// MCP list tools is in progress
    #[serde(rename = "mcp_list_tools.in_progress")]
    McpListToolsInProgress {
        /// Server-generated event ID
        event_id: String,
        /// ID of the MCP list tools item
        item_id: String,
    },

    /// MCP list tools completed
    #[serde(rename = "mcp_list_tools.completed")]
    McpListToolsCompleted {
        /// Server-generated event ID
        event_id: String,
        /// ID of the MCP list tools item
        item_id: String,
    },

    /// MCP list tools failed
    #[serde(rename = "mcp_list_tools.failed")]
    McpListToolsFailed {
        /// Server-generated event ID
        event_id: String,
        /// ID of the MCP list tools item
        item_id: String,
    },

    // === Rate Limit Events ===
    /// Rate limits were updated
    ///
    /// Emitted at the beginning of a Response to indicate the updated rate limits.
    /// When a Response is created some tokens will be "reserved" for the output
    /// tokens, the rate limits shown here reflect that reservation, which is then
    /// adjusted accordingly once the Response is completed.
    #[serde(rename = "rate_limits.updated")]
    RateLimitsUpdated {
        /// Server-generated event ID
        event_id: String,
        /// List of rate limit information
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
            Self::ConversationItemRetrieved { .. } => "conversation.item.retrieved",
            Self::ConversationItemAdded { .. } => "conversation.item.added",
            Self::ConversationItemDone { .. } => "conversation.item.done",
            Self::ConversationItemInputAudioTranscriptionDelta { .. } => {
                "conversation.item.input_audio_transcription.delta"
            }
            Self::ConversationItemInputAudioTranscriptionSegment { .. } => {
                "conversation.item.input_audio_transcription.segment"
            }
            Self::InputAudioBufferCommitted { .. } => "input_audio_buffer.committed",
            Self::InputAudioBufferCleared { .. } => "input_audio_buffer.cleared",
            Self::InputAudioBufferSpeechStarted { .. } => "input_audio_buffer.speech_started",
            Self::InputAudioBufferSpeechStopped { .. } => "input_audio_buffer.speech_stopped",
            Self::InputAudioBufferTimeoutTriggered { .. } => "input_audio_buffer.timeout_triggered",
            Self::InputAudioBufferDtmfEventReceived { .. } => {
                "input_audio_buffer.dtmf_event_received"
            }
            Self::OutputAudioBufferStarted { .. } => "output_audio_buffer.started",
            Self::OutputAudioBufferStopped { .. } => "output_audio_buffer.stopped",
            Self::OutputAudioBufferCleared { .. } => "output_audio_buffer.cleared",
            Self::TranscriptionSessionCreated { .. } => "transcription_session.created",
            Self::TranscriptionSessionUpdated { .. } => "transcription_session.updated",
            Self::ResponseCreated { .. } => "response.created",
            Self::ResponseDone { .. } => "response.done",
            Self::ResponseOutputItemAdded { .. } => "response.output_item.added",
            Self::ResponseOutputItemDone { .. } => "response.output_item.done",
            Self::ResponseContentPartAdded { .. } => "response.content_part.added",
            Self::ResponseContentPartDone { .. } => "response.content_part.done",
            Self::ResponseTextDelta { .. } => "response.output_text.delta",
            Self::ResponseTextDone { .. } => "response.output_text.done",
            Self::ResponseAudioTranscriptDelta { .. } => "response.output_audio_transcript.delta",
            Self::ResponseAudioTranscriptDone { .. } => "response.output_audio_transcript.done",
            Self::ResponseAudioDelta { .. } => "response.output_audio.delta",
            Self::ResponseAudioDone { .. } => "response.output_audio.done",
            Self::ResponseFunctionCallArgumentsDelta { .. } => {
                "response.function_call_arguments.delta"
            }
            Self::ResponseFunctionCallArgumentsDone { .. } => {
                "response.function_call_arguments.done"
            }
            Self::ResponseMcpCallArgumentsDelta { .. } => "response.mcp_call_arguments.delta",
            Self::ResponseMcpCallArgumentsDone { .. } => "response.mcp_call_arguments.done",
            Self::ResponseMcpCallInProgress { .. } => "response.mcp_call.in_progress",
            Self::ResponseMcpCallCompleted { .. } => "response.mcp_call.completed",
            Self::ResponseMcpCallFailed { .. } => "response.mcp_call.failed",
            Self::McpListToolsInProgress { .. } => "mcp_list_tools.in_progress",
            Self::McpListToolsCompleted { .. } => "mcp_list_tools.completed",
            Self::McpListToolsFailed { .. } => "mcp_list_tools.failed",
            Self::RateLimitsUpdated { .. } => "rate_limits.updated",
            Self::Unknown => "unknown",
        }
    }

    /// Get the event ID if present
    pub fn event_id(&self) -> Option<&str> {
        let event_id = match self {
            Self::Error { event_id, .. } => event_id,
            Self::SessionCreated { event_id, .. } => event_id,
            Self::SessionUpdated { event_id, .. } => event_id,
            Self::ConversationCreated { event_id, .. } => event_id,
            Self::ConversationItemCreated { event_id, .. } => event_id,
            Self::ConversationItemInputAudioTranscriptionCompleted { event_id, .. } => event_id,
            Self::ConversationItemInputAudioTranscriptionFailed { event_id, .. } => event_id,
            Self::ConversationItemTruncated { event_id, .. } => event_id,
            Self::ConversationItemDeleted { event_id, .. } => event_id,
            Self::ConversationItemRetrieved { event_id, .. } => event_id,
            Self::ConversationItemAdded { event_id, .. } => event_id,
            Self::ConversationItemDone { event_id, .. } => event_id,
            Self::ConversationItemInputAudioTranscriptionDelta { event_id, .. } => event_id,
            Self::ConversationItemInputAudioTranscriptionSegment { event_id, .. } => event_id,
            Self::InputAudioBufferCommitted { event_id, .. } => event_id,
            Self::InputAudioBufferCleared { event_id, .. } => event_id,
            Self::InputAudioBufferSpeechStarted { event_id, .. } => event_id,
            Self::InputAudioBufferSpeechStopped { event_id, .. } => event_id,
            Self::InputAudioBufferTimeoutTriggered { event_id, .. } => event_id,
            Self::OutputAudioBufferStarted { event_id, .. } => event_id,
            Self::OutputAudioBufferStopped { event_id, .. } => event_id,
            Self::OutputAudioBufferCleared { event_id, .. } => event_id,
            Self::TranscriptionSessionCreated { event_id, .. } => event_id,
            Self::TranscriptionSessionUpdated { event_id, .. } => event_id,
            Self::ResponseCreated { event_id, .. } => event_id,
            Self::ResponseDone { event_id, .. } => event_id,
            Self::ResponseOutputItemAdded { event_id, .. } => event_id,
            Self::ResponseOutputItemDone { event_id, .. } => event_id,
            Self::ResponseContentPartAdded { event_id, .. } => event_id,
            Self::ResponseContentPartDone { event_id, .. } => event_id,
            Self::ResponseTextDelta { event_id, .. } => event_id,
            Self::ResponseTextDone { event_id, .. } => event_id,
            Self::ResponseAudioTranscriptDelta { event_id, .. } => event_id,
            Self::ResponseAudioTranscriptDone { event_id, .. } => event_id,
            Self::ResponseAudioDelta { event_id, .. } => event_id,
            Self::ResponseAudioDone { event_id, .. } => event_id,
            Self::ResponseFunctionCallArgumentsDelta { event_id, .. } => event_id,
            Self::ResponseFunctionCallArgumentsDone { event_id, .. } => event_id,
            Self::ResponseMcpCallArgumentsDelta { event_id, .. } => event_id,
            Self::ResponseMcpCallArgumentsDone { event_id, .. } => event_id,
            Self::ResponseMcpCallInProgress { event_id, .. } => event_id,
            Self::ResponseMcpCallCompleted { event_id, .. } => event_id,
            Self::ResponseMcpCallFailed { event_id, .. } => event_id,
            Self::McpListToolsInProgress { event_id, .. } => event_id,
            Self::McpListToolsCompleted { event_id, .. } => event_id,
            Self::McpListToolsFailed { event_id, .. } => event_id,
            Self::RateLimitsUpdated { event_id, .. } => event_id,
            Self::InputAudioBufferDtmfEventReceived { .. } | Self::Unknown => return None,
        };
        Some(event_id.as_str())
    }

    /// Check if this is a function call completion event (for MCP interception)
    pub fn is_function_call_done(&self) -> bool {
        matches!(self, Self::ResponseFunctionCallArgumentsDone { .. })
    }

    /// Extract function call details if this is a function call done event
    ///
    /// Returns (call_id, name, item_id, arguments) tuple
    pub fn get_function_call(&self) -> Option<(&str, &str, &str, &str)> {
        match self {
            Self::ResponseFunctionCallArgumentsDone {
                call_id,
                name,
                item_id,
                arguments,
                ..
            } => Some((
                call_id.as_str(),
                name.as_str(),
                item_id.as_str(),
                arguments.as_str(),
            )),
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
        matches!(
            self,
            Self::SessionCreated { .. } | Self::SessionUpdated { .. }
        )
    }

    /// Check if this is a transcription session event
    pub fn is_transcription_session_event(&self) -> bool {
        matches!(
            self,
            Self::TranscriptionSessionCreated { .. } | Self::TranscriptionSessionUpdated { .. }
        )
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
                | Self::ResponseMcpCallArgumentsDelta { .. }
                | Self::ResponseMcpCallArgumentsDone { .. }
                | Self::ResponseMcpCallInProgress { .. }
                | Self::ResponseMcpCallCompleted { .. }
                | Self::ResponseMcpCallFailed { .. }
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
                | Self::ResponseMcpCallArgumentsDelta { .. }
                | Self::ConversationItemInputAudioTranscriptionDelta { .. }
        )
    }

    /// Check if this is an MCP-related event
    pub fn is_mcp_event(&self) -> bool {
        matches!(
            self,
            Self::ResponseMcpCallArgumentsDelta { .. }
                | Self::ResponseMcpCallArgumentsDone { .. }
                | Self::ResponseMcpCallInProgress { .. }
                | Self::ResponseMcpCallCompleted { .. }
                | Self::ResponseMcpCallFailed { .. }
                | Self::McpListToolsInProgress { .. }
                | Self::McpListToolsCompleted { .. }
                | Self::McpListToolsFailed { .. }
        )
    }

    /// Check if this is an output audio buffer event
    pub fn is_output_audio_buffer_event(&self) -> bool {
        matches!(
            self,
            Self::OutputAudioBufferStarted { .. }
                | Self::OutputAudioBufferStopped { .. }
                | Self::OutputAudioBufferCleared { .. }
        )
    }
}

// ============================================================================
// Supporting Types
// ============================================================================

/// Session type in server events.
///
/// Can be either a realtime session or a transcription session,
/// distinguished by the `type` field.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SessionOrTranscription {
    /// Realtime session (type: "realtime")
    Realtime(Box<Session>),
    /// Transcription session (type: "transcription")
    Transcription(Box<TranscriptionSession>),
}

impl SessionOrTranscription {
    /// Get the session ID
    pub fn id(&self) -> &str {
        match self {
            Self::Realtime(s) => &s.id,
            Self::Transcription(s) => &s.id,
        }
    }

    /// Check if this is a realtime session
    pub fn is_realtime(&self) -> bool {
        matches!(self, Self::Realtime(_))
    }

    /// Check if this is a transcription session
    pub fn is_transcription(&self) -> bool {
        matches!(self, Self::Transcription(_))
    }

    /// Get as a realtime session, if applicable
    pub fn as_realtime(&self) -> Option<&Session> {
        match self {
            Self::Realtime(s) => Some(s),
            _ => None,
        }
    }

    /// Get as a transcription session, if applicable
    pub fn as_transcription(&self) -> Option<&TranscriptionSession> {
        match self {
            Self::Transcription(s) => Some(s),
            _ => None,
        }
    }
}

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
    #[serde(default = "conversation_object_type")]
    pub object: String,
}

fn conversation_object_type() -> String {
    "realtime.conversation".to_string()
}

/// Rate limit category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RateLimitName {
    /// Request rate limit
    Requests,
    /// Token rate limit
    Tokens,
}

/// Rate limit information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimit {
    /// The rate limit category
    pub name: RateLimitName,
    /// Maximum allowed
    pub limit: u32,
    /// Remaining in current window
    pub remaining: u32,
    /// Seconds until reset
    pub reset_seconds: f32,
}

/// Log probability properties for transcription.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogProbProperties {
    /// The token
    pub token: String,
    /// Log probability of the token
    pub logprob: f32,
    /// Byte representation of the token
    pub bytes: Vec<u8>,
}

/// Input token details for transcription usage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptInputTokenDetails {
    /// Number of text tokens billed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_tokens: Option<u32>,
    /// Number of audio tokens billed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_tokens: Option<u32>,
}

/// Usage statistics for transcription.
///
/// Can be either token-based or duration-based billing.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TranscriptTextUsage {
    /// Token-based usage (for models billed by token usage)
    Tokens {
        /// Total number of tokens used (input + output)
        total_tokens: u32,
        /// Number of input tokens billed
        input_tokens: u32,
        /// Number of output tokens generated
        output_tokens: u32,
        /// Details about the input tokens billed
        #[serde(skip_serializing_if = "Option::is_none")]
        input_token_details: Option<TranscriptInputTokenDetails>,
    },
    /// Duration-based usage (for models billed by audio input duration)
    Duration {
        /// Duration of the input audio in seconds
        seconds: f32,
    },
}

// ============================================================================
// Tests
// ============================================================================

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
                "type": "realtime",
                "model": "gpt-4o-realtime-preview",
                "expires_at": 1699999999,
                "output_modalities": ["text", "audio"],
                "instructions": "",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24000}
                    },
                    "output": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "voice": "alloy"
                    }
                },
                "tools": [],
                "tool_choice": "auto",
                "max_output_tokens": "inf"
            }
        }"#;

        let event: RealtimeServerEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, RealtimeServerEvent::SessionCreated { .. }));
        assert_eq!(event.event_id(), Some("evt_001"));

        // Verify it's a realtime session
        if let RealtimeServerEvent::SessionCreated { session, .. } = &event {
            assert!(session.is_realtime());
            assert_eq!(session.id(), "sess_123");
        }
    }

    #[test]
    fn test_session_created_transcription() {
        let json = r#"{
            "type": "session.created",
            "event_id": "evt_002",
            "session": {
                "id": "trans_456",
                "type": "transcription",
                "object": "realtime.transcription_session",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24000}
                    }
                }
            }
        }"#;

        let event: RealtimeServerEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, RealtimeServerEvent::SessionCreated { .. }));

        // Verify it's a transcription session
        if let RealtimeServerEvent::SessionCreated { session, .. } = &event {
            assert!(session.is_transcription());
            assert_eq!(session.id(), "trans_456");
        }
    }

    #[test]
    fn test_function_call_done_extraction() {
        let event = RealtimeServerEvent::ResponseFunctionCallArgumentsDone {
            event_id: "evt_456".to_string(),
            response_id: "resp_123".to_string(),
            item_id: "item_789".to_string(),
            output_index: 0,
            call_id: "call_abc".to_string(),
            name: "get_weather".to_string(),
            arguments: r#"{"location":"NYC"}"#.to_string(),
        };

        assert!(event.is_function_call_done());
        let (call_id, name, item_id, args) = event.get_function_call().unwrap();
        assert_eq!(call_id, "call_abc");
        assert_eq!(name, "get_weather");
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
        assert!(json.contains("\"type\":\"response.output_audio.delta\""));
        assert!(json.contains("\"delta\":\"SGVsbG8gV29ybGQ=\""));
        assert!(event.is_delta_event());
    }

    #[test]
    fn test_rate_limits_updated() {
        let event = RealtimeServerEvent::RateLimitsUpdated {
            event_id: "evt_rate".to_string(),
            rate_limits: vec![
                RateLimit {
                    name: RateLimitName::Requests,
                    limit: 100,
                    remaining: 95,
                    reset_seconds: 60.0,
                },
                RateLimit {
                    name: RateLimitName::Tokens,
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
                    "id": "sess", "object": "realtime.session", "type": "realtime",
                    "model": "gpt-4o", "expires_at": 0, "output_modalities": [],
                    "instructions": "", "audio": {},
                    "tools": [], "tool_choice": "auto",
                    "max_output_tokens": "inf"
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

    #[test]
    fn test_mcp_events() {
        // Test MCP call arguments delta
        let mcp_delta = RealtimeServerEvent::ResponseMcpCallArgumentsDelta {
            event_id: "evt_mcp".to_string(),
            response_id: "resp_123".to_string(),
            item_id: "item_456".to_string(),
            output_index: 0,
            delta: r#"{"arg"#.to_string(),
            obfuscation: None,
        };
        let json = serde_json::to_string(&mcp_delta).unwrap();
        assert!(json.contains("\"type\":\"response.mcp_call_arguments.delta\""));
        assert!(mcp_delta.is_mcp_event());
        assert!(mcp_delta.is_delta_event());
        assert!(mcp_delta.is_response_event());

        // Test MCP list tools completed
        let mcp_tools = RealtimeServerEvent::McpListToolsCompleted {
            event_id: "evt_tools".to_string(),
            item_id: "item_mcp".to_string(),
        };
        let json = serde_json::to_string(&mcp_tools).unwrap();
        assert!(json.contains("\"type\":\"mcp_list_tools.completed\""));
        assert!(json.contains("\"item_id\":\"item_mcp\""));
        assert!(mcp_tools.is_mcp_event());
    }

    #[test]
    fn test_output_audio_buffer_events() {
        let started = RealtimeServerEvent::OutputAudioBufferStarted {
            event_id: "evt_start".to_string(),
            response_id: "resp_123".to_string(),
        };
        let json = serde_json::to_string(&started).unwrap();
        assert!(json.contains("\"type\":\"output_audio_buffer.started\""));
        assert!(started.is_output_audio_buffer_event());

        let cleared = RealtimeServerEvent::OutputAudioBufferCleared {
            event_id: "evt_clear".to_string(),
            response_id: "resp_456".to_string(),
        };
        assert!(cleared.is_output_audio_buffer_event());
        assert_eq!(cleared.event_type(), "output_audio_buffer.cleared");
    }

    #[test]
    fn test_transcription_session_events() {
        let json = r#"{
            "type": "transcription_session.created",
            "event_id": "evt_trans",
            "session": {
                "id": "trans_123",
                "type": "transcription",
                "object": "realtime.transcription_session",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24000}
                    }
                }
            }
        }"#;
        let event: RealtimeServerEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(
            event,
            RealtimeServerEvent::TranscriptionSessionCreated { .. }
        ));
        assert!(event.is_transcription_session_event());
        assert_eq!(event.event_type(), "transcription_session.created");
    }

    #[test]
    fn test_input_audio_buffer_new_events() {
        let timeout = RealtimeServerEvent::InputAudioBufferTimeoutTriggered {
            event_id: "evt_timeout".to_string(),
            audio_start_ms: 1000,
            audio_end_ms: 2000,
            item_id: "item_123".to_string(),
        };
        let json = serde_json::to_string(&timeout).unwrap();
        assert!(json.contains("\"type\":\"input_audio_buffer.timeout_triggered\""));
        assert!(json.contains("\"audio_start_ms\":1000"));
        assert!(json.contains("\"item_id\":\"item_123\""));

        let dtmf = RealtimeServerEvent::InputAudioBufferDtmfEventReceived {
            event: "5".to_string(),
            received_at: 1700000000,
        };
        let json = serde_json::to_string(&dtmf).unwrap();
        assert!(json.contains("\"type\":\"input_audio_buffer.dtmf_event_received\""));
        assert!(json.contains("\"event\":\"5\""));
        assert!(json.contains("\"received_at\":1700000000"));
    }

    #[test]
    fn test_conversation_item_new_events() {
        // Test item retrieved
        let json = r#"{
            "type": "conversation.item.retrieved",
            "event_id": "evt_retr",
            "item": {
                "id": "item_123",
                "type": "message",
                "role": "user",
                "content": []
            }
        }"#;
        let event: RealtimeServerEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(
            event,
            RealtimeServerEvent::ConversationItemRetrieved { .. }
        ));
        assert_eq!(event.event_type(), "conversation.item.retrieved");

        // Test transcription delta
        let delta = RealtimeServerEvent::ConversationItemInputAudioTranscriptionDelta {
            event_id: "evt_delta".to_string(),
            item_id: "item_456".to_string(),
            content_index: Some(0),
            delta: Some("Hello ".to_string()),
            logprobs: None,
        };
        assert!(delta.is_delta_event());
        assert_eq!(
            delta.event_type(),
            "conversation.item.input_audio_transcription.delta"
        );
    }
}
