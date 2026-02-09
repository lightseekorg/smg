// OpenAI Realtime API wire-format event types
// https://platform.openai.com/docs/api-reference/realtime
//
// This module defines the serializable/deserializable event structures
// for both client-to-server and server-to-client messages sent over
// WebSocket, WebRTC, or SIP connections.
//
// Session configuration types live in `realtime_session`.
// Conversation item types live in `realtime_conversation`.
// Response and usage types live in `realtime_response`.
// Event type string constants live in `event_types`.

use serde::{Deserialize, Serialize};

use crate::{
    event_types::{RealtimeClientEvent, RealtimeServerEvent},
    realtime_conversation::RealtimeConversationItem,
    realtime_response::{RealtimeResponse, ResponseCreateParams},
    realtime_session::{
        AudioTranscription, NoiseReduction, RealtimeIncludeOption, RealtimeSessionConfig,
        RealtimeTranscriptionSessionConfig, TurnDetection,
    },
};

// ============================================================================
// Client Events
// ============================================================================

/// A client-to-server event in the OpenAI Realtime API.
///
/// Sent by the client over WebSocket, WebRTC, or SIP connections.
/// Discriminated by the `type` field in the JSON wire format.
///
/// Large payloads (`SessionConfig` 624 B, `ResponseCreateParams` 384 B) are
/// `Box`-ed so the enum stays ≈224 bytes instead of ≈648.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ClientEvent {
    // ---- Session ----
    /// Update the session configuration.
    #[serde(rename = "session.update")]
    SessionUpdate {
        event_id: Option<String>,
        session: Box<SessionConfig>,
    },

    // ---- Conversation items ----
    /// Add a new item to the conversation.
    #[serde(rename = "conversation.item.create")]
    ConversationItemCreate {
        event_id: Option<String>,
        previous_item_id: Option<String>,
        item: RealtimeConversationItem,
    },

    /// Remove an item from the conversation history.
    #[serde(rename = "conversation.item.delete")]
    ConversationItemDelete {
        event_id: Option<String>,
        item_id: String,
    },

    /// Retrieve the server's representation of a conversation item.
    #[serde(rename = "conversation.item.retrieve")]
    ConversationItemRetrieve {
        event_id: Option<String>,
        item_id: String,
    },

    /// Truncate a previous assistant message's audio.
    #[serde(rename = "conversation.item.truncate")]
    ConversationItemTruncate {
        event_id: Option<String>,
        item_id: String,
        content_index: u32,
        audio_end_ms: u32,
    },

    // ---- Input audio buffer ----
    /// Append audio bytes to the input audio buffer.
    ///
    /// WARNING: `audio` contains a base64 audio blob that can be very large.
    /// Avoid logging this variant with `Debug` in production; prefer
    /// `event_type()` for structured logging.
    #[serde(rename = "input_audio_buffer.append")]
    InputAudioBufferAppend {
        event_id: Option<String>,
        audio: String,
    },

    /// Clear the input audio buffer.
    #[serde(rename = "input_audio_buffer.clear")]
    InputAudioBufferClear { event_id: Option<String> },

    /// Commit the input audio buffer as a user message.
    #[serde(rename = "input_audio_buffer.commit")]
    InputAudioBufferCommit { event_id: Option<String> },

    // ---- Output audio buffer (WebRTC/SIP only) ----
    /// Cut off the current audio response.
    #[serde(rename = "output_audio_buffer.clear")]
    OutputAudioBufferClear { event_id: Option<String> },

    // ---- Response ----
    /// Cancel an in-progress response.
    #[serde(rename = "response.cancel")]
    ResponseCancel {
        event_id: Option<String>,
        response_id: Option<String>,
    },

    /// Trigger model inference to create a response.
    #[serde(rename = "response.create")]
    ResponseCreate {
        event_id: Option<String>,
        response: Option<Box<ResponseCreateParams>>,
    },

    // ---- Transcription session ----
    /// Update a transcription session configuration.
    #[serde(rename = "transcription_session.update")]
    TranscriptionSessionUpdate {
        event_id: Option<String>,
        session: TranscriptionSessionUpdateConfig,
    },

    // ---- Unknown ----
    /// Unrecognized event type — preserves the full JSON payload for
    /// pass-through forwarding.
    #[serde(skip)]
    Unknown(serde_json::Value),
}

impl ClientEvent {
    /// Deserialize a JSON string into a `ClientEvent`, falling back to
    /// `Unknown` for unrecognized event types instead of returning an error.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        match serde_json::from_str::<ClientEvent>(json) {
            Ok(event) => Ok(event),
            Err(_) => {
                let value: serde_json::Value = serde_json::from_str(json)?;
                Ok(ClientEvent::Unknown(value))
            }
        }
    }

    /// Serialize to a JSON string. For `Unknown` variants, serializes the
    /// preserved `Value` directly; for known variants, uses the derived
    /// `Serialize` impl.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        match self {
            ClientEvent::Unknown(value) => serde_json::to_string(value),
            _ => serde_json::to_string(self),
        }
    }

    /// Returns the event type string (e.g. `"session.update"`).
    ///
    /// For unknown events, returns `"unknown"`.
    pub fn event_type(&self) -> &str {
        match self {
            ClientEvent::Unknown(_) => "unknown",
            _ => RealtimeClientEvent::from(self).as_str(),
        }
    }
}

impl From<&ClientEvent> for RealtimeClientEvent {
    fn from(event: &ClientEvent) -> Self {
        match event {
            ClientEvent::SessionUpdate { .. } => Self::SessionUpdate,
            ClientEvent::ConversationItemCreate { .. } => Self::ConversationItemCreate,
            ClientEvent::ConversationItemDelete { .. } => Self::ConversationItemDelete,
            ClientEvent::ConversationItemRetrieve { .. } => Self::ConversationItemRetrieve,
            ClientEvent::ConversationItemTruncate { .. } => Self::ConversationItemTruncate,
            ClientEvent::InputAudioBufferAppend { .. } => Self::InputAudioBufferAppend,
            ClientEvent::InputAudioBufferClear { .. } => Self::InputAudioBufferClear,
            ClientEvent::InputAudioBufferCommit { .. } => Self::InputAudioBufferCommit,
            ClientEvent::OutputAudioBufferClear { .. } => Self::OutputAudioBufferClear,
            ClientEvent::ResponseCancel { .. } => Self::ResponseCancel,
            ClientEvent::ResponseCreate { .. } => Self::ResponseCreate,
            ClientEvent::TranscriptionSessionUpdate { .. } => Self::TranscriptionSessionUpdate,
            ClientEvent::Unknown(_) => Self::SessionUpdate, // fallback; callers should check Unknown first
        }
    }
}

// ============================================================================
// Server Events
// ============================================================================

/// A server-to-client event in the OpenAI Realtime API.
///
/// Sent by the server over WebSocket, WebRTC, or SIP connections.
/// Discriminated by the `type` field in the JSON wire format.
///
/// Large payloads (`SessionConfig` 624 B, `RealtimeResponse` 352 B) are
/// `Box`-ed so the enum stays ≈232 bytes instead of ≈656.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ServerEvent {
    // ---- Session events ----
    /// Emitted when a new connection is established with the default session config.
    #[serde(rename = "session.created")]
    SessionCreated {
        event_id: String,
        session: Box<SessionConfig>,
    },

    /// Emitted after a successful `session.update`.
    #[serde(rename = "session.updated")]
    SessionUpdated {
        event_id: String,
        session: Box<SessionConfig>,
    },

    // ---- Conversation events ----
    /// Emitted when a conversation is created (right after session creation).
    #[serde(rename = "conversation.created")]
    ConversationCreated {
        event_id: String,
        conversation: ConversationInfo,
    },

    /// Emitted when a conversation item is created (legacy event).
    #[serde(rename = "conversation.item.created")]
    ConversationItemCreated {
        event_id: String,
        previous_item_id: Option<String>,
        item: RealtimeConversationItem,
    },

    /// Emitted when an item is added to the default conversation.
    #[serde(rename = "conversation.item.added")]
    ConversationItemAdded {
        event_id: String,
        previous_item_id: Option<String>,
        item: RealtimeConversationItem,
    },

    /// Emitted when a conversation item is finalized.
    #[serde(rename = "conversation.item.done")]
    ConversationItemDone {
        event_id: String,
        previous_item_id: Option<String>,
        item: RealtimeConversationItem,
    },

    /// Emitted when a conversation item is deleted.
    #[serde(rename = "conversation.item.deleted")]
    ConversationItemDeleted { event_id: String, item_id: String },

    /// Emitted in response to `conversation.item.retrieve`.
    #[serde(rename = "conversation.item.retrieved")]
    ConversationItemRetrieved {
        event_id: String,
        item: RealtimeConversationItem,
    },

    /// Emitted when an assistant audio message item is truncated.
    #[serde(rename = "conversation.item.truncated")]
    ConversationItemTruncated {
        event_id: String,
        item_id: String,
        content_index: u32,
        audio_end_ms: u32,
    },

    // ---- Input audio transcription events ----
    /// Emitted when input audio transcription completes.
    #[serde(rename = "conversation.item.input_audio_transcription.completed")]
    InputAudioTranscriptionCompleted {
        event_id: String,
        item_id: String,
        content_index: u32,
        transcript: String,
        logprobs: Option<Vec<LogProb>>,
        usage: TranscriptionUsage,
    },

    /// Emitted with incremental transcription results.
    #[serde(rename = "conversation.item.input_audio_transcription.delta")]
    InputAudioTranscriptionDelta {
        event_id: String,
        item_id: String,
        content_index: Option<u32>,
        delta: Option<String>,
        logprobs: Option<Vec<LogProb>>,
    },

    /// Emitted when input audio transcription fails.
    #[serde(rename = "conversation.item.input_audio_transcription.failed")]
    InputAudioTranscriptionFailed {
        event_id: String,
        item_id: String,
        content_index: u32,
        error: TranscriptionError,
    },

    /// Emitted when an input audio transcription segment is identified
    /// (used with diarization models).
    #[serde(rename = "conversation.item.input_audio_transcription.segment")]
    InputAudioTranscriptionSegment {
        event_id: String,
        item_id: String,
        content_index: u32,
        text: String,
        id: String,
        speaker: String,
        start: f32,
        end: f32,
    },

    // ---- Input audio buffer events ----
    /// Emitted when the input audio buffer is cleared.
    #[serde(rename = "input_audio_buffer.cleared")]
    InputAudioBufferCleared { event_id: String },

    /// Emitted when the input audio buffer is committed.
    #[serde(rename = "input_audio_buffer.committed")]
    InputAudioBufferCommitted {
        event_id: String,
        previous_item_id: Option<String>,
        item_id: String,
    },

    /// Emitted when speech is detected in the audio buffer (server VAD mode).
    #[serde(rename = "input_audio_buffer.speech_started")]
    InputAudioBufferSpeechStarted {
        event_id: String,
        audio_start_ms: u32,
        item_id: String,
    },

    /// Emitted when the end of speech is detected (server VAD mode).
    #[serde(rename = "input_audio_buffer.speech_stopped")]
    InputAudioBufferSpeechStopped {
        event_id: String,
        audio_end_ms: u32,
        item_id: String,
    },

    /// Emitted when the VAD idle timeout triggers.
    #[serde(rename = "input_audio_buffer.timeout_triggered")]
    InputAudioBufferTimeoutTriggered {
        event_id: String,
        audio_start_ms: u32,
        audio_end_ms: u32,
        item_id: String,
    },

    /// **SIP Only:** Emitted when a DTMF keypad event is received.
    ///
    /// NOTE: This is the only server event without an `event_id` field per the
    /// OpenAI spec. Downstream code that generically extracts `event_id` from
    /// all server events must handle this variant as a special case.
    #[serde(rename = "input_audio_buffer.dtmf_event_received")]
    InputAudioBufferDtmfEventReceived { event: String, received_at: i64 },

    // ---- Output audio buffer events (WebRTC/SIP only) ----
    /// Emitted when the server begins streaming audio to the client.
    #[serde(rename = "output_audio_buffer.started")]
    OutputAudioBufferStarted {
        event_id: String,
        response_id: String,
    },

    /// Emitted when the output audio buffer has been completely drained.
    #[serde(rename = "output_audio_buffer.stopped")]
    OutputAudioBufferStopped {
        event_id: String,
        response_id: String,
    },

    /// Emitted when the output audio buffer is cleared (user interrupt or
    /// explicit `output_audio_buffer.clear`).
    #[serde(rename = "output_audio_buffer.cleared")]
    OutputAudioBufferCleared {
        event_id: String,
        response_id: String,
    },

    // ---- Response lifecycle events ----
    /// Emitted when a new response is created (status `in_progress`).
    #[serde(rename = "response.created")]
    ResponseCreated {
        event_id: String,
        response: Box<RealtimeResponse>,
    },

    /// Emitted when a response is done streaming.
    #[serde(rename = "response.done")]
    ResponseDone {
        event_id: String,
        response: Box<RealtimeResponse>,
    },

    // ---- Response output item events ----
    /// Emitted when a new output item is created during response generation.
    #[serde(rename = "response.output_item.added")]
    ResponseOutputItemAdded {
        event_id: String,
        response_id: String,
        output_index: u32,
        item: RealtimeConversationItem,
    },

    /// Emitted when an output item is done streaming.
    #[serde(rename = "response.output_item.done")]
    ResponseOutputItemDone {
        event_id: String,
        response_id: String,
        output_index: u32,
        item: RealtimeConversationItem,
    },

    // ---- Response content part events ----
    /// Emitted when a new content part is added to an assistant message.
    #[serde(rename = "response.content_part.added")]
    ResponseContentPartAdded {
        event_id: String,
        response_id: String,
        item_id: String,
        output_index: u32,
        content_index: u32,
        part: ResponseContentPart,
    },

    /// Emitted when a content part is done streaming.
    #[serde(rename = "response.content_part.done")]
    ResponseContentPartDone {
        event_id: String,
        response_id: String,
        item_id: String,
        output_index: u32,
        content_index: u32,
        part: ResponseContentPart,
    },

    // ---- Response text events ----
    /// Emitted when the text of an output_text content part is updated.
    #[serde(rename = "response.output_text.delta")]
    ResponseOutputTextDelta {
        event_id: String,
        response_id: String,
        item_id: String,
        output_index: u32,
        content_index: u32,
        delta: String,
    },

    /// Emitted when an output_text content part is done streaming.
    #[serde(rename = "response.output_text.done")]
    ResponseOutputTextDone {
        event_id: String,
        response_id: String,
        item_id: String,
        output_index: u32,
        content_index: u32,
        text: String,
    },

    // ---- Response audio events ----
    /// Emitted when model-generated audio is updated.
    ///
    /// WARNING: `delta` contains a base64 audio chunk. Avoid logging this
    /// variant with `Debug` in production; prefer `event_type()`.
    #[serde(rename = "response.output_audio.delta")]
    ResponseOutputAudioDelta {
        event_id: String,
        response_id: String,
        item_id: String,
        output_index: u32,
        content_index: u32,
        delta: String,
    },

    /// Emitted when model-generated audio is done.
    #[serde(rename = "response.output_audio.done")]
    ResponseOutputAudioDone {
        event_id: String,
        response_id: String,
        item_id: String,
        output_index: u32,
        content_index: u32,
    },

    // ---- Response audio transcript events ----
    /// Emitted when the transcription of audio output is updated.
    #[serde(rename = "response.output_audio_transcript.delta")]
    ResponseOutputAudioTranscriptDelta {
        event_id: String,
        response_id: String,
        item_id: String,
        output_index: u32,
        content_index: u32,
        delta: String,
    },

    /// Emitted when the transcription of audio output is done.
    #[serde(rename = "response.output_audio_transcript.done")]
    ResponseOutputAudioTranscriptDone {
        event_id: String,
        response_id: String,
        item_id: String,
        output_index: u32,
        content_index: u32,
        transcript: String,
    },

    // ---- Response function call events ----
    /// Emitted when function call arguments are updated.
    #[serde(rename = "response.function_call_arguments.delta")]
    ResponseFunctionCallArgumentsDelta {
        event_id: String,
        response_id: String,
        item_id: String,
        output_index: u32,
        call_id: String,
        delta: String,
    },

    /// Emitted when function call arguments are done streaming.
    #[serde(rename = "response.function_call_arguments.done")]
    ResponseFunctionCallArgumentsDone {
        event_id: String,
        response_id: String,
        item_id: String,
        output_index: u32,
        call_id: String,
        name: String,
        arguments: String,
    },

    // ---- Response MCP call events ----
    /// Emitted when MCP tool call arguments are updated.
    #[serde(rename = "response.mcp_call_arguments.delta")]
    ResponseMcpCallArgumentsDelta {
        event_id: String,
        response_id: String,
        item_id: String,
        output_index: u32,
        delta: String,
        obfuscation: Option<String>,
    },

    /// Emitted when MCP tool call arguments are finalized.
    #[serde(rename = "response.mcp_call_arguments.done")]
    ResponseMcpCallArgumentsDone {
        event_id: String,
        response_id: String,
        item_id: String,
        output_index: u32,
        arguments: String,
    },

    /// Emitted when an MCP tool call starts.
    #[serde(rename = "response.mcp_call.in_progress")]
    ResponseMcpCallInProgress {
        event_id: String,
        output_index: u32,
        item_id: String,
    },

    /// Emitted when an MCP tool call completes successfully.
    #[serde(rename = "response.mcp_call.completed")]
    ResponseMcpCallCompleted {
        event_id: String,
        output_index: u32,
        item_id: String,
    },

    /// Emitted when an MCP tool call fails.
    #[serde(rename = "response.mcp_call.failed")]
    ResponseMcpCallFailed {
        event_id: String,
        output_index: u32,
        item_id: String,
    },

    // ---- MCP list tools events ----
    /// Emitted when listing MCP tools is in progress.
    #[serde(rename = "mcp_list_tools.in_progress")]
    McpListToolsInProgress { event_id: String, item_id: String },

    /// Emitted when listing MCP tools has completed.
    #[serde(rename = "mcp_list_tools.completed")]
    McpListToolsCompleted { event_id: String, item_id: String },

    /// Emitted when listing MCP tools has failed.
    #[serde(rename = "mcp_list_tools.failed")]
    McpListToolsFailed { event_id: String, item_id: String },

    // ---- Rate limits ----
    /// Emitted at the beginning of a response with updated rate limit info.
    #[serde(rename = "rate_limits.updated")]
    RateLimitsUpdated {
        event_id: String,
        rate_limits: Vec<RateLimitInfo>,
    },

    // ---- Error ----
    /// Emitted when an error occurs. Most errors are recoverable.
    #[serde(rename = "error")]
    Error {
        event_id: String,
        error: RealtimeError,
    },

    // ---- Unknown ----
    /// Unrecognized event type — preserves the full JSON payload for
    /// pass-through forwarding. This ensures forward compatibility when
    /// OpenAI adds new event types.
    #[serde(skip)]
    Unknown(serde_json::Value),
}

impl ServerEvent {
    /// Deserialize a JSON string into a `ServerEvent`, falling back to
    /// `Unknown` for unrecognized event types instead of returning an error.
    ///
    /// Use this instead of `serde_json::from_str` when you want forward
    /// compatibility with new event types (e.g. in a proxy gateway).
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        match serde_json::from_str::<ServerEvent>(json) {
            Ok(event) => Ok(event),
            Err(_) => {
                let value: serde_json::Value = serde_json::from_str(json)?;
                Ok(ServerEvent::Unknown(value))
            }
        }
    }

    /// Serialize to a JSON string. For `Unknown` variants, serializes the
    /// preserved `Value` directly; for known variants, uses the derived
    /// `Serialize` impl.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        match self {
            ServerEvent::Unknown(value) => serde_json::to_string(value),
            _ => serde_json::to_string(self),
        }
    }

    /// Returns the event type string (e.g. `"session.created"`).
    ///
    /// For known events, returns a `&'static str` from the event type constants.
    /// For unknown events, returns `"unknown"`.
    pub fn event_type(&self) -> &str {
        match self {
            ServerEvent::Unknown(_) => "unknown",
            _ => RealtimeServerEvent::from(self).as_str(),
        }
    }

    /// Returns true if this is a `response.function_call_arguments.done` event.
    pub fn is_function_call_done(&self) -> bool {
        matches!(self, ServerEvent::ResponseFunctionCallArgumentsDone { .. })
    }

    /// For `response.function_call_arguments.done`, returns `(call_id, item_id, arguments)`.
    pub fn get_function_call(&self) -> Option<(&str, &str, &str)> {
        match self {
            ServerEvent::ResponseFunctionCallArgumentsDone {
                call_id,
                item_id,
                arguments,
                ..
            } => Some((call_id, item_id, arguments)),
            _ => None,
        }
    }
}

impl From<&ServerEvent> for RealtimeServerEvent {
    fn from(event: &ServerEvent) -> Self {
        match event {
            ServerEvent::SessionCreated { .. } => Self::SessionCreated,
            ServerEvent::SessionUpdated { .. } => Self::SessionUpdated,
            ServerEvent::ConversationCreated { .. } => Self::ConversationCreated,
            ServerEvent::ConversationItemCreated { .. } => Self::ConversationItemCreated,
            ServerEvent::ConversationItemAdded { .. } => Self::ConversationItemAdded,
            ServerEvent::ConversationItemDone { .. } => Self::ConversationItemDone,
            ServerEvent::ConversationItemDeleted { .. } => Self::ConversationItemDeleted,
            ServerEvent::ConversationItemRetrieved { .. } => Self::ConversationItemRetrieved,
            ServerEvent::ConversationItemTruncated { .. } => Self::ConversationItemTruncated,
            ServerEvent::InputAudioTranscriptionCompleted { .. } => {
                Self::ConversationItemInputAudioTranscriptionCompleted
            }
            ServerEvent::InputAudioTranscriptionDelta { .. } => {
                Self::ConversationItemInputAudioTranscriptionDelta
            }
            ServerEvent::InputAudioTranscriptionFailed { .. } => {
                Self::ConversationItemInputAudioTranscriptionFailed
            }
            ServerEvent::InputAudioTranscriptionSegment { .. } => {
                Self::ConversationItemInputAudioTranscriptionSegment
            }
            ServerEvent::InputAudioBufferCleared { .. } => Self::InputAudioBufferCleared,
            ServerEvent::InputAudioBufferCommitted { .. } => Self::InputAudioBufferCommitted,
            ServerEvent::InputAudioBufferSpeechStarted { .. } => {
                Self::InputAudioBufferSpeechStarted
            }
            ServerEvent::InputAudioBufferSpeechStopped { .. } => {
                Self::InputAudioBufferSpeechStopped
            }
            ServerEvent::InputAudioBufferTimeoutTriggered { .. } => {
                Self::InputAudioBufferTimeoutTriggered
            }
            ServerEvent::InputAudioBufferDtmfEventReceived { .. } => {
                Self::InputAudioBufferDtmfEventReceived
            }
            ServerEvent::OutputAudioBufferStarted { .. } => Self::OutputAudioBufferStarted,
            ServerEvent::OutputAudioBufferStopped { .. } => Self::OutputAudioBufferStopped,
            ServerEvent::OutputAudioBufferCleared { .. } => Self::OutputAudioBufferCleared,
            ServerEvent::ResponseCreated { .. } => Self::ResponseCreated,
            ServerEvent::ResponseDone { .. } => Self::ResponseDone,
            ServerEvent::ResponseOutputItemAdded { .. } => Self::ResponseOutputItemAdded,
            ServerEvent::ResponseOutputItemDone { .. } => Self::ResponseOutputItemDone,
            ServerEvent::ResponseContentPartAdded { .. } => Self::ResponseContentPartAdded,
            ServerEvent::ResponseContentPartDone { .. } => Self::ResponseContentPartDone,
            ServerEvent::ResponseOutputTextDelta { .. } => Self::ResponseOutputTextDelta,
            ServerEvent::ResponseOutputTextDone { .. } => Self::ResponseOutputTextDone,
            ServerEvent::ResponseOutputAudioDelta { .. } => Self::ResponseOutputAudioDelta,
            ServerEvent::ResponseOutputAudioDone { .. } => Self::ResponseOutputAudioDone,
            ServerEvent::ResponseOutputAudioTranscriptDelta { .. } => {
                Self::ResponseOutputAudioTranscriptDelta
            }
            ServerEvent::ResponseOutputAudioTranscriptDone { .. } => {
                Self::ResponseOutputAudioTranscriptDone
            }
            ServerEvent::ResponseFunctionCallArgumentsDelta { .. } => {
                Self::ResponseFunctionCallArgumentsDelta
            }
            ServerEvent::ResponseFunctionCallArgumentsDone { .. } => {
                Self::ResponseFunctionCallArgumentsDone
            }
            ServerEvent::ResponseMcpCallArgumentsDelta { .. } => {
                Self::ResponseMcpCallArgumentsDelta
            }
            ServerEvent::ResponseMcpCallArgumentsDone { .. } => Self::ResponseMcpCallArgumentsDone,
            ServerEvent::ResponseMcpCallInProgress { .. } => Self::ResponseMcpCallInProgress,
            ServerEvent::ResponseMcpCallCompleted { .. } => Self::ResponseMcpCallCompleted,
            ServerEvent::ResponseMcpCallFailed { .. } => Self::ResponseMcpCallFailed,
            ServerEvent::McpListToolsInProgress { .. } => Self::McpListToolsInProgress,
            ServerEvent::McpListToolsCompleted { .. } => Self::McpListToolsCompleted,
            ServerEvent::McpListToolsFailed { .. } => Self::McpListToolsFailed,
            ServerEvent::RateLimitsUpdated { .. } => Self::RateLimitsUpdated,
            ServerEvent::Error { .. } => Self::Error,
            ServerEvent::Unknown(_) => Self::Error, // fallback; callers should check Unknown first
        }
    }
}

// ============================================================================
// Session Config Union
// ============================================================================

/// Union of realtime and transcription session configurations.
///
/// Discriminated by the `type` field: `"realtime"` or `"transcription"`.
/// Used by `session.update`, `session.created`, and `session.updated` events.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SessionConfig {
    #[serde(rename = "realtime")]
    Realtime(Box<RealtimeSessionConfig>),
    #[serde(rename = "transcription")]
    Transcription(Box<RealtimeTranscriptionSessionConfig>),
}

// ============================================================================
// Transcription Session Update Config (flat format)
// ============================================================================

/// Transcription session configuration in the flat format used by the
/// `transcription_session.update` client event.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionSessionUpdateConfig {
    pub input_audio_format: Option<String>,
    pub input_audio_transcription: Option<AudioTranscription>,
    pub turn_detection: Option<TurnDetection>,
    pub input_audio_noise_reduction: Option<NoiseReduction>,
    pub include: Option<Vec<RealtimeIncludeOption>>,
}

// ============================================================================
// Supporting Types
// ============================================================================

/// Conversation metadata returned in `conversation.created` events.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationInfo {
    pub id: Option<String>,
    pub object: Option<String>,
}

/// A content part within a response (used in `response.content_part.*` events).
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseContentPart {
    #[serde(rename = "type")]
    pub r#type: Option<String>,
    pub text: Option<String>,
    pub audio: Option<String>,
    pub transcript: Option<String>,
}

/// Log probability entry for input audio transcription.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogProb {
    pub token: String,
    pub logprob: f64,
    /// UTF-8 byte values of the token. Serializes as a JSON array of integers
    /// (e.g. `[104, 101, 108, 108, 111]`), matching the OpenAI spec.
    pub bytes: Vec<u8>,
}

/// Input token details for transcription usage.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionTokenInputDetails {
    pub text_tokens: Option<u32>,
    pub audio_tokens: Option<u32>,
}

/// Usage statistics for input audio transcription.
///
/// Discriminated by the `type` field: `"tokens"` or `"duration"`.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TranscriptionUsage {
    /// Token-based usage (e.g. for `gpt-4o-transcribe`).
    #[serde(rename = "tokens")]
    Tokens {
        total_tokens: Option<u32>,
        input_tokens: Option<u32>,
        input_token_details: Option<TranscriptionTokenInputDetails>,
        output_tokens: Option<u32>,
    },
    /// Duration-based usage (e.g. for `whisper-1`).
    #[serde(rename = "duration")]
    Duration { seconds: f64 },
}

/// Error details for a failed input audio transcription.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionError {
    #[serde(rename = "type")]
    pub r#type: Option<String>,
    pub code: Option<String>,
    pub message: Option<String>,
    pub param: Option<String>,
}

/// Rate limit information returned in `rate_limits.updated` events.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitInfo {
    pub name: Option<String>,
    pub limit: Option<u32>,
    pub remaining: Option<u32>,
    pub reset_seconds: Option<f64>,
}

/// Error details returned in the `error` server event.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeError {
    #[serde(rename = "type")]
    pub r#type: String,
    pub code: Option<String>,
    pub message: String,
    pub param: Option<String>,
    pub event_id: Option<String>,
}
