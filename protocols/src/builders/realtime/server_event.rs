//! Builder for ServerEvent
//!
//! Provides an ergonomic fluent API for constructing server-to-client realtime events.
//!
//! Many response-streaming events share fields (`event_id`, `response_id`, `item_id`,
//! `output_index`, `content_index`). A hierarchical builder captures this shared context
//! progressively:
//!
//! ```ignore
//! // Simple events:
//! ServerEvent::builder("evt_1").session_created(config)
//! ServerEvent::builder("evt_1").error(error)
//!
//! // Response streaming events (progressive context):
//! ServerEvent::builder("evt_1")
//!     .for_response("resp_1")
//!     .for_item("item_1", 0)
//!     .for_content(0)
//!     .output_text_delta("Hello")
//! ```

use crate::{
    realtime_conversation::RealtimeConversationItem, realtime_events::*,
    realtime_response::RealtimeResponse,
};

// ============================================================================
// Level 1: ServerEventBuilder (event_id)
// ============================================================================

/// Top-level builder for `ServerEvent`.
///
/// Holds `event_id` and provides terminal methods for events that only need it.
/// Call `.for_response()` to descend into response-scoped events.
#[must_use = "Builder does nothing until a terminal method is called"]
#[derive(Clone, Debug)]
pub struct ServerEventBuilder {
    event_id: String,
}

impl ServerEventBuilder {
    /// Create a new builder with the given event ID.
    pub fn new(event_id: impl Into<String>) -> Self {
        Self {
            event_id: event_id.into(),
        }
    }

    // ---- Transition ----

    /// Descend into response-scoped events.
    pub fn for_response(self, response_id: impl Into<String>) -> ResponseEventBuilder {
        ResponseEventBuilder {
            event_id: self.event_id,
            response_id: response_id.into(),
        }
    }

    // ---- Session events ----

    /// Build a `session.created` event.
    pub fn session_created(self, session: SessionConfig) -> ServerEvent {
        ServerEvent::SessionCreated {
            event_id: self.event_id,
            session: Box::new(session),
        }
    }

    /// Build a `session.updated` event.
    pub fn session_updated(self, session: SessionConfig) -> ServerEvent {
        ServerEvent::SessionUpdated {
            event_id: self.event_id,
            session: Box::new(session),
        }
    }

    // ---- Conversation events ----

    /// Build a `conversation.created` event.
    pub fn conversation_created(self, conversation: ConversationInfo) -> ServerEvent {
        ServerEvent::ConversationCreated {
            event_id: self.event_id,
            conversation,
        }
    }

    /// Build a `conversation.item.created` event.
    pub fn conversation_item_created(
        self,
        previous_item_id: Option<String>,
        item: RealtimeConversationItem,
    ) -> ServerEvent {
        ServerEvent::ConversationItemCreated {
            event_id: self.event_id,
            previous_item_id,
            item,
        }
    }

    /// Build a `conversation.item.added` event.
    pub fn conversation_item_added(
        self,
        previous_item_id: Option<String>,
        item: RealtimeConversationItem,
    ) -> ServerEvent {
        ServerEvent::ConversationItemAdded {
            event_id: self.event_id,
            previous_item_id,
            item,
        }
    }

    /// Build a `conversation.item.done` event.
    pub fn conversation_item_done(
        self,
        previous_item_id: Option<String>,
        item: RealtimeConversationItem,
    ) -> ServerEvent {
        ServerEvent::ConversationItemDone {
            event_id: self.event_id,
            previous_item_id,
            item,
        }
    }

    /// Build a `conversation.item.deleted` event.
    pub fn conversation_item_deleted(self, item_id: impl Into<String>) -> ServerEvent {
        ServerEvent::ConversationItemDeleted {
            event_id: self.event_id,
            item_id: item_id.into(),
        }
    }

    /// Build a `conversation.item.retrieved` event.
    pub fn conversation_item_retrieved(self, item: RealtimeConversationItem) -> ServerEvent {
        ServerEvent::ConversationItemRetrieved {
            event_id: self.event_id,
            item,
        }
    }

    /// Build a `conversation.item.truncated` event.
    pub fn conversation_item_truncated(
        self,
        item_id: impl Into<String>,
        content_index: u32,
        audio_end_ms: u32,
    ) -> ServerEvent {
        ServerEvent::ConversationItemTruncated {
            event_id: self.event_id,
            item_id: item_id.into(),
            content_index,
            audio_end_ms,
        }
    }

    // ---- Input audio transcription events ----

    /// Build a `conversation.item.input_audio_transcription.completed` event.
    pub fn input_audio_transcription_completed(
        self,
        item_id: impl Into<String>,
        content_index: u32,
        transcript: impl Into<String>,
        usage: TranscriptionUsage,
    ) -> ServerEvent {
        ServerEvent::InputAudioTranscriptionCompleted {
            event_id: self.event_id,
            item_id: item_id.into(),
            content_index,
            transcript: transcript.into(),
            logprobs: None,
            usage,
        }
    }

    /// Build a `conversation.item.input_audio_transcription.completed` event with logprobs.
    pub fn input_audio_transcription_completed_with_logprobs(
        self,
        item_id: impl Into<String>,
        content_index: u32,
        transcript: impl Into<String>,
        usage: TranscriptionUsage,
        logprobs: Vec<LogProb>,
    ) -> ServerEvent {
        ServerEvent::InputAudioTranscriptionCompleted {
            event_id: self.event_id,
            item_id: item_id.into(),
            content_index,
            transcript: transcript.into(),
            logprobs: Some(logprobs),
            usage,
        }
    }

    /// Build a `conversation.item.input_audio_transcription.delta` event.
    pub fn input_audio_transcription_delta(
        self,
        item_id: impl Into<String>,
        content_index: Option<u32>,
        delta: Option<String>,
        logprobs: Option<Vec<LogProb>>,
    ) -> ServerEvent {
        ServerEvent::InputAudioTranscriptionDelta {
            event_id: self.event_id,
            item_id: item_id.into(),
            content_index,
            delta,
            logprobs,
        }
    }

    /// Build a `conversation.item.input_audio_transcription.failed` event.
    pub fn input_audio_transcription_failed(
        self,
        item_id: impl Into<String>,
        content_index: u32,
        error: TranscriptionError,
    ) -> ServerEvent {
        ServerEvent::InputAudioTranscriptionFailed {
            event_id: self.event_id,
            item_id: item_id.into(),
            content_index,
            error,
        }
    }

    /// Build a `conversation.item.input_audio_transcription.segment` event.
    #[allow(clippy::too_many_arguments)]
    pub fn input_audio_transcription_segment(
        self,
        item_id: impl Into<String>,
        content_index: u32,
        text: impl Into<String>,
        id: impl Into<String>,
        speaker: impl Into<String>,
        start: f32,
        end: f32,
    ) -> ServerEvent {
        ServerEvent::InputAudioTranscriptionSegment {
            event_id: self.event_id,
            item_id: item_id.into(),
            content_index,
            text: text.into(),
            id: id.into(),
            speaker: speaker.into(),
            start,
            end,
        }
    }

    // ---- Input audio buffer events ----

    /// Build an `input_audio_buffer.cleared` event.
    pub fn input_audio_buffer_cleared(self) -> ServerEvent {
        ServerEvent::InputAudioBufferCleared {
            event_id: self.event_id,
        }
    }

    /// Build an `input_audio_buffer.committed` event.
    pub fn input_audio_buffer_committed(
        self,
        item_id: impl Into<String>,
        previous_item_id: Option<String>,
    ) -> ServerEvent {
        ServerEvent::InputAudioBufferCommitted {
            event_id: self.event_id,
            previous_item_id,
            item_id: item_id.into(),
        }
    }

    /// Build an `input_audio_buffer.speech_started` event.
    pub fn input_audio_buffer_speech_started(
        self,
        audio_start_ms: u32,
        item_id: impl Into<String>,
    ) -> ServerEvent {
        ServerEvent::InputAudioBufferSpeechStarted {
            event_id: self.event_id,
            audio_start_ms,
            item_id: item_id.into(),
        }
    }

    /// Build an `input_audio_buffer.speech_stopped` event.
    pub fn input_audio_buffer_speech_stopped(
        self,
        audio_end_ms: u32,
        item_id: impl Into<String>,
    ) -> ServerEvent {
        ServerEvent::InputAudioBufferSpeechStopped {
            event_id: self.event_id,
            audio_end_ms,
            item_id: item_id.into(),
        }
    }

    /// Build an `input_audio_buffer.timeout_triggered` event.
    pub fn input_audio_buffer_timeout_triggered(
        self,
        audio_start_ms: u32,
        audio_end_ms: u32,
        item_id: impl Into<String>,
    ) -> ServerEvent {
        ServerEvent::InputAudioBufferTimeoutTriggered {
            event_id: self.event_id,
            audio_start_ms,
            audio_end_ms,
            item_id: item_id.into(),
        }
    }

    /// Build an `input_audio_buffer.dtmf_event_received` event.
    ///
    /// This is a static method because DTMF events have no `event_id`.
    pub fn dtmf_event_received(event: impl Into<String>, received_at: i64) -> ServerEvent {
        ServerEvent::InputAudioBufferDtmfEventReceived {
            event: event.into(),
            received_at,
        }
    }

    // ---- MCP list tools events ----

    /// Build an `mcp_list_tools.in_progress` event.
    pub fn mcp_list_tools_in_progress(self, item_id: impl Into<String>) -> ServerEvent {
        ServerEvent::McpListToolsInProgress {
            event_id: self.event_id,
            item_id: item_id.into(),
        }
    }

    /// Build an `mcp_list_tools.completed` event.
    pub fn mcp_list_tools_completed(self, item_id: impl Into<String>) -> ServerEvent {
        ServerEvent::McpListToolsCompleted {
            event_id: self.event_id,
            item_id: item_id.into(),
        }
    }

    /// Build an `mcp_list_tools.failed` event.
    pub fn mcp_list_tools_failed(self, item_id: impl Into<String>) -> ServerEvent {
        ServerEvent::McpListToolsFailed {
            event_id: self.event_id,
            item_id: item_id.into(),
        }
    }

    // ---- MCP call lifecycle events ----
    // These only need event_id + item_id + output_index (no response_id).

    /// Build a `response.mcp_call.in_progress` event.
    pub fn mcp_call_in_progress(
        self,
        item_id: impl Into<String>,
        output_index: u32,
    ) -> ServerEvent {
        ServerEvent::ResponseMcpCallInProgress {
            event_id: self.event_id,
            output_index,
            item_id: item_id.into(),
        }
    }

    /// Build a `response.mcp_call.completed` event.
    pub fn mcp_call_completed(self, item_id: impl Into<String>, output_index: u32) -> ServerEvent {
        ServerEvent::ResponseMcpCallCompleted {
            event_id: self.event_id,
            output_index,
            item_id: item_id.into(),
        }
    }

    /// Build a `response.mcp_call.failed` event.
    pub fn mcp_call_failed(self, item_id: impl Into<String>, output_index: u32) -> ServerEvent {
        ServerEvent::ResponseMcpCallFailed {
            event_id: self.event_id,
            output_index,
            item_id: item_id.into(),
        }
    }

    // ---- Rate limits ----

    /// Build a `rate_limits.updated` event.
    pub fn rate_limits_updated(self, rate_limits: Vec<RateLimitInfo>) -> ServerEvent {
        ServerEvent::RateLimitsUpdated {
            event_id: self.event_id,
            rate_limits,
        }
    }

    // ---- Response lifecycle events ----

    /// Build a `response.created` event.
    pub fn response_created(self, response: RealtimeResponse) -> ServerEvent {
        ServerEvent::ResponseCreated {
            event_id: self.event_id,
            response: Box::new(response),
        }
    }

    /// Build a `response.done` event.
    pub fn response_done(self, response: RealtimeResponse) -> ServerEvent {
        ServerEvent::ResponseDone {
            event_id: self.event_id,
            response: Box::new(response),
        }
    }

    // ---- Error ----

    /// Build an `error` event.
    pub fn error(self, error: RealtimeError) -> ServerEvent {
        ServerEvent::Error {
            event_id: self.event_id,
            error,
        }
    }
}

// ============================================================================
// Level 2: ResponseEventBuilder (event_id + response_id)
// ============================================================================

/// Builder for response-scoped server events.
///
/// Holds `event_id` and `response_id`. Provides terminal methods for output
/// audio buffer events. Call `.for_item()` to descend into item-scoped events.
#[must_use = "Builder does nothing until a terminal method is called"]
#[derive(Clone, Debug)]
pub struct ResponseEventBuilder {
    event_id: String,
    response_id: String,
}

impl ResponseEventBuilder {
    // ---- Transition ----

    /// Descend into item-scoped events.
    pub fn for_item(self, item_id: impl Into<String>, output_index: u32) -> ItemEventBuilder {
        ItemEventBuilder {
            event_id: self.event_id,
            response_id: self.response_id,
            item_id: item_id.into(),
            output_index,
        }
    }

    // ---- Output audio buffer events (WebRTC/SIP only) ----

    /// Build an `output_audio_buffer.started` event.
    pub fn output_audio_buffer_started(self) -> ServerEvent {
        ServerEvent::OutputAudioBufferStarted {
            event_id: self.event_id,
            response_id: self.response_id,
        }
    }

    /// Build an `output_audio_buffer.stopped` event.
    pub fn output_audio_buffer_stopped(self) -> ServerEvent {
        ServerEvent::OutputAudioBufferStopped {
            event_id: self.event_id,
            response_id: self.response_id,
        }
    }

    /// Build an `output_audio_buffer.cleared` event.
    pub fn output_audio_buffer_cleared(self) -> ServerEvent {
        ServerEvent::OutputAudioBufferCleared {
            event_id: self.event_id,
            response_id: self.response_id,
        }
    }
}

// ============================================================================
// Level 3: ItemEventBuilder (event_id + response_id + item_id + output_index)
// ============================================================================

/// Builder for item-scoped server events.
///
/// Holds `event_id`, `response_id`, `item_id`, and `output_index`. Provides
/// terminal methods for output item and function call events. Call `.for_content()`
/// to descend into content-scoped events.
#[must_use = "Builder does nothing until a terminal method is called"]
#[derive(Clone, Debug)]
pub struct ItemEventBuilder {
    event_id: String,
    response_id: String,
    item_id: String,
    output_index: u32,
}

impl ItemEventBuilder {
    // ---- Transition ----

    /// Descend into content-scoped events.
    pub fn for_content(self, content_index: u32) -> ContentEventBuilder {
        ContentEventBuilder {
            event_id: self.event_id,
            response_id: self.response_id,
            item_id: self.item_id,
            output_index: self.output_index,
            content_index,
        }
    }

    // ---- Response output item events ----

    /// Build a `response.output_item.added` event.
    pub fn output_item_added(self, item: RealtimeConversationItem) -> ServerEvent {
        ServerEvent::ResponseOutputItemAdded {
            event_id: self.event_id,
            response_id: self.response_id,
            output_index: self.output_index,
            item,
        }
    }

    /// Build a `response.output_item.done` event.
    pub fn output_item_done(self, item: RealtimeConversationItem) -> ServerEvent {
        ServerEvent::ResponseOutputItemDone {
            event_id: self.event_id,
            response_id: self.response_id,
            output_index: self.output_index,
            item,
        }
    }

    // ---- Response function call events ----

    /// Build a `response.function_call_arguments.delta` event.
    pub fn function_call_arguments_delta(
        self,
        call_id: impl Into<String>,
        delta: impl Into<String>,
    ) -> ServerEvent {
        ServerEvent::ResponseFunctionCallArgumentsDelta {
            event_id: self.event_id,
            response_id: self.response_id,
            item_id: self.item_id,
            output_index: self.output_index,
            call_id: call_id.into(),
            delta: delta.into(),
        }
    }

    /// Build a `response.function_call_arguments.done` event.
    pub fn function_call_arguments_done(
        self,
        call_id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> ServerEvent {
        ServerEvent::ResponseFunctionCallArgumentsDone {
            event_id: self.event_id,
            response_id: self.response_id,
            item_id: self.item_id,
            output_index: self.output_index,
            call_id: call_id.into(),
            name: name.into(),
            arguments: arguments.into(),
        }
    }

    // ---- Response MCP call events ----

    /// Build a `response.mcp_call_arguments.delta` event.
    pub fn mcp_call_arguments_delta(
        self,
        delta: impl Into<String>,
        obfuscation: Option<String>,
    ) -> ServerEvent {
        ServerEvent::ResponseMcpCallArgumentsDelta {
            event_id: self.event_id,
            response_id: self.response_id,
            item_id: self.item_id,
            output_index: self.output_index,
            delta: delta.into(),
            obfuscation,
        }
    }

    /// Build a `response.mcp_call_arguments.done` event.
    pub fn mcp_call_arguments_done(self, arguments: impl Into<String>) -> ServerEvent {
        ServerEvent::ResponseMcpCallArgumentsDone {
            event_id: self.event_id,
            response_id: self.response_id,
            item_id: self.item_id,
            output_index: self.output_index,
            arguments: arguments.into(),
        }
    }
}

// ============================================================================
// Level 4: ContentEventBuilder
//   (event_id + response_id + item_id + output_index + content_index)
// ============================================================================

/// Builder for content-scoped server events.
///
/// Holds all five shared fields. Provides terminal methods for content part,
/// text, audio, and audio transcript streaming events.
#[must_use = "Builder does nothing until a terminal method is called"]
#[derive(Clone, Debug)]
pub struct ContentEventBuilder {
    event_id: String,
    response_id: String,
    item_id: String,
    output_index: u32,
    content_index: u32,
}

impl ContentEventBuilder {
    // ---- Response content part events ----

    /// Build a `response.content_part.added` event.
    pub fn content_part_added(self, part: ResponseContentPart) -> ServerEvent {
        ServerEvent::ResponseContentPartAdded {
            event_id: self.event_id,
            response_id: self.response_id,
            item_id: self.item_id,
            output_index: self.output_index,
            content_index: self.content_index,
            part,
        }
    }

    /// Build a `response.content_part.done` event.
    pub fn content_part_done(self, part: ResponseContentPart) -> ServerEvent {
        ServerEvent::ResponseContentPartDone {
            event_id: self.event_id,
            response_id: self.response_id,
            item_id: self.item_id,
            output_index: self.output_index,
            content_index: self.content_index,
            part,
        }
    }

    // ---- Response text events ----

    /// Build a `response.output_text.delta` event.
    pub fn output_text_delta(self, delta: impl Into<String>) -> ServerEvent {
        ServerEvent::ResponseOutputTextDelta {
            event_id: self.event_id,
            response_id: self.response_id,
            item_id: self.item_id,
            output_index: self.output_index,
            content_index: self.content_index,
            delta: delta.into(),
        }
    }

    /// Build a `response.output_text.done` event.
    pub fn output_text_done(self, text: impl Into<String>) -> ServerEvent {
        ServerEvent::ResponseOutputTextDone {
            event_id: self.event_id,
            response_id: self.response_id,
            item_id: self.item_id,
            output_index: self.output_index,
            content_index: self.content_index,
            text: text.into(),
        }
    }

    // ---- Response audio events ----

    /// Build a `response.output_audio.delta` event.
    pub fn output_audio_delta(self, delta: impl Into<String>) -> ServerEvent {
        ServerEvent::ResponseOutputAudioDelta {
            event_id: self.event_id,
            response_id: self.response_id,
            item_id: self.item_id,
            output_index: self.output_index,
            content_index: self.content_index,
            delta: delta.into(),
        }
    }

    /// Build a `response.output_audio.done` event.
    pub fn output_audio_done(self) -> ServerEvent {
        ServerEvent::ResponseOutputAudioDone {
            event_id: self.event_id,
            response_id: self.response_id,
            item_id: self.item_id,
            output_index: self.output_index,
            content_index: self.content_index,
        }
    }

    // ---- Response audio transcript events ----

    /// Build a `response.output_audio_transcript.delta` event.
    pub fn output_audio_transcript_delta(self, delta: impl Into<String>) -> ServerEvent {
        ServerEvent::ResponseOutputAudioTranscriptDelta {
            event_id: self.event_id,
            response_id: self.response_id,
            item_id: self.item_id,
            output_index: self.output_index,
            content_index: self.content_index,
            delta: delta.into(),
        }
    }

    /// Build a `response.output_audio_transcript.done` event.
    pub fn output_audio_transcript_done(self, transcript: impl Into<String>) -> ServerEvent {
        ServerEvent::ResponseOutputAudioTranscriptDone {
            event_id: self.event_id,
            response_id: self.response_id,
            item_id: self.item_id,
            output_index: self.output_index,
            content_index: self.content_index,
            transcript: transcript.into(),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::realtime_session::{OutputModality, RealtimeSessionConfig, RealtimeSessionType};

    // Level 1: ServerEventBuilder
    #[test]
    fn test_level1_session_created() {
        let config = SessionConfig::Realtime(Box::new(RealtimeSessionConfig {
            r#type: RealtimeSessionType::Realtime,
            output_modalities: vec![OutputModality::Audio],
            model: None,
            instructions: None,
            audio: None,
            include: None,
            tracing: None,
            tools: None,
            tool_choice: None,
            max_output_tokens: None,
            truncation: None,
            prompt: None,
        }));

        let event = ServerEventBuilder::new("evt_1").session_created(config);
        assert_eq!(event.event_type(), "session.created");
    }

    #[test]
    fn test_dtmf_static_no_event_id() {
        let event = ServerEventBuilder::dtmf_event_received("5", 1700000000);
        assert_eq!(event.event_type(), "input_audio_buffer.dtmf_event_received");
    }

    #[test]
    fn test_response_created() {
        let response = RealtimeResponse::builder("resp_1").build();
        let event = ServerEventBuilder::new("evt_2").response_created(response);
        assert_eq!(event.event_type(), "response.created");
    }

    // Level 2: ResponseEventBuilder
    #[test]
    fn test_level2_output_audio_buffer_started() {
        let event = ServerEventBuilder::new("evt_5")
            .for_response("resp_1")
            .output_audio_buffer_started();
        assert_eq!(event.event_type(), "output_audio_buffer.started");
    }

    // Level 3: ItemEventBuilder
    #[test]
    fn test_level3_function_call_done() {
        let event = ServerEventBuilder::new("evt_3")
            .for_response("resp_1")
            .for_item("item_1", 0)
            .function_call_arguments_done("call_1", "get_weather", r#"{"location":"NYC"}"#);
        assert_eq!(event.event_type(), "response.function_call_arguments.done");
    }

    // Level 4: ContentEventBuilder (with field verification)
    #[test]
    fn test_level4_output_text_delta() {
        let event = ServerEventBuilder::new("evt_4")
            .for_response("resp_1")
            .for_item("item_1", 0)
            .for_content(0)
            .output_text_delta("Hello");

        assert_eq!(event.event_type(), "response.output_text.delta");
        if let ServerEvent::ResponseOutputTextDelta {
            event_id,
            response_id,
            item_id,
            output_index,
            content_index,
            delta,
        } = &event
        {
            assert_eq!(event_id, "evt_4");
            assert_eq!(response_id, "resp_1");
            assert_eq!(item_id, "item_1");
            assert_eq!(*output_index, 0);
            assert_eq!(*content_index, 0);
            assert_eq!(delta, "Hello");
        } else {
            panic!("Expected ResponseOutputTextDelta");
        }
    }
}
