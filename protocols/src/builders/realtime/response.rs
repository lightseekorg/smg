//! Builder for RealtimeResponse
//!
//! Provides an ergonomic fluent API for constructing realtime response instances.

use std::collections::HashMap;

use crate::{
    realtime_conversation::RealtimeConversationItem,
    realtime_response::*,
    realtime_session::{MaxOutputTokens, OutputModality},
};

/// Builder for RealtimeResponse
///
/// Provides a fluent interface for constructing realtime responses with sensible defaults.
#[must_use = "Builder does nothing until .build() is called"]
#[derive(Clone, Debug)]
pub struct RealtimeResponseBuilder {
    id: String,
    object: Option<RealtimeResponseObject>,
    status: Option<RealtimeResponseStatus>,
    status_details: Option<RealtimeResponseStatusDetails>,
    output: Vec<RealtimeConversationItem>,
    metadata: HashMap<String, String>,
    audio: Option<ResponseAudioConfig>,
    usage: Option<RealtimeUsage>,
    conversation_id: Option<String>,
    output_modalities: Option<Vec<OutputModality>>,
    max_output_tokens: Option<MaxOutputTokens>,
}

impl RealtimeResponseBuilder {
    /// Create a new builder with the response ID.
    ///
    /// # Arguments
    /// - `id`: Response ID (e.g., "resp_abc123")
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            object: Some(RealtimeResponseObject::RealtimeResponse),
            status: Some(RealtimeResponseStatus::InProgress),
            status_details: None,
            output: Vec::new(),
            metadata: HashMap::new(),
            audio: None,
            usage: None,
            conversation_id: None,
            output_modalities: None,
            max_output_tokens: None,
        }
    }

    /// Set the response status (default: InProgress).
    pub fn status(mut self, status: RealtimeResponseStatus) -> Self {
        self.status = Some(status);
        self
    }

    /// Set status details (e.g. cancellation reason, failure error).
    pub fn status_details(mut self, details: RealtimeResponseStatusDetails) -> Self {
        self.status_details = Some(details);
        self
    }

    /// Set the full output items list.
    pub fn output(mut self, output: Vec<RealtimeConversationItem>) -> Self {
        self.output = output;
        self
    }

    /// Add a single output item.
    pub fn add_output(mut self, item: RealtimeConversationItem) -> Self {
        self.output.push(item);
        self
    }

    /// Set metadata.
    pub fn metadata(mut self, metadata: HashMap<String, String>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Add a single metadata entry.
    pub fn add_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Set audio configuration.
    pub fn audio(mut self, audio: ResponseAudioConfig) -> Self {
        self.audio = Some(audio);
        self
    }

    /// Set usage statistics.
    pub fn usage(mut self, usage: RealtimeUsage) -> Self {
        self.usage = Some(usage);
        self
    }

    /// Set usage if provided (handles Option).
    pub fn maybe_usage(mut self, usage: Option<RealtimeUsage>) -> Self {
        if let Some(u) = usage {
            self.usage = Some(u);
        }
        self
    }

    /// Set the conversation ID.
    pub fn conversation_id(mut self, id: impl Into<String>) -> Self {
        self.conversation_id = Some(id.into());
        self
    }

    /// Set output modalities.
    pub fn output_modalities(mut self, modalities: Vec<OutputModality>) -> Self {
        self.output_modalities = Some(modalities);
        self
    }

    /// Set max output tokens.
    pub fn max_output_tokens(mut self, max: MaxOutputTokens) -> Self {
        self.max_output_tokens = Some(max);
        self
    }

    /// Build the RealtimeResponse.
    pub fn build(self) -> RealtimeResponse {
        let metadata = if self.metadata.is_empty() {
            None
        } else {
            Some(self.metadata)
        };

        let output = if self.output.is_empty() {
            None
        } else {
            Some(self.output)
        };

        RealtimeResponse {
            id: Some(self.id),
            object: self.object,
            status: self.status,
            status_details: self.status_details,
            output,
            metadata,
            audio: self.audio,
            usage: self.usage,
            conversation_id: self.conversation_id,
            output_modalities: self.output_modalities,
            max_output_tokens: self.max_output_tokens,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_minimal() {
        let response = RealtimeResponseBuilder::new("resp_123").build();

        assert_eq!(response.id.as_ref().unwrap(), "resp_123");
        assert_eq!(
            response.object.as_ref().unwrap(),
            &RealtimeResponseObject::RealtimeResponse
        );
        assert_eq!(
            response.status.as_ref().unwrap(),
            &RealtimeResponseStatus::InProgress
        );
        assert!(response.output.is_none());
        assert!(response.metadata.is_none());
        assert!(response.usage.is_none());
    }

    #[test]
    fn test_build_complete() {
        let usage = RealtimeUsage {
            total_tokens: Some(100),
            input_tokens: Some(40),
            output_tokens: Some(60),
            input_token_details: None,
            output_token_details: None,
        };

        let response = RealtimeResponseBuilder::new("resp_full")
            .status(RealtimeResponseStatus::Completed)
            .conversation_id("conv_123")
            .output_modalities(vec![OutputModality::Audio, OutputModality::Text])
            .max_output_tokens(MaxOutputTokens::Integer(4096))
            .add_metadata("session", "test")
            .maybe_usage(Some(usage))
            .build();

        assert_eq!(response.id.as_ref().unwrap(), "resp_full");
        assert_eq!(
            response.status.as_ref().unwrap(),
            &RealtimeResponseStatus::Completed
        );
        assert_eq!(response.conversation_id.as_ref().unwrap(), "conv_123");
        assert_eq!(response.output_modalities.as_ref().unwrap().len(), 2);
        assert!(response.metadata.is_some());
        assert_eq!(response.usage.as_ref().unwrap().total_tokens, Some(100));
    }
}
