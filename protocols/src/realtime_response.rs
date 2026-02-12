// OpenAI Realtime Conversation API types
// https://platform.openai.com/docs/api-reference/realtime
//
// Session configuration and audio types live in `realtime_session`.
// Event type constants live in `event_types`.
// This module covers the realtime response
// object, usage, errors, rate limits.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{
    realtime_conversation::RealtimeConversationItem,
    realtime_session::{
        MaxOutputTokens, OutputModality, Prompt, RealtimeAudioFormat, RealtimeTool,
        RealtimeToolChoice, Voice,
    },
};

// ============================================================================
// Realtime Response
// ============================================================================

/// A response object in the Realtime API.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeResponse {
    pub id: Option<String>,
    pub object: Option<RealtimeResponseObject>,
    pub status: Option<RealtimeResponseStatus>,
    pub status_details: Option<RealtimeResponseStatusDetails>,
    pub output: Option<Vec<RealtimeConversationItem>>,
    pub metadata: Option<HashMap<String, String>>,
    pub audio: Option<ResponseAudioConfig>,
    pub usage: Option<RealtimeUsage>,
    pub conversation_id: Option<String>,
    pub output_modalities: Option<Vec<OutputModality>>,
    pub max_output_tokens: Option<MaxOutputTokens>,
}

impl RealtimeResponse {
    /// Create a builder for constructing a RealtimeResponse.
    pub fn builder(id: impl Into<String>) -> crate::builders::realtime::RealtimeResponseBuilder {
        crate::builders::realtime::RealtimeResponseBuilder::new(id)
    }
}

// ============================================================================
// Response Create Params
// ============================================================================

/// Parameters for creating a realtime response.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseCreateParams {
    pub output_modalities: Option<Vec<OutputModality>>,
    pub instructions: Option<String>,
    pub audio: Option<ResponseAudioConfig>,
    pub tools: Option<Vec<RealtimeTool>>,
    pub tool_choice: Option<RealtimeToolChoice>,
    pub max_output_tokens: Option<MaxOutputTokens>,
    pub conversation: Option<String>,
    pub metadata: Option<HashMap<String, String>>,
    pub prompt: Option<Prompt>,
    pub input: Option<Vec<RealtimeConversationItem>>,
}

// ============================================================================
// Response Status
// ============================================================================

/// Object type for realtime responses. Always `"realtime.response"`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RealtimeResponseObject {
    #[serde(rename = "realtime.response")]
    RealtimeResponse,
}

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

/// The type within status details.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StatusDetailsType {
    Completed,
    Cancelled,
    Failed,
    Incomplete,
}

/// Reason the response did not complete.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StatusDetailsReason {
    TurnDetected,
    ClientCancelled,
    MaxOutputTokens,
    ContentFilter,
}

/// Error that caused the response to fail.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseStatusError {
    #[serde(rename = "type")]
    pub r#type: Option<String>,
    pub code: Option<String>,
}

/// Additional details about the response status.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeResponseStatusDetails {
    #[serde(rename = "type")]
    pub r#type: Option<StatusDetailsType>,
    pub reason: Option<StatusDetailsReason>,
    pub error: Option<ResponseStatusError>,
}

// ============================================================================
// Response Audio Configuration
// ============================================================================

/// Audio output configuration for a response.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseAudioOutputConfig {
    pub format: Option<RealtimeAudioFormat>,
    pub voice: Option<Voice>,
}

/// Audio configuration for a response (output only).
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseAudioConfig {
    pub output: Option<ResponseAudioOutputConfig>,
}

// ============================================================================
// Usage
// ============================================================================

/// Breakdown of cached token usage by modality.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedTokensDetails {
    pub text_tokens: Option<u32>,
    pub image_tokens: Option<u32>,
    pub audio_tokens: Option<u32>,
}

/// Input token usage details.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeInputTokenDetails {
    pub cached_tokens: Option<u32>,
    pub text_tokens: Option<u32>,
    pub image_tokens: Option<u32>,
    pub audio_tokens: Option<u32>,
    pub cached_tokens_details: Option<CachedTokensDetails>,
}

/// Output token usage details.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeOutputTokenDetails {
    pub text_tokens: Option<u32>,
    pub audio_tokens: Option<u32>,
}

/// Token usage for a realtime response.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeUsage {
    pub total_tokens: Option<u32>,
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
    pub input_token_details: Option<RealtimeInputTokenDetails>,
    pub output_token_details: Option<RealtimeOutputTokenDetails>,
}
