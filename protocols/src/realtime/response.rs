//! Response types for the Realtime API.
//!
//! This module contains types for response configuration and the response
//! object returned by the API during streaming.

use serde::{Deserialize, Serialize};

use super::conversation::ConversationItem;
use super::session::{MaxResponseOutputTokens, Modality, Voice};

// ============================================================================
// Response Status
// ============================================================================

/// Status of a response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseStatus {
    /// Response is being generated
    InProgress,
    /// Response completed successfully
    Completed,
    /// Response was cancelled
    Cancelled,
    /// Response generation was interrupted
    Incomplete,
    /// Response failed due to an error
    Failed,
}

// ============================================================================
// Response Status Details
// ============================================================================

/// Details about why a response ended.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseStatusDetails {
    /// Response completed normally
    Completed,
    /// Response was cancelled by the client
    Cancelled,
    /// Response was incomplete
    Incomplete {
        /// Reason for incompleteness
        #[serde(skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    /// Response failed with an error
    Failed {
        /// Error details
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<ResponseError>,
    },
}

/// Error details for a failed response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseError {
    /// Error type
    #[serde(rename = "type")]
    pub error_type: String,
    /// Error code
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    /// Error message
    pub message: String,
}

// ============================================================================
// Usage Statistics
// ============================================================================

/// Token usage statistics for a response.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResponseUsage {
    /// Total tokens used
    #[serde(default)]
    pub total_tokens: u32,
    /// Input tokens used
    #[serde(default)]
    pub input_tokens: u32,
    /// Output tokens generated
    #[serde(default)]
    pub output_tokens: u32,
    /// Detailed input token breakdown
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_token_details: Option<InputTokenDetails>,
    /// Detailed output token breakdown
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_token_details: Option<OutputTokenDetails>,
}

/// Detailed breakdown of input tokens.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InputTokenDetails {
    /// Tokens from cached content
    #[serde(default)]
    pub cached_tokens: u32,
    /// Tokens from text content
    #[serde(default)]
    pub text_tokens: u32,
    /// Tokens from audio content
    #[serde(default)]
    pub audio_tokens: u32,
}

/// Detailed breakdown of output tokens.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OutputTokenDetails {
    /// Tokens for text output
    #[serde(default)]
    pub text_tokens: u32,
    /// Tokens for audio output
    #[serde(default)]
    pub audio_tokens: u32,
}

// ============================================================================
// Response Configuration
// ============================================================================

/// Configuration for creating a response.
///
/// Used in `response.create` events to customize the response.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResponseConfig {
    /// Modalities to generate (text, audio, or both)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<Modality>>,

    /// Instructions specific to this response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,

    /// Voice to use for audio output
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice: Option<Voice>,

    /// Format for output audio
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_audio_format: Option<super::session::AudioFormat>,

    /// Sampling temperature
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Maximum output tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_response_output_tokens: Option<MaxResponseOutputTokens>,

    /// Conversation items to use for context
    /// (allows creating response from specific context)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation: Option<ResponseConversation>,

    /// Additional metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

/// Conversation context for response generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponseConversation {
    /// Use items from conversation (auto mode)
    Auto,
    /// Use none of the conversation items
    None,
}

// ============================================================================
// Response Object
// ============================================================================

/// A response object returned in `response.created` and `response.done` events.
///
/// Represents the state of an in-progress or completed response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    /// Unique identifier for this response
    pub id: String,
    /// Object type (always "realtime.response")
    pub object: String,
    /// Status of the response
    pub status: ResponseStatus,
    /// Details about the status (e.g., cancellation reason)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status_details: Option<ResponseStatusDetails>,
    /// Output items generated by this response
    pub output: Vec<ConversationItem>,
    /// Usage statistics (populated when response is done)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<ResponseUsage>,
    /// Additional metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

impl Response {
    /// Check if the response is still in progress
    pub fn is_in_progress(&self) -> bool {
        self.status == ResponseStatus::InProgress
    }

    /// Check if the response completed successfully
    pub fn is_completed(&self) -> bool {
        self.status == ResponseStatus::Completed
    }

    /// Check if the response failed or was cancelled
    pub fn is_failed(&self) -> bool {
        matches!(
            self.status,
            ResponseStatus::Failed | ResponseStatus::Cancelled
        )
    }

    /// Get all function calls from the response output
    pub fn function_calls(&self) -> impl Iterator<Item = &ConversationItem> {
        self.output.iter().filter(|item| item.is_function_call())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_response_status_serialization() {
        assert_eq!(
            serde_json::to_string(&ResponseStatus::InProgress).unwrap(),
            "\"in_progress\""
        );
        assert_eq!(
            serde_json::to_string(&ResponseStatus::Completed).unwrap(),
            "\"completed\""
        );
        assert_eq!(
            serde_json::to_string(&ResponseStatus::Cancelled).unwrap(),
            "\"cancelled\""
        );
    }

    #[test]
    fn test_response_config_default() {
        let config = ResponseConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        assert_eq!(json, "{}");
    }

    #[test]
    fn test_response_config_with_options() {
        let config = ResponseConfig {
            modalities: Some(vec![Modality::Text, Modality::Audio]),
            instructions: Some("Be concise".to_string()),
            temperature: Some(0.8),
            ..Default::default()
        };
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"modalities\""));
        assert!(json.contains("\"instructions\""));
        assert!(json.contains("\"temperature\""));
    }

    #[test]
    fn test_response_usage() {
        let usage = ResponseUsage {
            total_tokens: 100,
            input_tokens: 50,
            output_tokens: 50,
            input_token_details: Some(InputTokenDetails {
                cached_tokens: 10,
                text_tokens: 30,
                audio_tokens: 10,
            }),
            output_token_details: Some(OutputTokenDetails {
                text_tokens: 25,
                audio_tokens: 25,
            }),
        };
        let json = serde_json::to_string(&usage).unwrap();
        assert!(json.contains("\"total_tokens\":100"));
        assert!(json.contains("\"cached_tokens\":10"));
    }

    #[test]
    fn test_response_deserialization() {
        let json = r#"{
            "id": "resp_123",
            "object": "realtime.response",
            "status": "completed",
            "output": []
        }"#;
        let response: Response = serde_json::from_str(json).unwrap();
        assert_eq!(response.id, "resp_123");
        assert!(response.is_completed());
    }
}
