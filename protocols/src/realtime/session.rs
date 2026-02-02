//! Session configuration and types for the Realtime API.
//!
//! This module contains types for configuring realtime sessions, including
//! audio formats, voice options, turn detection, and tool definitions.

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ============================================================================
// Audio Format
// ============================================================================

/// Audio format for input/output.
///
/// The Realtime API supports three audio formats:
/// - `pcm16`: 16-bit PCM audio (default)
/// - `g711_ulaw`: G.711 μ-law
/// - `g711_alaw`: G.711 A-law
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AudioFormat {
    #[default]
    Pcm16,
    G711Ulaw,
    G711Alaw,
}

// ============================================================================
// Voice
// ============================================================================

/// Voice options for audio output.
///
/// These are the available voices for text-to-speech in the Realtime API.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Voice {
    Alloy,
    Ash,
    Ballad,
    Coral,
    Echo,
    Sage,
    Shimmer,
    Verse,
}

impl Default for Voice {
    fn default() -> Self {
        Self::Alloy
    }
}

// ============================================================================
// Modality
// ============================================================================

/// Modality types for input/output.
///
/// Specifies whether the session handles text, audio, or both.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    Text,
    Audio,
}

// ============================================================================
// Turn Detection
// ============================================================================

/// Turn detection configuration.
///
/// Controls how the API detects when the user has finished speaking.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TurnDetection {
    /// Server-side Voice Activity Detection (VAD)
    ServerVad {
        /// Activation threshold (0.0-1.0, default 0.5)
        #[serde(skip_serializing_if = "Option::is_none")]
        threshold: Option<f32>,
        /// Audio to include before speech starts (ms, default 300)
        #[serde(skip_serializing_if = "Option::is_none")]
        prefix_padding_ms: Option<u32>,
        /// Duration of silence to detect end of speech (ms, default 500)
        #[serde(skip_serializing_if = "Option::is_none")]
        silence_duration_ms: Option<u32>,
        /// Whether to automatically create a response (default true)
        #[serde(skip_serializing_if = "Option::is_none")]
        create_response: Option<bool>,
    },
    /// Disable turn detection (manual control)
    #[serde(rename = "none")]
    Disabled,
}

impl Default for TurnDetection {
    fn default() -> Self {
        Self::ServerVad {
            threshold: None,
            prefix_padding_ms: None,
            silence_duration_ms: None,
            create_response: None,
        }
    }
}

// ============================================================================
// Input Audio Transcription
// ============================================================================

/// Input audio transcription configuration.
///
/// Enables transcription of user audio input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputAudioTranscription {
    /// The transcription model to use (e.g., "whisper-1")
    pub model: String,
}

impl Default for InputAudioTranscription {
    fn default() -> Self {
        Self {
            model: "whisper-1".to_string(),
        }
    }
}

// ============================================================================
// Tool Definitions
// ============================================================================

/// Tool definition for the Realtime API.
///
/// Defines a function that can be called by the model during a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeTool {
    /// The type of tool (always "function")
    #[serde(rename = "type")]
    pub tool_type: String,
    /// The name of the function
    pub name: String,
    /// A description of what the function does
    pub description: String,
    /// JSON Schema for the function parameters
    pub parameters: Value,
}

impl RealtimeTool {
    /// Create a new function tool
    pub fn function(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            tool_type: "function".to_string(),
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }
}

// ============================================================================
// Tool Choice
// ============================================================================

/// Tool choice configuration.
///
/// Controls how the model selects which tools to use.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// Let the model decide (default)
    Mode(ToolChoiceMode),
    /// Force a specific function
    Function(ToolChoiceFunction),
}

/// Tool choice mode
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoiceMode {
    /// Let the model decide whether to call tools
    Auto,
    /// Don't call any tools
    None,
    /// Force the model to call a tool
    Required,
}

/// Specific function choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolChoiceFunction {
    #[serde(rename = "type")]
    pub choice_type: String,
    pub function: ToolChoiceFunctionName,
}

/// Function name for tool choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolChoiceFunctionName {
    pub name: String,
}

impl Default for ToolChoice {
    fn default() -> Self {
        Self::Mode(ToolChoiceMode::Auto)
    }
}

impl ToolChoice {
    pub fn auto() -> Self {
        Self::Mode(ToolChoiceMode::Auto)
    }

    pub fn none() -> Self {
        Self::Mode(ToolChoiceMode::None)
    }

    pub fn required() -> Self {
        Self::Mode(ToolChoiceMode::Required)
    }

    pub fn function(name: impl Into<String>) -> Self {
        Self::Function(ToolChoiceFunction {
            choice_type: "function".to_string(),
            function: ToolChoiceFunctionName { name: name.into() },
        })
    }
}

// ============================================================================
// Max Tokens
// ============================================================================

/// Maximum response output tokens configuration.
///
/// Can be a specific number or "inf" for unlimited.
#[derive(Debug, Clone)]
pub enum MaxResponseOutputTokens {
    /// Unlimited tokens
    Inf,
    /// Specific token limit
    Number(u32),
}

impl Serialize for MaxResponseOutputTokens {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Inf => serializer.serialize_str("inf"),
            Self::Number(n) => serializer.serialize_u32(*n),
        }
    }
}

impl<'de> Deserialize<'de> for MaxResponseOutputTokens {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{self, Visitor};

        struct MaxTokensVisitor;

        impl<'de> Visitor<'de> for MaxTokensVisitor {
            type Value = MaxResponseOutputTokens;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("\"inf\" or a positive integer")
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                if value == "inf" {
                    Ok(MaxResponseOutputTokens::Inf)
                } else {
                    Err(de::Error::custom(format!("expected \"inf\", got \"{}\"", value)))
                }
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                if value <= u32::MAX as u64 {
                    Ok(MaxResponseOutputTokens::Number(value as u32))
                } else {
                    Err(de::Error::custom(format!("value {} is too large for u32", value)))
                }
            }

            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                if value >= 0 && value <= u32::MAX as i64 {
                    Ok(MaxResponseOutputTokens::Number(value as u32))
                } else {
                    Err(de::Error::custom(format!("value {} is out of range", value)))
                }
            }
        }

        deserializer.deserialize_any(MaxTokensVisitor)
    }
}

impl Default for MaxResponseOutputTokens {
    fn default() -> Self {
        Self::Inf
    }
}

// ============================================================================
// Session Configuration
// ============================================================================

/// Session configuration for `session.update` events.
///
/// All fields are optional - only specified fields will be updated.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SessionConfig {
    /// The modalities the session supports (text, audio, or both)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<Modality>>,

    /// System instructions for the model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,

    /// Voice for audio output
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice: Option<Voice>,

    /// Format for input audio
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_audio_format: Option<AudioFormat>,

    /// Format for output audio
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_audio_format: Option<AudioFormat>,

    /// Configuration for input audio transcription
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_audio_transcription: Option<InputAudioTranscription>,

    /// Turn detection configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_detection: Option<TurnDetection>,

    /// Tools available to the model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<RealtimeTool>>,

    /// How the model should choose tools
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    /// Sampling temperature (0.6-1.2)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Maximum output tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_response_output_tokens: Option<MaxResponseOutputTokens>,
}

// ============================================================================
// Full Session Object
// ============================================================================

/// Full session object returned in `session.created` and `session.updated` events.
///
/// This represents the complete state of a session as returned by the API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    /// Unique identifier for the session
    pub id: String,
    /// Object type (always "realtime.session")
    pub object: String,
    /// The model being used
    pub model: String,
    /// Unix timestamp when the session expires
    pub expires_at: u64,
    /// Active modalities for this session
    pub modalities: Vec<Modality>,
    /// System instructions
    pub instructions: String,
    /// Voice for audio output
    pub voice: Voice,
    /// Input audio format
    pub input_audio_format: AudioFormat,
    /// Output audio format
    pub output_audio_format: AudioFormat,
    /// Input audio transcription config (if enabled)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_audio_transcription: Option<InputAudioTranscription>,
    /// Turn detection configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_detection: Option<TurnDetection>,
    /// Available tools
    pub tools: Vec<RealtimeTool>,
    /// Tool choice configuration
    pub tool_choice: ToolChoice,
    /// Sampling temperature
    pub temperature: f32,
    /// Maximum response output tokens
    pub max_response_output_tokens: MaxResponseOutputTokens,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_format_serialization() {
        assert_eq!(serde_json::to_string(&AudioFormat::Pcm16).unwrap(), "\"pcm16\"");
        assert_eq!(serde_json::to_string(&AudioFormat::G711Ulaw).unwrap(), "\"g711_ulaw\"");
        assert_eq!(serde_json::to_string(&AudioFormat::G711Alaw).unwrap(), "\"g711_alaw\"");
    }

    #[test]
    fn test_voice_serialization() {
        assert_eq!(serde_json::to_string(&Voice::Alloy).unwrap(), "\"alloy\"");
        assert_eq!(serde_json::to_string(&Voice::Sage).unwrap(), "\"sage\"");
    }

    #[test]
    fn test_turn_detection_serialization() {
        let vad = TurnDetection::ServerVad {
            threshold: Some(0.5),
            prefix_padding_ms: Some(300),
            silence_duration_ms: Some(500),
            create_response: Some(true),
        };
        let json = serde_json::to_string(&vad).unwrap();
        assert!(json.contains("\"type\":\"server_vad\""));

        let disabled = TurnDetection::Disabled;
        let json = serde_json::to_string(&disabled).unwrap();
        assert!(json.contains("\"type\":\"none\""));
    }

    #[test]
    fn test_session_config_default() {
        let config = SessionConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        assert_eq!(json, "{}");
    }

    #[test]
    fn test_tool_choice_variants() {
        let auto = ToolChoice::auto();
        assert_eq!(serde_json::to_string(&auto).unwrap(), "\"auto\"");

        let func = ToolChoice::function("my_tool");
        let json = serde_json::to_string(&func).unwrap();
        assert!(json.contains("\"name\":\"my_tool\""));
    }
}
