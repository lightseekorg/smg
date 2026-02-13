// OpenAI Realtime Session API types
// https://platform.openai.com/docs/api-reference/realtime

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ============================================================================
// Session Configuration (RealtimeSessionCreateRequestGA)
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeSessionConfig {
    #[serde(rename = "type")]
    pub r#type: RealtimeSessionType,
    #[serde(default = "default_output_modalities")]
    pub output_modalities: Vec<OutputModality>,
    pub model: Option<String>,
    pub instructions: Option<String>,
    pub audio: Option<RealtimeSessionAudioConfig>,
    pub include: Option<Vec<RealtimeIncludeOption>>,
    pub tracing: Option<Tracing>,
    pub tools: Option<Vec<RealtimeTool>>,
    pub tool_choice: Option<RealtimeToolChoice>,
    pub max_output_tokens: Option<MaxOutputTokens>,
    pub truncation: Option<Truncation>,
    pub prompt: Option<Prompt>,
}

// ============================================================================
// Session Object (RealtimeSessionCreateResponseGA)
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeSessionObject {
    pub client_secret: ClientSecret,
    #[serde(flatten)]
    pub config: RealtimeSessionConfig,
}

// ============================================================================
// Transcription Session Configuration
// (RealtimeTranscriptionSessionCreateRequestGA)
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeTranscriptionSessionConfig {
    #[serde(rename = "type")]
    pub r#type: RealtimeTranscriptionSessionType,
    pub audio: Option<RealtimeTranscriptionAudioConfig>,
    pub include: Option<Vec<RealtimeIncludeOption>>,
}

// ============================================================================
// Transcription Session Object
// (RealtimeTranscriptionSessionCreateResponseGA)
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeTranscriptionSessionObject {
    pub id: String,
    pub object: String,
    pub expires_at: Option<i64>,
    #[serde(flatten)]
    pub config: RealtimeTranscriptionSessionConfig,
}

// ============================================================================
// Audio Formats
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RealtimeAudioFormat {
    #[serde(rename = "audio/pcm")]
    Pcm { rate: Option<u32> },
    #[serde(rename = "audio/pcmu")]
    Pcmu {},
    #[serde(rename = "audio/pcma")]
    Pcma {},
}

// ============================================================================
// Audio Transcription
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioTranscription {
    pub model: Option<String>,
    pub language: Option<String>,
    pub prompt: Option<String>,
}

// ============================================================================
// Noise Reduction
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NoiseReductionType {
    NearField,
    FarField,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseReduction {
    #[serde(rename = "type")]
    pub r#type: NoiseReductionType,
}

// ============================================================================
// Turn Detection
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SemanticVadEagerness {
    Low,
    Medium,
    High,
    #[default]
    Auto,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TurnDetection {
    #[serde(rename = "server_vad")]
    ServerVad {
        threshold: Option<f64>,
        prefix_padding_ms: Option<u32>,
        silence_duration_ms: Option<u32>,
        create_response: Option<bool>,
        interrupt_response: Option<bool>,
        idle_timeout_ms: Option<u32>,
    },
    #[serde(rename = "semantic_vad")]
    SemanticVad {
        eagerness: Option<SemanticVadEagerness>,
        create_response: Option<bool>,
        interrupt_response: Option<bool>,
    },
}

// ============================================================================
// Voice
// ============================================================================

/// Built-in voice name (e.g. "alloy", "ash") or custom voice reference.
///
/// Variant order matters for `#[serde(untagged)]`: a bare JSON string (e.g. `"alloy"`)
/// always deserializes as `BuiltIn`. A JSON object `{"id": "..."}` fails `BuiltIn`
/// and falls through to `Custom`. The two forms are structurally distinct (string vs
/// object) per the OpenAI spec, so there is no ambiguity.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Voice {
    BuiltIn(String),
    Custom { id: String },
}

// ============================================================================
// Output Modality
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputModality {
    Text,
    Audio,
}

fn default_output_modalities() -> Vec<OutputModality> {
    vec![OutputModality::Audio]
}
// ============================================================================
// Tracing
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracingConfig {
    pub workflow_name: Option<String>,
    pub group_id: Option<String>,
    pub metadata: Option<Value>,
}

/// The tracing mode. Always `"auto"`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TracingMode {
    #[serde(rename = "auto")]
    Auto,
}

/// Either the string `"auto"` or a granular tracing configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Tracing {
    Mode(TracingMode),
    Config(TracingConfig),
}

// ============================================================================
// Connector ID
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConnectorId {
    ConnectorDropbox,
    ConnectorGmail,
    ConnectorGooglecalendar,
    ConnectorGoogledrive,
    ConnectorMicrosoftteams,
    ConnectorOutlookcalendar,
    ConnectorOutlookemail,
    ConnectorSharepoint,
}

// ============================================================================
// Tools
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RealtimeTool {
    #[serde(rename = "function")]
    Function {
        name: String,
        description: Option<String>,
        parameters: Option<Value>,
    },
    #[serde(rename = "mcp")]
    Mcp {
        server_label: String,
        server_url: Option<String>,
        connector_id: Option<ConnectorId>,
        authorization: Option<String>,
        server_description: Option<String>,
        headers: Option<HashMap<String, String>>,
        allowed_tools: Option<Value>,
        require_approval: Option<Value>,
    },
}

impl std::fmt::Debug for RealtimeTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Function {
                name,
                description,
                parameters,
            } => f
                .debug_struct("Function")
                .field("name", name)
                .field("description", description)
                .field("parameters", parameters)
                .finish(),
            Self::Mcp {
                server_label,
                server_url,
                connector_id,
                authorization,
                server_description,
                headers,
                allowed_tools,
                require_approval,
            } => f
                .debug_struct("Mcp")
                .field("server_label", server_label)
                .field("server_url", server_url)
                .field("connector_id", connector_id)
                .field(
                    "authorization",
                    &authorization.as_ref().map(|_| "[REDACTED]"),
                )
                .field("server_description", server_description)
                .field("headers", &headers.as_ref().map(|_| "[REDACTED]"))
                .field("allowed_tools", allowed_tools)
                .field("require_approval", require_approval)
                .finish(),
        }
    }
}

// ============================================================================
// Tool Choice
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RealtimeToolChoiceSpecific {
    Function {
        name: String,
    },
    Mcp {
        server_label: String,
        name: Option<String>,
    },
}

/// The tool choice mode.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RealtimeToolChoiceMode {
    None,
    #[default]
    Auto,
    Required,
}

/// `"none"`, `"auto"`, `"required"`, or a specific function/MCP tool reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RealtimeToolChoice {
    Mode(RealtimeToolChoiceMode),
    Specific(RealtimeToolChoiceSpecific),
}

// ============================================================================
// Max Output Tokens
// ============================================================================

/// The infinite token marker. Always `"inf"`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InfToken {
    #[serde(rename = "inf")]
    Inf,
}

/// Integer token limit or `"inf"` for unlimited.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MaxOutputTokens {
    Integer(u32),
    Inf(InfToken),
}

// ============================================================================
// Truncation
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruncationTokenLimits {
    pub post_instructions: Option<u32>,
}

/// The retention ratio truncation type. Always `"retention_ratio"`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetentionRatioTruncationType {
    #[serde(rename = "retention_ratio")]
    RetentionRatio,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionRatioTruncation {
    #[serde(rename = "type")]
    pub r#type: RetentionRatioTruncationType,
    pub retention_ratio: f64,
    pub token_limits: Option<TruncationTokenLimits>,
}

/// The truncation mode.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TruncationMode {
    #[default]
    Auto,
    Disabled,
}

/// `"auto"`, `"disabled"`, or a retention ratio configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Truncation {
    Mode(TruncationMode),
    RetentionRatio(RetentionRatioTruncation),
}

// ============================================================================
// Prompt
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prompt {
    pub id: String,
    pub version: Option<String>,
    pub variables: Option<Value>,
}

// ============================================================================
// Client Secret
// ============================================================================

#[derive(Clone, Serialize, Deserialize)]
pub struct ClientSecret {
    pub value: String,
    pub expires_at: i64,
}

impl std::fmt::Debug for ClientSecret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClientSecret")
            .field("value", &"[REDACTED]")
            .field("expires_at", &self.expires_at)
            .finish()
    }
}

// ============================================================================
// Audio Configuration
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeAudioInputConfig {
    pub format: Option<RealtimeAudioFormat>,
    pub transcription: Option<AudioTranscription>,
    pub noise_reduction: Option<NoiseReduction>,
    pub turn_detection: Option<TurnDetection>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeAudioOutputConfig {
    pub format: Option<RealtimeAudioFormat>,
    pub voice: Option<Voice>,
    pub speed: Option<f64>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeSessionAudioConfig {
    pub input: Option<RealtimeAudioInputConfig>,
    pub output: Option<RealtimeAudioOutputConfig>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeTranscriptionAudioConfig {
    pub input: Option<RealtimeAudioInputConfig>,
}

// ============================================================================
// Include Options
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RealtimeIncludeOption {
    #[serde(rename = "item.input_audio_transcription.logprobs")]
    InputAudioTranscriptionLogprobs,
}

// ============================================================================
// Session Type
// ============================================================================

/// The type of session. Always `"realtime"` for the Realtime API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RealtimeSessionType {
    #[serde(rename = "realtime")]
    Realtime,
}

// ============================================================================
// TranscriptionSession Type
// ============================================================================

/// The type of session. Always `"transcription"` for the Realtime API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RealtimeTranscriptionSessionType {
    #[serde(rename = "transcription")]
    Transcription,
}
