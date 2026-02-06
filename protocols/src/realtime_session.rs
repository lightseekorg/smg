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
pub struct RealtimeSessionCreateRequest {
    #[serde(rename = "type")]
    pub r#type: RealtimeSessionType,
    pub audio: Option<RealtimeAudioConfig>,
    pub include: Option<Vec<RealtimeIncludeOption>>,
    pub instructions: Option<String>,
    pub max_output_tokens: Option<MaxOutputTokens>,
    pub model: Option<String>,
    #[serde(default = "audio")]
    pub output_modalities: Option<Vec<OutputModality>>,
    pub prompt: Option<ResponsePrompt>,
    pub tool_choice: Option<RealtimeToolChoiceConfig>,
    pub tools: Option<RealtimeToolsConfig>,
    pub tracing: Option<RealtimeTracingConfig>,
    pub truncation: Option<RealtimeTruncation>,
}

// ============================================================================
// Session Object (RealtimeSessionCreateResponseGA)
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeSessionCreateResponse {
    pub client_secret: RealtimeSessionClientSecret,
    #[serde(rename = "type")]
    pub r#type: RealtimeSessionType,
    pub audio: Option<RealtimeAudioConfig>,
    pub include: Option<Vec<RealtimeIncludeOption>>,
    pub instructions: Option<String>,
    pub max_output_tokens: Option<MaxOutputTokens>,
    pub model: Option<String>,
    #[serde(default = "audio")]
    pub output_modalities: Option<Vec<OutputModality>>,
    pub prompt: Option<ResponsePrompt>,
    pub tool_choice: Option<RealtimeToolChoiceConfig>,
    pub tools: Option<Vec<RealtimeToolsConfig>>,
    pub tracing: Option<RealtimeTracingConfig>,
    pub truncation: Option<RealtimeTruncation>,
}

// ============================================================================
// Transcription Session Configuration
// (RealtimeTranscriptionSessionCreateRequestGA)
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeTranscriptionSessionCreateRequest {
    #[serde(rename = "type")]
    pub r#type: RealtimeTranscriptionSessionType,
    pub audio: Option<RealtimeTranscriptionAudioConfig>,
    pub include: Option<Vec<RealtimeIncludeOption>>,
}

// ============================================================================
// Transcription Session Object from Create Response
// (RealtimeTranscriptionSessionCreateResponseGA)
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeTranscriptionSessionCreateResponse {
    pub id: String,
    pub object: String,
    #[serde(rename = "type")]
    pub r#type: RealtimeTranscriptionSessionType,
    pub audio: Option<RealtimeTranscriptionAudioConfig>,
    pub expires_at: Option<i64>,
    pub include: Option<Vec<RealtimeIncludeOption>>,
}

// ============================================================================
// Audio Formats
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RealtimeAudioFormats {
    #[serde(rename = "audio/pcm")]
    Pcm {
        /// Sample rate. Only 24000 is supported.
        #[serde(skip_serializing_if = "Option::is_none")]
        rate: Option<u32>,
    },
    #[serde(rename = "audio/pcmu")]
    Pcmu,
    #[serde(rename = "audio/pcma")]
    Pcma,
}

// ============================================================================
// Audio Transcription
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioTranscription {
    pub language: Option<String>,
    pub model: Option<String>,
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
/// Used only for `semantic_vad` mode. The eagerness of the model to respond. 
/// `low` will wait longer for the user to continue speaking, `high` will respond more quickly. 
/// `auto` is the default and is equivalent to `medium`. `low`, `medium`, and `high` have max timeouts of 8s, 4s, and 2s respectively.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SemanticVadEagerness {
    Low,
    Medium,
    High,
    #[default]
    Auto,
}

impl SemanticVadEagerness {
    /// Resolves `Auto` to `Medium` per the OpenAI spec.
    pub fn resolve(&self) -> &Self {
        match self {
            Self::Auto => &Self::Medium,
            other => other,
        }
    }
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TurnDetection {
    #[serde(rename = "server_vad")]
    ServerVad {
        create_response: Option<bool>,
        idle_timeout_ms: Option<u32>,
        interrupt_response: Option<bool>,
        prefix_padding_ms: Option<u32>,
        silence_duration_ms: Option<u32>,
        threshold: Option<f64>,
    },
    #[serde(rename = "semantic_vad")]
    SemanticVad {
        create_response: Option<bool>,
        eagerness: Option<SemanticVadEagerness>,
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
    VoiceIDsShared(String),
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

fn audio() -> Option<Vec<OutputModality>> {
    Some(vec![OutputModality::Audio])
}

// ============================================================================
// Tracing
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracingConfig {
    pub group_id: Option<String>,
    pub metadata: Option<Value>,
    pub workflow_name: Option<String>,
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
pub enum RealtimeTracingConfig {
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
pub enum RealtimeToolsConfig {
    #[serde(rename = "function")]
    RealtimeFunctionTool {
        description: Option<String>,
        name: Option<String>,
        parameters: Option<Value>,
    },
    #[serde(rename = "mcp")]
    McpTool {
        server_label: String,
        allowed_tools: Option<McpAllowedTools>,
        authorization: Option<String>,
        connector_id: Option<ConnectorId>,
        headers: Option<HashMap<String, String>>,
        require_approval: Option<McpToolApproval>,
        server_description: Option<String>,
        server_url: Option<String>,
    },
}

impl std::fmt::Debug for RealtimeToolsConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RealtimeFunctionTool {
                name,
                description,
                parameters,
            } => f
                .debug_struct("Function")
                .field("name", name)
                .field("description", description)
                .field("parameters", parameters)
                .finish(),
            Self::McpTool {
                server_label,
                server_url,
                connector_id,
                authorization,
                server_description,
                headers,
                allowed_tools,
                require_approval,
            } => f
                .debug_struct("McpTool")
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
// MCP Tool Filter
// ============================================================================

/// List of allowed tool names or a filter object.
///
/// Variant order matters for `#[serde(untagged)]`: serde tries `List` first
/// (JSON array). A JSON object falls through to `Filter`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum McpAllowedTools {
    List(Vec<String>),
    Filter(McpToolFilter),
}

/// A filter object to specify which tools are allowed.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolFilter {
    pub read_only: Option<bool>,
    pub tool_names: Option<Vec<String>>,
}

/// Approval policy for MCP tools: a filter object or `"always"`/`"never"`.
///
/// Variant order matters for `#[serde(untagged)]`: serde tries `Setting` first
/// (plain string). A JSON object falls through to `Filter`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum McpToolApproval {
    Setting(McpToolApprovalSetting),
    Filter(McpToolApprovalFilter),
}

/// Single approval policy for all tools.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum McpToolApprovalSetting {
    Always,
    Never,
}

/// Granular approval filter specifying which tools always/never require approval.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolApprovalFilter {
    pub always: Option<McpToolFilter>,
    pub never: Option<McpToolFilter>,
}

// ============================================================================
// Tool Choice
// ============================================================================

/// `"none"`, `"auto"`, `"required"`, or a specific function/MCP tool reference.
///
/// Variant order matters for `#[serde(untagged)]`: serde tries `Options` first
/// (plain string). A JSON object fails and falls through to `Function`/`Mcp`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RealtimeToolChoiceConfig {
    Options(ToolChoiceOptions),
    Function(ToolChoiceFunction),
    Mcp(ToolChoiceMcp),
}

/// Controls which (if any) tool is called by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoiceOptions {
    None,
    Auto,
    Required,
}

/// Force the model to call a specific function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolChoiceFunction {
    pub name: String,
    #[serde(rename = "type")]
    pub r#type: ToolChoiceFunctionType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolChoiceFunctionType {
    #[serde(rename = "function")]
    Function,
}

/// Force the model to call a specific tool on a remote MCP server.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolChoiceMcp {
    pub server_label: String,
    #[serde(rename = "type")]
    pub r#type: ToolChoiceMcpType,
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolChoiceMcpType {
    #[serde(rename = "mcp")]
    Mcp,
}

// ============================================================================
// Max Output Tokens
// ============================================================================

/// Integer token limit (1–4096) or `"inf"` for the maximum available tokens.
/// Defaults to `"inf"`.
///
/// Variant order matters for `#[serde(untagged)]`: serde tries `Integer` first.
/// A JSON number succeeds immediately; the string `"inf"` fails `Integer` and
/// falls through to `Inf`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MaxOutputTokens {
    /// An integer between 1 and 4096.
    Integer(u32),
    Inf(InfMarker),
}

impl Default for MaxOutputTokens {
    fn default() -> Self {
        Self::Inf(InfMarker::Inf)
    }
}

/// The literal string `"inf"`. Used by [`MaxOutputTokens::Inf`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InfMarker {
    #[serde(rename = "inf")]
    Inf,
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
    pub retention_ratio: f64,
    #[serde(rename = "type")]
    pub r#type: RetentionRatioTruncationType,
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
pub enum RealtimeTruncation {
    Mode(TruncationMode),
    RetentionRatio(RetentionRatioTruncation),
}

// ============================================================================
// Prompt
// ============================================================================

/// Reference to a prompt template and its variables.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsePrompt {
    pub id: String,
    pub variables: Option<HashMap<String, PromptVariable>>,
    pub version: Option<String>,
}

/// A prompt variable value: plain string or a typed input (text, image, file).
///
/// Variant order matters for `#[serde(untagged)]`: a bare JSON string succeeds
/// as `String`; a JSON object falls through to `Typed`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PromptVariable {
    String(String),
    Typed(PromptVariableTyped),
}

/// Typed prompt variable input.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum PromptVariableTyped {
    #[serde(rename = "input_text")]
    ResponseInputText { text: String },
    #[serde(rename = "input_image")]
    ResponseInputImage {
        detail: Option<Detail>,
        file_id: Option<String>,
        image_url: Option<String>,
    },
    #[serde(rename = "input_file")]
    ResponseInputFile {
        file_data: Option<String>,
        file_id: Option<String>,
        file_url: Option<String>,
        filename: Option<String>,
    },
}

/// Image detail level for [`PromptVariableTyped::InputImage`].
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Detail {
    Low,
    High,
    #[default]
    Auto,
}

// ============================================================================
// Client Secret
// ============================================================================

#[derive(Clone, Serialize, Deserialize)]
pub struct RealtimeSessionClientSecret {
    pub expires_at: i64,
    pub value: String,
}

impl std::fmt::Debug for RealtimeSessionClientSecret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RealtimeSessionClientSecret")
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
pub struct RealtimeAudioConfigInput {
    pub format: Option<RealtimeAudioFormats>,
    pub noise_reduction: Option<NoiseReduction>,
    pub transcription: Option<AudioTranscription>,
    pub turn_detection: Option<TurnDetection>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeAudioConfigOutput {
    pub format: Option<RealtimeAudioFormats>,
    pub speed: Option<f64>,
    pub voice: Option<Voice>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeAudioConfig {
    pub input: Option<RealtimeAudioConfigInput>,
    pub output: Option<RealtimeAudioConfigOutput>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeTranscriptionAudioConfig {
    pub input: Option<RealtimeAudioConfigInput>,
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
