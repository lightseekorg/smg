//! Session configuration and types for the Realtime API.
//!
//! This module contains types for configuring realtime sessions, including
//! audio formats, voice options, turn detection, and tool definitions.
//! Types match the OpenAI Realtime API v2.3.0 GA specification.

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ============================================================================
// Session Configuration (Request)
// ============================================================================

/// Session configuration for `session.update` events.
///
/// All fields are optional - only specified fields will be updated.
/// Matches the GA `RealtimeSessionCreateRequestGA` schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    /// Session type (always "realtime" for Realtime API, defaults to "realtime")
    #[serde(rename = "type", default = "session_type_default")]
    pub session_type: String,

    /// Output modalities (["text"] or ["audio"])
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_modalities: Option<Vec<Modality>>,

    /// Model to use for the session
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// System instructions for the model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,

    /// Audio input/output configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<AudioConfig>,

    /// Additional fields to include in server outputs
    /// (e.g., ["item.input_audio_transcription.logprobs"])
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<String>>,

    /// Tracing configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tracing: Option<TracingConfig>,

    /// Tools available to the model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,

    /// How the model should choose tools
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    /// Maximum output tokens per response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<MaxResponseOutputTokens>,

    /// Truncation configuration for context management
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<Truncation>,

    /// Prompt template reference
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<Prompt>,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            session_type: "realtime".to_string(),
            output_modalities: None,
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
        }
    }
}

// ============================================================================
// Full Session Object (Response)
// ============================================================================

fn session_object_type() -> String {
    "realtime.session".to_string()
}

fn session_type_default() -> String {
    "realtime".to_string()
}

/// Full session object returned in `session.created` and `session.updated` events.
///
/// This represents the complete state of a session as returned by the API.
/// Matches the GA `RealtimeSessionCreateResponseGA` schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    /// Unique identifier for the session
    pub id: String,
    /// Object type (always "realtime.session")
    #[serde(default = "session_object_type")]
    pub object: String,
    /// Ephemeral client secret (only in session creation response)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_secret: Option<ClientSecret>,
    /// Session type (always "realtime")
    #[serde(rename = "type", default = "session_type_default")]
    pub session_type: String,
    /// Output modalities for this session
    #[serde(default)]
    pub output_modalities: Vec<Modality>,
    /// The model being used
    pub model: String,
    /// System instructions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// Audio input/output configuration
    #[serde(default)]
    pub audio: AudioConfig,
    /// Additional fields included in outputs
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<String>>,
    /// Tracing configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tracing: Option<TracingConfig>,
    /// Available tools
    #[serde(default)]
    pub tools: Vec<ToolDefinition>,
    /// Tool choice configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    /// Maximum output tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<MaxResponseOutputTokens>,
    /// Truncation configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<Truncation>,
    /// Prompt template reference
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<Prompt>,
    /// Unix timestamp when the session expires
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<u64>,
}

// ============================================================================
// Audio Configuration (nested GA structure)
// ============================================================================

/// Audio configuration containing input and output settings.
///
/// This is the nested `audio` object in the GA spec.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AudioConfig {
    /// Input audio configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<AudioInputConfig>,
    /// Output audio configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<AudioOutputConfig>,
}

/// Audio input configuration.
///
/// Nested under `audio.input` in the GA spec.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AudioInputConfig {
    /// Format for input audio
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<AudioFormat>,
    /// Input audio transcription configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transcription: Option<InputAudioTranscription>,
    /// Input audio noise reduction configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub noise_reduction: Option<InputAudioNoiseReduction>,
    /// Turn detection configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_detection: Option<TurnDetection>,
}

/// Audio output configuration.
///
/// Nested under `audio.output` in the GA spec.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AudioOutputConfig {
    /// Format for output audio
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<AudioFormat>,
    /// Voice for audio output (supports built-in and custom voices)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice: Option<VoiceOption>,
    /// Spoken response speed (0.25-1.5, default 1.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<f32>,
}

// ============================================================================
// Tool Definitions
// ============================================================================

/// Tool definition for the Realtime API.
///
/// Tools can be either function tools (called by the model) or MCP tools
/// (proxied through remote MCP servers).
#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ToolDefinition {
    /// A function tool that the model can call directly
    #[serde(rename = "function")]
    Function {
        /// The name of the function
        name: String,
        /// A description of what the function does
        description: String,
        /// JSON Schema for the function parameters
        parameters: Value,
    },
    /// An MCP tool that proxies through a remote MCP server
    #[serde(rename = "mcp")]
    Mcp {
        /// A label for this MCP server, used to identify it in tool calls
        server_label: String,
        /// The URL for the MCP server (one of `server_url` or `connector_id` required)
        #[serde(skip_serializing_if = "Option::is_none")]
        server_url: Option<String>,
        /// Connector ID for service connectors.
        /// One of `server_url` or `connector_id` required.
        #[serde(skip_serializing_if = "Option::is_none")]
        connector_id: Option<ConnectorId>,
        /// OAuth access token for the MCP server
        #[serde(skip_serializing_if = "Option::is_none")]
        authorization: Option<String>,
        /// Optional description of the MCP server
        #[serde(skip_serializing_if = "Option::is_none")]
        server_description: Option<String>,
        /// Optional HTTP headers to send to the MCP server
        /// (object mapping header names to values, or null)
        #[serde(skip_serializing_if = "Option::is_none")]
        headers: Option<Value>,
        /// Allowed tool names: string array or MCPToolFilter object, or null
        #[serde(skip_serializing_if = "Option::is_none")]
        allowed_tools: Option<Value>,
        /// Approval policy: "always", "never", or filter object
        /// with `always`/`never` MCPToolFilter fields. Defaults to "always".
        #[serde(skip_serializing_if = "Option::is_none")]
        require_approval: Option<Value>,
    },
}

impl ToolDefinition {
    /// Create a new function tool
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
    ) -> Self {
        Self::Function {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }

    /// Create a new MCP tool with a server URL
    pub fn mcp_with_url(server_label: impl Into<String>, server_url: impl Into<String>) -> Self {
        Self::Mcp {
            server_label: server_label.into(),
            server_url: Some(server_url.into()),
            connector_id: None,
            authorization: None,
            server_description: None,
            headers: None,
            allowed_tools: None,
            require_approval: None,
        }
    }

    /// Create a new MCP tool with a connector ID
    pub fn mcp_with_connector(server_label: impl Into<String>, connector_id: ConnectorId) -> Self {
        Self::Mcp {
            server_label: server_label.into(),
            server_url: None,
            connector_id: Some(connector_id),
            authorization: None,
            server_description: None,
            headers: None,
            allowed_tools: None,
            require_approval: None,
        }
    }
}

impl std::fmt::Debug for ToolDefinition {
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
            } => {
                let mut debug = f.debug_struct("Mcp");
                debug.field("server_label", server_label);
                debug.field("server_url", server_url);
                debug.field("connector_id", connector_id);
                // Redact sensitive authorization token
                debug.field(
                    "authorization",
                    &authorization.as_ref().map(|_| "[REDACTED]"),
                );
                debug.field("server_description", server_description);
                // Redact headers as they may contain sensitive tokens
                debug.field("headers", &headers.as_ref().map(|_| "[REDACTED]"));
                debug.field("allowed_tools", allowed_tools);
                debug.field("require_approval", require_approval);
                debug.finish()
            }
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
    /// Force a specific MCP tool
    Mcp(ToolChoiceMcp),
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

/// Specific MCP tool choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolChoiceMcp {
    #[serde(rename = "type")]
    pub choice_type: String,
    /// The MCP server label
    pub server_label: String,
    /// The tool name
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

    /// Force a specific MCP tool
    pub fn mcp(server_label: impl Into<String>, name: impl Into<String>) -> Self {
        Self::Mcp(ToolChoiceMcp {
            choice_type: "mcp".to_string(),
            server_label: server_label.into(),
            name: name.into(),
        })
    }
}

// ============================================================================
// Supporting Types
// ============================================================================

// --- Modality ---

/// Modality types for output.
///
/// Specifies whether the session handles text or audio output.
/// The GA spec uses `output_modalities` (not `modalities`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    Text,
    Audio,
}

// --- Audio Format ---

/// Audio format for input/output.
///
/// The GA Realtime API uses object-based audio formats:
/// - `audio/pcm`: PCM 16-bit audio at 24kHz
/// - `audio/pcmu`: G.711 μ-law
/// - `audio/pcma`: G.711 A-law
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AudioFormat {
    /// PCM 16-bit audio at 24kHz
    #[serde(rename = "audio/pcm")]
    Pcm {
        /// Sample rate (always 24000)
        #[serde(default = "default_pcm_rate")]
        rate: u32,
    },
    /// G.711 μ-law
    #[serde(rename = "audio/pcmu")]
    Pcmu,
    /// G.711 A-law
    #[serde(rename = "audio/pcma")]
    Pcma,
}

fn default_pcm_rate() -> u32 {
    24000
}

impl Default for AudioFormat {
    fn default() -> Self {
        Self::Pcm {
            rate: default_pcm_rate(),
        }
    }
}

impl AudioFormat {
    /// Create a PCM format (24kHz)
    pub fn pcm() -> Self {
        Self::Pcm {
            rate: default_pcm_rate(),
        }
    }

    /// Create a G.711 μ-law format
    pub fn pcmu() -> Self {
        Self::Pcmu
    }

    /// Create a G.711 A-law format
    pub fn pcma() -> Self {
        Self::Pcma
    }
}

// --- Voice ---

/// Voice options for audio output.
///
/// These are the available built-in voices for text-to-speech in the Realtime API.
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
    /// High quality voice (recommended)
    Marin,
    /// High quality voice (recommended)
    Cedar,
}

/// Voice option that supports both built-in voices and custom voice IDs.
///
/// Custom voices can be specified with an ID like `{ "id": "voice_1234" }`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum VoiceOption {
    /// A built-in voice
    BuiltIn(Voice),
    /// A custom voice with an ID
    Custom {
        /// The custom voice ID (e.g., "voice_1234")
        id: String,
    },
}

impl Default for VoiceOption {
    fn default() -> Self {
        Self::BuiltIn(Voice::Alloy)
    }
}

impl From<Voice> for VoiceOption {
    fn from(voice: Voice) -> Self {
        Self::BuiltIn(voice)
    }
}

// --- Turn Detection ---

/// Eagerness level for Semantic VAD.
///
/// Controls how eagerly the model responds when using semantic turn detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SemanticVadEagerness {
    /// Wait longer for user to continue (8s max timeout)
    Low,
    /// Balanced response timing (4s max timeout)
    Medium,
    /// Respond more quickly (2s max timeout)
    High,
    /// Equivalent to medium
    #[default]
    Auto,
}

/// Turn detection configuration.
///
/// Controls how the API detects when the user has finished speaking.
/// Supports both Server VAD (volume-based) and Semantic VAD (model-based).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TurnDetection {
    /// Server-side Voice Activity Detection (VAD)
    ///
    /// Detects speech start/end based on audio volume levels.
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
        /// Whether to interrupt ongoing response on speech start (default true)
        #[serde(skip_serializing_if = "Option::is_none")]
        interrupt_response: Option<bool>,
        /// Idle timeout in milliseconds (5000-30000)
        #[serde(skip_serializing_if = "Option::is_none")]
        idle_timeout_ms: Option<u32>,
    },
    /// Semantic Voice Activity Detection
    ///
    /// Uses a turn detection model to semantically estimate whether
    /// the user has finished speaking, with dynamic timeout based on
    /// probability. More natural but may have higher latency.
    SemanticVad {
        /// How eagerly the model should respond
        #[serde(skip_serializing_if = "Option::is_none")]
        eagerness: Option<SemanticVadEagerness>,
        /// Whether to automatically create a response (default true)
        #[serde(skip_serializing_if = "Option::is_none")]
        create_response: Option<bool>,
        /// Whether to interrupt ongoing response on speech start (default true)
        #[serde(skip_serializing_if = "Option::is_none")]
        interrupt_response: Option<bool>,
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
            interrupt_response: None,
            idle_timeout_ms: None,
        }
    }
}

// --- Noise Reduction ---

/// Type of noise reduction to apply to input audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NoiseReductionType {
    /// For headphones and close-talking microphones
    NearField,
    /// For laptop microphones and conference room setups
    FarField,
}

/// Input audio noise reduction configuration.
///
/// Filters audio before it is sent to VAD and the model.
/// Can improve VAD accuracy and model performance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputAudioNoiseReduction {
    /// The type of noise reduction to apply
    #[serde(rename = "type")]
    pub noise_type: NoiseReductionType,
}

impl InputAudioNoiseReduction {
    /// Create a near-field noise reduction config
    pub fn near_field() -> Self {
        Self {
            noise_type: NoiseReductionType::NearField,
        }
    }

    /// Create a far-field noise reduction config
    pub fn far_field() -> Self {
        Self {
            noise_type: NoiseReductionType::FarField,
        }
    }
}

// --- Input Audio Transcription ---

/// Input audio transcription configuration.
///
/// Enables transcription of user audio input. Transcription runs
/// asynchronously and should be treated as guidance rather than
/// precisely what the model heard.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct InputAudioTranscription {
    /// The transcription model to use (e.g., "whisper-1", "gpt-4o-transcribe")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Language hint for transcription (ISO-639-1 code, e.g., "en")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Optional prompt to guide transcription style
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
}

// --- Connector ID ---

/// Connector ID for MCP service connectors.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectorId {
    /// Dropbox
    #[serde(rename = "connector_dropbox")]
    Dropbox,
    /// Gmail
    #[serde(rename = "connector_gmail")]
    Gmail,
    /// Google Calendar
    #[serde(rename = "connector_googlecalendar")]
    GoogleCalendar,
    /// Google Drive
    #[serde(rename = "connector_googledrive")]
    GoogleDrive,
    /// Microsoft Teams
    #[serde(rename = "connector_microsoftteams")]
    MicrosoftTeams,
    /// Outlook Calendar
    #[serde(rename = "connector_outlookcalendar")]
    OutlookCalendar,
    /// Outlook Email
    #[serde(rename = "connector_outlookemail")]
    OutlookEmail,
    /// SharePoint
    #[serde(rename = "connector_sharepoint")]
    SharePoint,
}

// --- Max Tokens ---

/// Maximum response output tokens configuration.
///
/// Can be a specific number or "inf" for unlimited.
#[derive(Debug, Clone, Default)]
pub enum MaxResponseOutputTokens {
    /// Unlimited tokens
    #[default]
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
                    Err(de::Error::custom(format!(
                        "expected \"inf\", got \"{}\"",
                        value
                    )))
                }
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                if value <= u32::MAX as u64 {
                    Ok(MaxResponseOutputTokens::Number(value as u32))
                } else {
                    Err(de::Error::custom(format!(
                        "value {} is too large for u32",
                        value
                    )))
                }
            }

            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                if value >= 0 && value <= u32::MAX as i64 {
                    Ok(MaxResponseOutputTokens::Number(value as u32))
                } else {
                    Err(de::Error::custom(format!(
                        "value {} is out of range",
                        value
                    )))
                }
            }
        }

        deserializer.deserialize_any(MaxTokensVisitor)
    }
}

// --- Tracing Configuration ---

/// Tracing mode (simple string value).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TracingMode {
    /// Enables tracing with default settings
    Auto,
}

/// Tracing configuration for the Realtime API.
///
/// Traces can be written to the OpenAI Traces Dashboard.
/// Once enabled for a session, tracing cannot be disabled.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TracingConfig {
    /// Use "auto" for default tracing settings
    Auto(TracingMode),
    /// Granular tracing configuration
    Custom {
        /// Workflow name for the trace
        #[serde(skip_serializing_if = "Option::is_none")]
        workflow_name: Option<String>,
        /// Group ID for filtering/grouping in the dashboard
        #[serde(skip_serializing_if = "Option::is_none")]
        group_id: Option<String>,
        /// Arbitrary metadata for filtering in the dashboard
        #[serde(skip_serializing_if = "Option::is_none")]
        metadata: Option<Value>,
    },
}

// --- Truncation Configuration ---

/// Simple truncation mode.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TruncationMode {
    /// Default truncation strategy
    Auto,
    /// Disable truncation (emit errors on overflow)
    Disabled,
}

/// Truncation configuration for managing conversation context size.
///
/// Controls behavior when conversation tokens exceed the model's input limit.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Truncation {
    /// Simple truncation mode ("auto" or "disabled")
    Mode(TruncationMode),
    /// Retention ratio truncation strategy
    RetentionRatio {
        /// Always "retention_ratio"
        #[serde(rename = "type")]
        truncation_type: String,
        /// Fraction of tokens to retain (0.0-1.0)
        retention_ratio: f32,
        /// Optional custom token limits
        #[serde(skip_serializing_if = "Option::is_none")]
        token_limits: Option<Value>,
    },
}

impl Default for Truncation {
    fn default() -> Self {
        Self::Mode(TruncationMode::Auto)
    }
}

// --- Prompt Template ---

/// Reference to a prompt template.
///
/// Allows using pre-defined prompt templates with variable substitution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prompt {
    /// The unique identifier of the prompt template
    pub id: String,
    /// Optional version of the prompt template
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// Variables to substitute in the template
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variables: Option<Value>,
}

// --- Client Secret ---

/// Ephemeral client secret for client-side authentication.
///
/// Returned by the session creation API. Use this in client environments
/// instead of a standard API token.
#[derive(Clone, Serialize, Deserialize)]
pub struct ClientSecret {
    /// The ephemeral key value
    pub value: String,
    /// Unix timestamp when the key expires (typically 1 minute)
    pub expires_at: u64,
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
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_format_serialization() {
        let pcm = AudioFormat::pcm();
        let json = serde_json::to_string(&pcm).unwrap();
        assert!(json.contains("\"type\":\"audio/pcm\""));
        assert!(json.contains("\"rate\":24000"));

        let pcmu = AudioFormat::pcmu();
        let json = serde_json::to_string(&pcmu).unwrap();
        assert_eq!(json, "{\"type\":\"audio/pcmu\"}");

        let pcma = AudioFormat::pcma();
        let json = serde_json::to_string(&pcma).unwrap();
        assert_eq!(json, "{\"type\":\"audio/pcma\"}");
    }

    #[test]
    fn test_audio_format_deserialization() {
        let json = r#"{"type":"audio/pcm","rate":24000}"#;
        let format: AudioFormat = serde_json::from_str(json).unwrap();
        assert!(matches!(format, AudioFormat::Pcm { rate: 24000 }));

        let json = r#"{"type":"audio/pcmu"}"#;
        let format: AudioFormat = serde_json::from_str(json).unwrap();
        assert!(matches!(format, AudioFormat::Pcmu));
    }

    #[test]
    fn test_voice_serialization() {
        assert_eq!(serde_json::to_string(&Voice::Alloy).unwrap(), "\"alloy\"");
        assert_eq!(serde_json::to_string(&Voice::Marin).unwrap(), "\"marin\"");
        assert_eq!(serde_json::to_string(&Voice::Cedar).unwrap(), "\"cedar\"");
    }

    #[test]
    fn test_voice_option_serialization() {
        let builtin = VoiceOption::BuiltIn(Voice::Alloy);
        assert_eq!(serde_json::to_string(&builtin).unwrap(), "\"alloy\"");

        let custom = VoiceOption::Custom {
            id: "voice_1234".to_string(),
        };
        let json = serde_json::to_string(&custom).unwrap();
        assert!(json.contains("\"id\":\"voice_1234\""));
    }

    #[test]
    fn test_turn_detection_serialization() {
        let vad = TurnDetection::ServerVad {
            threshold: Some(0.5),
            prefix_padding_ms: Some(300),
            silence_duration_ms: Some(500),
            create_response: Some(true),
            interrupt_response: None,
            idle_timeout_ms: None,
        };
        let json = serde_json::to_string(&vad).unwrap();
        assert!(json.contains("\"type\":\"server_vad\""));

        let semantic = TurnDetection::SemanticVad {
            eagerness: Some(SemanticVadEagerness::High),
            create_response: Some(true),
            interrupt_response: Some(true),
        };
        let json = serde_json::to_string(&semantic).unwrap();
        assert!(json.contains("\"type\":\"semantic_vad\""));
        assert!(json.contains("\"eagerness\":\"high\""));

        let disabled = TurnDetection::Disabled;
        let json = serde_json::to_string(&disabled).unwrap();
        assert!(json.contains("\"type\":\"none\""));
    }

    #[test]
    fn test_session_config_default() {
        let config = SessionConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"type\":\"realtime\""));
    }

    #[test]
    fn test_session_config_nested_audio() {
        let config = SessionConfig {
            output_modalities: Some(vec![Modality::Audio]),
            audio: Some(AudioConfig {
                input: Some(AudioInputConfig {
                    format: Some(AudioFormat::pcm()),
                    transcription: Some(InputAudioTranscription {
                        model: Some("gpt-4o-transcribe".to_string()),
                        ..Default::default()
                    }),
                    noise_reduction: Some(InputAudioNoiseReduction::near_field()),
                    turn_detection: Some(TurnDetection::SemanticVad {
                        eagerness: Some(SemanticVadEagerness::Medium),
                        create_response: None,
                        interrupt_response: None,
                    }),
                }),
                output: Some(AudioOutputConfig {
                    format: Some(AudioFormat::pcm()),
                    voice: Some(VoiceOption::BuiltIn(Voice::Marin)),
                    speed: Some(1.0),
                }),
            }),
            ..Default::default()
        };
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"output_modalities\":[\"audio\"]"));
        assert!(json.contains("\"audio\":{"));
        assert!(json.contains("\"input\":{"));
        assert!(json.contains("\"output\":{"));
        assert!(json.contains("\"voice\":\"marin\""));
        assert!(json.contains("\"speed\":1.0"));
    }

    #[test]
    fn test_tool_choice_variants() {
        let auto = ToolChoice::auto();
        assert_eq!(serde_json::to_string(&auto).unwrap(), "\"auto\"");

        let func = ToolChoice::function("my_tool");
        let json = serde_json::to_string(&func).unwrap();
        assert!(json.contains("\"name\":\"my_tool\""));

        let mcp = ToolChoice::mcp("my_server", "my_tool");
        let json = serde_json::to_string(&mcp).unwrap();
        assert!(json.contains("\"server_label\":\"my_server\""));
        assert!(json.contains("\"name\":\"my_tool\""));
    }

    #[test]
    fn test_truncation_serialization() {
        let auto = Truncation::Mode(TruncationMode::Auto);
        assert_eq!(serde_json::to_string(&auto).unwrap(), "\"auto\"");

        let disabled = Truncation::Mode(TruncationMode::Disabled);
        assert_eq!(serde_json::to_string(&disabled).unwrap(), "\"disabled\"");

        let retention = Truncation::RetentionRatio {
            truncation_type: "retention_ratio".to_string(),
            retention_ratio: 0.8,
            token_limits: None,
        };
        let json = serde_json::to_string(&retention).unwrap();
        assert!(json.contains("\"type\":\"retention_ratio\""));
        assert!(json.contains("\"retention_ratio\":0.8"));
    }

    #[test]
    fn test_tracing_serialization() {
        let auto = TracingConfig::Auto(TracingMode::Auto);
        assert_eq!(serde_json::to_string(&auto).unwrap(), "\"auto\"");

        let custom = TracingConfig::Custom {
            workflow_name: Some("my_workflow".to_string()),
            group_id: None,
            metadata: None,
        };
        let json = serde_json::to_string(&custom).unwrap();
        assert!(json.contains("\"workflow_name\":\"my_workflow\""));
    }
}
