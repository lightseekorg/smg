// Gemini Interactions API types
// https://ai.google.dev/gemini-api/docs/interactions

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use serde_with::skip_serializing_none;
use validator::{Validate, ValidationError};

use super::common::{default_model, default_true, Function, GenerationRequest};

// ============================================================================
// Request Type
// ============================================================================

#[skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, Validate)]
#[validate(schema(function = "validate_interactions_request"))]
pub struct InteractionsRequest {
    /// Model identifier (e.g., "gemini-2.0-flash")
    /// Required if agent is not provided
    #[serde(default = "default_model")]
    pub model: String,

    /// Agent name (e.g., "deep-research-pro-preview-12-2025")
    /// Required if model is not provided
    pub agent: Option<String>,

    /// Input content - can be string or array of Content objects
    pub input: InteractionsInput,

    /// System instruction for the model
    pub system_instruction: Option<String>,

    /// Available tools
    pub tools: Option<Vec<InteractionsTool>>,

    /// Response format for structured outputs
    pub response_format: Option<Value>,

    /// MIME type for the response (required if response_format is set)
    pub response_mime_type: Option<String>,

    /// Whether to stream the response
    #[serde(default)]
    pub stream: bool,

    /// Whether to store the interaction (default: true)
    #[serde(default = "default_true")]
    pub store: bool,

    /// Run request in background (agents only)
    #[serde(default)]
    pub background: bool,

    /// Generation configuration
    pub generation_config: Option<GenerationConfig>,

    /// Agent configuration (only applicable when agent is specified)
    pub agent_config: Option<AgentConfig>,

    /// Response modalities (text, image, audio)
    pub response_modalities: Option<Vec<ResponseModality>>,

    /// Link to prior interaction for stateful conversations
    pub previous_interaction_id: Option<String>,
}

fn validate_interactions_request(req: &InteractionsRequest) -> Result<(), ValidationError> {
    // Either model or agent must be provided
    if (req.model.is_empty() || req.model == default_model()) && req.agent.is_none() {
        return Err(ValidationError::new("model_or_agent_required"));
    }
    // response_mime_type is required when response_format is set
    if req.response_format.is_some() && req.response_mime_type.is_none() {
        return Err(ValidationError::new("response_mime_type_required"));
    }
    Ok(())
}

impl Default for InteractionsRequest {
    fn default() -> Self {
        Self {
            model: default_model(),
            agent: None,
            agent_config: None,
            input: InteractionsInput::Text(String::new()),
            system_instruction: None,
            previous_interaction_id: None,
            tools: None,
            generation_config: None,
            response_format: None,
            response_mime_type: None,
            response_modalities: None,
            stream: false,
            background: false,
            store: true,
        }
    }
}

impl GenerationRequest for InteractionsRequest {
    fn is_stream(&self) -> bool {
        self.stream
    }

    fn get_model(&self) -> Option<&str> {
        Some(self.model.as_str())
    }

    fn extract_text_for_routing(&self) -> String {
        fn extract_from_content(content: &Content) -> Option<String> {
            match content {
                Content::Text { text, .. } => text.clone(),
                _ => None,
            }
        }

        fn extract_from_turn(turn: &Turn) -> String {
            match &turn.content {
                TurnContent::Text(text) => text.clone(),
                TurnContent::Contents(contents) => contents
                    .iter()
                    .filter_map(extract_from_content)
                    .collect::<Vec<String>>()
                    .join(" "),
            }
        }

        match &self.input {
            InteractionsInput::Text(text) => text.clone(),
            InteractionsInput::Content(content) => {
                extract_from_content(content).unwrap_or_default()
            }
            InteractionsInput::Contents(contents) => contents
                .iter()
                .filter_map(extract_from_content)
                .collect::<Vec<String>>()
                .join(" "),
            InteractionsInput::Turns(turns) => turns
                .iter()
                .map(extract_from_turn)
                .collect::<Vec<String>>()
                .join(" "),
        }
    }
}

// ============================================================================
// Response Type
// ============================================================================

#[skip_serializing_none]
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct InteractionsResponse {
    /// Model used
    pub model: Option<String>,

    /// Agent used
    pub agent: Option<String>,

    /// Interaction ID
    pub id: Option<String>,

    /// Interaction status
    pub status: Option<InteractionsStatus>,

    /// Creation timestamp (ISO 8601)
    pub created: Option<String>,

    /// Last update timestamp (ISO 8601)
    pub updated: Option<String>,

    /// Role of the interaction
    pub role: Option<String>,

    /// Output content
    pub outputs: Option<Vec<OutputContent>>,

    /// Usage information
    pub usage: Option<InteractionsUsage>,

    /// Previous interaction ID for conversation threading
    pub previous_interaction_id: Option<String>,
}

impl InteractionsResponse {
    /// Check if the interaction is complete
    pub fn is_complete(&self) -> bool {
        matches!(self.status, Some(InteractionsStatus::Completed))
    }

    /// Check if the interaction is in progress
    pub fn is_in_progress(&self) -> bool {
        matches!(self.status, Some(InteractionsStatus::InProgress))
    }

    /// Check if the interaction failed
    pub fn is_failed(&self) -> bool {
        matches!(self.status, Some(InteractionsStatus::Failed))
    }

    /// Check if the interaction requires action (tool execution)
    pub fn requires_action(&self) -> bool {
        matches!(self.status, Some(InteractionsStatus::RequiresAction))
    }
}

// ============================================================================
// Query Parameters
// ============================================================================

/// Query parameters for GET /interactions/{id}
#[skip_serializing_none]
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct InteractionsGetParams {
    /// Whether to stream the response
    pub stream: Option<bool>,
    /// Last event ID for resuming a stream
    pub last_event_id: Option<String>,
    /// API version
    pub api_version: Option<String>,
}

/// Query parameters for DELETE /interactions/{id}
#[skip_serializing_none]
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct InteractionsDeleteParams {
    /// API version
    pub api_version: Option<String>,
}

/// Query parameters for POST /interactions/{id}/cancel
#[skip_serializing_none]
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct InteractionsCancelParams {
    /// API version
    pub api_version: Option<String>,
}

// ============================================================================
// Interaction Tools
// ============================================================================

/// Interaction tool types
/// See: https://ai.google.dev/api/interactions-api#Resource:Tool
#[skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InteractionsTool {
    /// Function tool with function declaration
    Function(Function),
    /// Google Search built-in tool
    GoogleSearch {},
    /// Code Execution built-in tool
    CodeExecution {},
    /// URL Context built-in tool
    UrlContext {},
    /// MCP Server tool
    McpServer {
        name: Option<String>,
        url: Option<String>,
        headers: Option<HashMap<String, String>>,
        allowed_tools: Option<AllowedTools>,
    },
}

/// Allowed tools configuration for MCP server
#[skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct AllowedTools {
    /// Tool choice mode: auto, any, none, or validated
    pub mode: Option<ToolChoiceType>,
    /// List of allowed tool names
    pub tools: Option<Vec<String>>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoiceType {
    Auto,
    Any,
    None,
    Validated,
}

// ============================================================================
// Generation Config (Gemini-specific)
// ============================================================================

#[skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GenerationConfig {
    pub temperature: Option<f32>,

    pub top_p: Option<f32>,

    pub seed: Option<i64>,

    pub stop_sequences: Option<Vec<String>>,

    pub tool_choice: Option<ToolChoice>,

    pub thinking_level: Option<ThinkingLevel>,

    pub thinking_summaries: Option<ThinkingSummaries>,

    pub max_output_tokens: Option<u32>,

    pub speech_config: Option<SpeechConfig>,

    pub image_config: Option<ImageConfig>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingLevel {
    Minimal,
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingSummaries {
    Auto,
    None,
}

/// Tool choice can be a simple mode or a detailed config
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ToolChoice {
    Type(ToolChoiceType),
    Config(ToolChoiceConfig),
}

#[skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct ToolChoiceConfig {
    pub allowed_tools: Option<AllowedTools>,
}

#[skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SpeechConfig {
    pub voice: Option<String>,
    pub language: Option<String>,
    pub speaker: Option<String>,
}

#[skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ImageConfig {
    pub aspect_ratio: Option<AspectRatio>,
    pub image_size: Option<ImageSize>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub enum AspectRatio {
    #[serde(rename = "1:1")]
    Square,
    #[serde(rename = "2:3")]
    Portrait2x3,
    #[serde(rename = "3:2")]
    Landscape3x2,
    #[serde(rename = "3:4")]
    Portrait3x4,
    #[serde(rename = "4:3")]
    Landscape4x3,
    #[serde(rename = "4:5")]
    Portrait4x5,
    #[serde(rename = "5:4")]
    Landscape5x4,
    #[serde(rename = "9:16")]
    Portrait9x16,
    #[serde(rename = "16:9")]
    Landscape16x9,
    #[serde(rename = "21:9")]
    UltraWide,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub enum ImageSize {
    #[serde(rename = "1K")]
    OneK,
    #[serde(rename = "2K")]
    TwoK,
    #[serde(rename = "4K")]
    FourK,
}

/// Agent configuration
/// See: https://ai.google.dev/api/interactions-api#CreateInteraction-deep_research
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentConfig {
    /// Dynamic agent configuration
    Dynamic {},
    /// Deep Research agent configuration
    DeepResearch {
        /// Whether to include thought summaries ("auto" or "none")
        #[serde(skip_serializing_if = "Option::is_none")]
        thinking_summaries: Option<ThinkingSummaries>,
    },
}

// ============================================================================
// Input/Output Types
// ============================================================================

/// Input can be Content, array of Content, array of Turn, or string
/// See: https://ai.google.dev/api/interactions-api#request-body
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum InteractionsInput {
    /// Simple text input
    Text(String),
    /// Single content block
    Content(Content),
    /// Array of content blocks
    Contents(Vec<Content>),
    /// Array of turns (conversation history)
    Turns(Vec<Turn>),
}

/// A turn in a conversation with role and content
/// See: https://ai.google.dev/api/interactions-api#Resource:Turn
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Turn {
    /// Role: "user" or "model"
    pub role: String,
    /// Content can be array of Content or string
    pub content: TurnContent,
}

/// Turn content can be array of Content or a simple string
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum TurnContent {
    Contents(Vec<Content>),
    Text(String),
}

/// Content is a polymorphic type representing different content types
/// See: https://ai.google.dev/api/interactions-api#Resource:Content
#[skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Content {
    /// Text content
    Text {
        text: Option<String>,
        annotations: Option<Vec<Annotation>>,
    },

    /// Image content
    Image {
        data: Option<String>,
        uri: Option<String>,
        mime_type: Option<ImageMimeType>,
        resolution: Option<MediaResolution>,
    },

    /// Audio content
    Audio {
        data: Option<String>,
        uri: Option<String>,
        mime_type: Option<AudioMimeType>,
    },

    /// Document content (PDF)
    Document {
        data: Option<String>,
        uri: Option<String>,
        mime_type: Option<DocumentMimeType>,
    },

    /// Video content
    Video {
        data: Option<String>,
        uri: Option<String>,
        mime_type: Option<VideoMimeType>,
        resolution: Option<MediaResolution>,
    },

    /// Function call content
    FunctionCall {
        name: String,
        arguments: Value,
        id: String,
    },

    /// Function result content
    FunctionResult {
        name: Option<String>,
        is_error: Option<bool>,
        result: Value,
        call_id: String,
    },

    /// Code execution call content
    CodeExecutionCall {
        arguments: Option<CodeExecutionArguments>,
        id: Option<String>,
    },

    /// Code execution result content
    CodeExecutionResult {
        result: Option<String>,
        is_error: Option<bool>,
        signature: Option<String>,
        call_id: Option<String>,
    },

    /// URL context call content
    UrlContextCall {
        arguments: Option<UrlContextArguments>,
        id: Option<String>,
    },

    /// URL context result content
    UrlContextResult {
        signature: Option<String>,
        result: Option<UrlContextResultData>,
        is_error: Option<bool>,
        call_id: Option<String>,
    },

    /// Google search call content
    GoogleSearchCall {
        arguments: Option<GoogleSearchArguments>,
        id: Option<String>,
    },

    /// Google search result content
    GoogleSearchResult {
        signature: Option<String>,
        result: Option<GoogleSearchResultData>,
        is_error: Option<bool>,
        call_id: Option<String>,
    },

    /// MCP server tool call content
    McpServerToolCall {
        name: String,
        server_name: String,
        arguments: Value,
        id: String,
    },

    /// MCP server tool result content
    McpServerToolResult {
        name: Option<String>,
        server_name: Option<String>,
        result: Value,
        call_id: String,
    },
}

/// Thought content for extended thinking (output only)
/// See: https://ai.google.dev/api/interactions-api#Resource:ThoughtContent
#[skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ThoughtContent {
    /// Type discriminator, always "thought"
    #[serde(rename = "type")]
    pub content_type: ThoughtContentType,
    /// Opaque signature for the thought
    pub signature: Option<String>,
    /// Summary is an array of Content (typically text)
    pub summary: Option<Vec<Content>>,
}

/// Type marker for ThoughtContent
#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum ThoughtContentType {
    #[serde(rename = "thought")]
    Thought,
}

/// Output content can be either regular Content or ThoughtContent
/// Used in InteractionsResponse.outputs
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum OutputContent {
    /// Thought content (for extended thinking)
    Thought(ThoughtContent),
    /// Regular content types
    Content(Content),
}

/// Annotation for text content (citations)
#[skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Annotation {
    /// Start of the attributed segment, measured in bytes
    pub start_index: Option<u32>,
    /// End of the attributed segment, exclusive
    pub end_index: Option<u32>,
    /// Source attributed for a portion of the text (URL, title, or other identifier)
    pub source: Option<String>,
}

/// Arguments for URL context call
#[skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct UrlContextArguments {
    /// The URLs to fetch
    pub urls: Option<Vec<String>>,
}

/// Result data for URL context result
#[skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct UrlContextResultData {
    /// The URL that was fetched
    pub url: Option<String>,
    /// The status of the URL retrieval
    pub status: Option<UrlContextStatus>,
}

/// Status of URL context retrieval
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum UrlContextStatus {
    Success,
    Error,
    Paywall,
    Unsafe,
}

/// Arguments for Google search call
#[skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GoogleSearchArguments {
    /// Web search queries
    pub queries: Option<Vec<String>>,
}

/// Result data for Google search result
#[skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GoogleSearchResultData {
    /// URI reference of the search result
    pub url: Option<String>,
    /// Title of the search result
    pub title: Option<String>,
    /// Web content snippet
    pub rendered_content: Option<String>,
}

/// Arguments for code execution call
#[skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CodeExecutionArguments {
    /// Programming language (currently only Python is supported)
    pub language: Option<CodeExecutionLanguage>,
    /// The code to be executed
    pub code: Option<String>,
}

/// Supported languages for code execution
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum CodeExecutionLanguage {
    Python,
}

/// Image/video resolution options
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum MediaResolution {
    Low,
    Medium,
    High,
    UltraHigh,
}

/// Supported image MIME types
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub enum ImageMimeType {
    #[serde(rename = "image/png")]
    Png,
    #[serde(rename = "image/jpeg")]
    Jpeg,
    #[serde(rename = "image/webp")]
    Webp,
    #[serde(rename = "image/heic")]
    Heic,
    #[serde(rename = "image/heif")]
    Heif,
}

impl ImageMimeType {
    pub fn as_str(&self) -> &'static str {
        match self {
            ImageMimeType::Png => "image/png",
            ImageMimeType::Jpeg => "image/jpeg",
            ImageMimeType::Webp => "image/webp",
            ImageMimeType::Heic => "image/heic",
            ImageMimeType::Heif => "image/heif",
        }
    }
}

/// Supported audio MIME types
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub enum AudioMimeType {
    #[serde(rename = "audio/wav")]
    Wav,
    #[serde(rename = "audio/mp3")]
    Mp3,
    #[serde(rename = "audio/aiff")]
    Aiff,
    #[serde(rename = "audio/aac")]
    Aac,
    #[serde(rename = "audio/ogg")]
    Ogg,
    #[serde(rename = "audio/flac")]
    Flac,
}

impl AudioMimeType {
    pub fn as_str(&self) -> &'static str {
        match self {
            AudioMimeType::Wav => "audio/wav",
            AudioMimeType::Mp3 => "audio/mp3",
            AudioMimeType::Aiff => "audio/aiff",
            AudioMimeType::Aac => "audio/aac",
            AudioMimeType::Ogg => "audio/ogg",
            AudioMimeType::Flac => "audio/flac",
        }
    }
}

/// Supported document MIME types
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub enum DocumentMimeType {
    #[serde(rename = "application/pdf")]
    Pdf,
}

impl DocumentMimeType {
    pub fn as_str(&self) -> &'static str {
        match self {
            DocumentMimeType::Pdf => "application/pdf",
        }
    }
}

/// Supported video MIME types
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub enum VideoMimeType {
    #[serde(rename = "video/mp4")]
    Mp4,
    #[serde(rename = "video/mpeg")]
    Mpeg,
    #[serde(rename = "video/mov")]
    Mov,
    #[serde(rename = "video/avi")]
    Avi,
    #[serde(rename = "video/x-flv")]
    Flv,
    #[serde(rename = "video/mpg")]
    Mpg,
    #[serde(rename = "video/webm")]
    Webm,
    #[serde(rename = "video/wmv")]
    Wmv,
    #[serde(rename = "video/3gpp")]
    ThreeGpp,
}

impl VideoMimeType {
    pub fn as_str(&self) -> &'static str {
        match self {
            VideoMimeType::Mp4 => "video/mp4",
            VideoMimeType::Mpeg => "video/mpeg",
            VideoMimeType::Mov => "video/mov",
            VideoMimeType::Avi => "video/avi",
            VideoMimeType::Flv => "video/x-flv",
            VideoMimeType::Mpg => "video/mpg",
            VideoMimeType::Webm => "video/webm",
            VideoMimeType::Wmv => "video/wmv",
            VideoMimeType::ThreeGpp => "video/3gpp",
        }
    }
}

// ============================================================================
// Status Types
// ============================================================================

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InteractionsStatus {
    Completed,
    InProgress,
    RequiresAction,
    Failed,
    Cancelled,
}

// ============================================================================
// Usage Types
// ============================================================================

/// Token count by modality
#[skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModalityTokens {
    pub modality: Option<ResponseModality>,
    pub tokens: Option<u32>,
}

#[skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InteractionsUsage {
    pub total_input_tokens: Option<u32>,
    pub input_tokens_by_modality: Option<Vec<ModalityTokens>>,
    pub total_cached_tokens: Option<u32>,
    pub cached_tokens_by_modality: Option<Vec<ModalityTokens>>,
    pub total_output_tokens: Option<u32>,
    pub output_tokens_by_modality: Option<Vec<ModalityTokens>>,
    pub total_tool_use_tokens: Option<u32>,
    pub tool_use_tokens_by_modality: Option<Vec<ModalityTokens>>,
    pub total_thought_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ResponseModality {
    Text,
    Image,
    Audio,
}
