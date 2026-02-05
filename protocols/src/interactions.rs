// Gemini Interactions API types
// https://ai.google.dev/gemini-api/docs/interactions

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use validator::Validate;

use super::common::{default_model, default_true, Function, GenerationRequest};

// ============================================================================
// Interaction Tools
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InteractionTool {
    #[serde(rename = "type")]
    pub r#type: InteractionToolType,

    // Function tool fields (used when type == "function")
    #[serde(flatten)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<Function>,

    // McpServer fields (used when type == "mcp_server")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_tools: Option<AllowedTools>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum InteractionToolType {
    Function,
    GoogleSearch,
    CodeExecution,
    UrlContext,
    McpServer,
}

/// Allowed tools configuration for MCP server
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct AllowedTools {
    /// Tool choice mode: auto, any, none, or validated
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<ToolChoiceMode>,
    /// List of allowed tool names
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<String>>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoiceMode {
    Auto,
    Any,
    None,
    Validated,
}

// ============================================================================
// Generation Config (Gemini-specific)
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_level: Option<ThinkingLevel>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_summaries: Option<ThinkingSummaries>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub speech_config: Option<SpeechConfig>,

    #[serde(skip_serializing_if = "Option::is_none")]
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
    Mode(ToolChoiceMode),
    Config(ToolChoiceConfig),
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct ToolChoiceConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_tools: Option<AllowedTools>
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SpeechConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ImageConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<AspectRatio>,
    #[serde(skip_serializing_if = "Option::is_none")]
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

// ============================================================================
// Input/Output Types
// ============================================================================

/// Input can be Content, array of Content, array of Turn, or string
/// See: https://ai.google.dev/api/interactions-api#request-body
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum InteractionInput {
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
    #[serde(flatten)]
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
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Content {
    /// Text content
    #[serde(rename = "text")]
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        annotations: Option<Vec<Annotation>>,
    },

    /// Image content
    #[serde(rename = "image")]
    Image {
        #[serde(skip_serializing_if = "Option::is_none")]
        data: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        uri: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        mime_type: Option<ImageMimeType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        resolution: Option<ImageResolution>,
    },

    /// Audio content
    #[serde(rename = "audio")]
    Audio {
        #[serde(skip_serializing_if = "Option::is_none")]
        data: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        uri: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        mime_type: Option<AudioMimeType>,
    },

    /// Document content (PDF)
    #[serde(rename = "document")]
    Document {
        #[serde(skip_serializing_if = "Option::is_none")]
        data: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        uri: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        mime_type: Option<DocumentMimeType>,
    },

    /// Video content
    #[serde(rename = "video")]
    Video {
        #[serde(skip_serializing_if = "Option::is_none")]
        data: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        uri: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        mime_type: Option<VideoMimeType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        resolution: Option<ImageResolution>,
    },

    /// Thought content (for extended thinking)
    #[serde(rename = "thought")]
    Thought {
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        summary: Option<String>,
    },

    /// Function call content
    #[serde(rename = "function_call")]
    FunctionCall {
        id: String,
        name: String,
        arguments: Value,
    },

    /// Function result content
    #[serde(rename = "function_result")]
    FunctionResult {
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        call_id: String,
        result: Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },

    /// URL context call content
    #[serde(rename = "url_context_call")]
    UrlContextCall {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        arguments: Option<UrlContextArguments>,
    },

    /// URL context result content
    #[serde(rename = "url_context_result")]
    UrlContextResult {
        #[serde(skip_serializing_if = "Option::is_none")]
        call_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        result: Option<UrlContextResultData>,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },

    /// Google search call content
    #[serde(rename = "google_search_call")]
    GoogleSearchCall {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        arguments: Option<GoogleSearchArguments>,
    },

    /// Google search result content
    #[serde(rename = "google_search_result")]
    GoogleSearchResult {
        #[serde(skip_serializing_if = "Option::is_none")]
        call_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        result: Option<GoogleSearchResultData>,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },

    /// MCP server tool call content
    #[serde(rename = "mcp_server_tool_call")]
    McpServerToolCall {
        id: String,
        name: String,
        server_name: String,
        arguments: Value,
    },

    /// MCP server tool result content
    #[serde(rename = "mcp_server_tool_result")]
    McpServerToolResult {
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        server_name: Option<String>,
        call_id: String,
        result: Value,
    },
}

/// Annotation for text content (citations)
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Annotation {
    /// Start of the attributed segment, measured in bytes
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_index: Option<u32>,
    /// End of the attributed segment, exclusive
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_index: Option<u32>,
    /// Source attributed for a portion of the text (URL, title, or other identifier)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

/// Arguments for URL context call
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct UrlContextArguments {
    /// The URLs to fetch
    #[serde(skip_serializing_if = "Option::is_none")]
    pub urls: Option<Vec<String>>,
}

/// Result data for URL context result
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct UrlContextResultData {
    /// The URL that was fetched
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// The status of the URL retrieval
    #[serde(skip_serializing_if = "Option::is_none")]
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
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GoogleSearchArguments {
    /// Web search queries
    #[serde(skip_serializing_if = "Option::is_none")]
    pub queries: Option<Vec<String>>,
}

/// Result data for Google search result
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GoogleSearchResultData {
    /// URI reference of the search result
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// Title of the search result
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Web content snippet
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rendered_content: Option<String>,
}

/// Image/video resolution options
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ImageResolution {
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
pub enum InteractionStatus {
    Completed,
    InProgress,
    RequiresAction,
    Failed,
    Cancelled
}

// ============================================================================
// Usage Types
// ============================================================================

/// Token count by modality
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModalityTokens {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modality: Option<ResponseModality>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_count: Option<u32>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InteractionUsage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens_by_modality: Option<Vec<ModalityTokens>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_cached_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens_by_modality: Option<Vec<ModalityTokens>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens_by_modality: Option<Vec<ModalityTokens>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_tool_use_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_use_tokens_by_modality: Option<Vec<ModalityTokens>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_thought_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u32>,
}

// ============================================================================
// Request Type
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize, Validate)]
pub struct InteractionsRequest {
    /// Model identifier (e.g., "gemini-2.0-flash")
    #[serde(default = "default_model")]
    pub model: String,

    /// Input content - can be string or array of Content objects
    pub input: InteractionInput,

    /// System instruction for the model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<String>,

    /// Link to prior interaction for stateful conversations
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_interaction_id: Option<String>,

    /// Available tools
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<InteractionTool>>,

    /// Generation configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GenerationConfig>,

    /// Response format for structured outputs
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<Value>,

    /// MIME type for the response (required if response_format is set)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_mime_type: Option<String>,

    /// Response modalities (text, image, audio)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_modalities: Option<Vec<ResponseModality>>,

    /// Whether to stream the response
    #[serde(default)]
    pub stream: bool,

    /// Run request in background (agents only)
    #[serde(default)]
    pub background: bool,

    /// Whether to store the interaction (default: true)
    #[serde(default = "default_true")]
    pub store: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ResponseModality {
    Text,
    Image,
    Audio,
}

impl Default for InteractionsRequest {
    fn default() -> Self {
        Self {
            model: default_model(),
            input: InteractionInput::Text(String::new()),
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
                Content::Text { text, .. } => Some(text.clone()),
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
            InteractionInput::Text(text) => text.clone(),
            InteractionInput::Content(content) => extract_from_content(content).unwrap_or_default(),
            InteractionInput::Contents(contents) => contents
                .iter()
                .filter_map(extract_from_content)
                .collect::<Vec<String>>()
                .join(" "),
            InteractionInput::Turns(turns) => turns
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

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct InteractionsResponse {
    /// Interaction ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    /// Creation timestamp (ISO 8601)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created: Option<String>,

    /// Last update timestamp (ISO 8601)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated: Option<String>,

    /// Role of the interaction
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,

    /// Interaction status
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<InteractionStatus>,

    /// Model used
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// Agent used
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent: Option<String>,

    /// Output content
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outputs: Option<Vec<Content>>,

    /// Usage information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<InteractionUsage>,

    /// Previous interaction ID for conversation threading
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_interaction_id: Option<String>,
}

impl InteractionsResponse {
    /// Check if the interaction is complete
    pub fn is_complete(&self) -> bool {
        matches!(self.status, Some(InteractionStatus::Completed))
    }

    /// Check if the interaction is in progress
    pub fn is_in_progress(&self) -> bool {
        matches!(self.status, Some(InteractionStatus::InProgress))
    }

    /// Check if the interaction failed
    pub fn is_failed(&self) -> bool {
        matches!(self.status, Some(InteractionStatus::Failed))
    }

    /// Check if the interaction requires action (tool execution)
    pub fn requires_action(&self) -> bool {
        matches!(self.status, Some(InteractionStatus::RequiresAction))
    }
}

// ============================================================================
// Query Parameters
// ============================================================================

/// Query parameters for GET /interactions/{id}
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct InteractionsGetParams {
    /// Whether to stream the response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Last event ID for resuming a stream
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_event_id: Option<String>,
    /// API version
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_version: Option<String>,
}

/// Query parameters for DELETE /interactions/{id}
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct InteractionsDeleteParams {
    /// API version
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_version: Option<String>,
}

/// Query parameters for POST /interactions/{id}/cancel
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct InteractionsCancelParams {
    /// API version
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_version: Option<String>,
}

// ============================================================================
// Helper Functions
// ============================================================================

pub fn generate_interaction_id() -> String {
    use rand::RngCore;
    let mut rng = rand::rng();
    let mut bytes = [0u8; 16];
    rng.fill_bytes(&mut bytes);
    let hex_string: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
    format!("int_{}", hex_string)
}
