//! Tokenize and Detokenize API protocol types
//!
//! These types mirror the SGLang Python implementation for compatibility.
//! See: python/sglang/srt/entrypoints/openai/protocol.py

use serde::{Deserialize, Serialize};

use super::UNKNOWN_MODEL_ID;

// ============================================================================
// Tokenize API
// ============================================================================

/// Request schema for the /v1/tokenize endpoint
///
/// Supports both single string and batch tokenization.
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct TokenizeRequest {
    /// Model name for tokenizer selection
    #[serde(default = "default_model_name")]
    pub model: String,

    /// Text(s) to tokenize - can be a single string or array of strings
    pub prompt: StringOrArray,
}

/// Response schema for the /v1/tokenize endpoint
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct TokenizeResponse {
    /// Token IDs - single list for single input, nested list for batch
    pub tokens: TokensResult,

    /// Token count(s) - single int for single input, list for batch
    pub count: CountResult,

    /// Character count(s) of input - single int for single input, list for batch
    pub char_count: CountResult,
}

/// Token IDs result - either single or batch
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum TokensResult {
    Single(Vec<u32>),
    Batch(Vec<Vec<u32>>),
}

/// Count result - either single or batch
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum CountResult {
    Single(i32),
    Batch(Vec<i32>),
}

// ============================================================================
// Detokenize API
// ============================================================================

/// Request schema for the /v1/detokenize endpoint
///
/// Supports both single sequence and batch detokenization.
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct DetokenizeRequest {
    /// Model name for tokenizer selection
    #[serde(default = "default_model_name")]
    pub model: String,

    /// Token IDs to detokenize - single list or batch (list of lists)
    pub tokens: TokensInput,

    /// Whether to skip special tokens (e.g., padding or EOS) during decoding
    #[serde(default = "default_true")]
    pub skip_special_tokens: bool,
}

/// Token input - either single sequence or batch
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum TokensInput {
    /// Single sequence of token IDs
    Single(Vec<u32>),
    /// Batch of token sequences
    Batch(Vec<Vec<u32>>),
}

impl TokensInput {
    /// Check if this is a batch input
    pub fn is_batch(&self) -> bool {
        matches!(self, TokensInput::Batch(_))
    }

    /// Get the sequences (always returns a vec of vecs for uniform processing)
    pub fn sequences(&self) -> Vec<&[u32]> {
        match self {
            TokensInput::Single(seq) => vec![seq.as_slice()],
            TokensInput::Batch(seqs) => seqs.iter().map(|s| s.as_slice()).collect(),
        }
    }
}

/// Response schema for the /v1/detokenize endpoint
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct DetokenizeResponse {
    /// Decoded text - single string for single input, list for batch
    pub text: TextResult,
}

/// Text result - either single or batch
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum TextResult {
    Single(String),
    Batch(Vec<String>),
}

// ============================================================================
// Tokenizer Management API
// ============================================================================

/// Request schema for adding a tokenizer
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct AddTokenizerRequest {
    /// Name to register the tokenizer under
    pub name: String,

    /// Source: either a local path or HuggingFace model ID
    pub source: String,

    /// Optional path to chat template file
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_template_path: Option<String>,
}

/// Response schema for adding a tokenizer (async)
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct AddTokenizerResponse {
    /// Unique identifier for the tokenizer (UUID)
    pub id: String,
    /// Status of the request: "pending", "processing", "completed", "failed"
    pub status: String,
    pub message: String,
    /// Vocabulary size of the loaded tokenizer (only set on completion)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vocab_size: Option<usize>,
}

/// Response schema for listing tokenizers
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ListTokenizersResponse {
    pub tokenizers: Vec<TokenizerInfo>,
}

/// Information about a registered tokenizer
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct TokenizerInfo {
    /// Unique identifier (UUID)
    pub id: String,
    /// User-provided name
    pub name: String,
    /// Source path or HuggingFace model ID
    pub source: String,
    pub vocab_size: usize,
}

/// Request schema for removing a tokenizer
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct RemoveTokenizerRequest {
    /// Name of the tokenizer to remove
    pub name: String,
}

/// Response schema for removing a tokenizer
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct RemoveTokenizerResponse {
    pub success: bool,
    pub message: String,
}

// ============================================================================
// Render API (chat template + tokenization, no generation)
// ============================================================================

/// Request schema for the /v1/chat/completions/render endpoint
///
/// Applies the chat template to messages and tokenizes the result,
/// returning token IDs without performing any generation.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct RenderChatRequest {
    /// ID of the model to use for tokenizer selection
    pub model: String,

    /// A list of messages comprising the conversation
    pub messages: Vec<super::chat::ChatMessage>,

    /// A list of tools the model may call
    #[serde(default)]
    pub tools: Option<Vec<super::common::Tool>>,

    /// Controls which (if any) tool is called by the model
    pub tool_choice: Option<super::common::ToolChoice>,

    /// Custom chat template (path or inline Jinja2 string)
    pub chat_template: Option<String>,

    /// Whether to add a generation prompt after the messages
    #[serde(default = "default_true")]
    pub add_generation_prompt: bool,

    /// Continue generating from final assistant message
    #[serde(default)]
    pub continue_final_message: bool,

    /// Additional template keyword arguments
    pub chat_template_kwargs: Option<std::collections::HashMap<String, serde_json::Value>>,
}

/// Request schema for the /v1/completions/render endpoint
///
/// Tokenizes the prompt text with optional special tokens,
/// returning token IDs without performing any generation.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct RenderCompletionRequest {
    /// ID of the model to use for tokenizer selection
    pub model: String,

    /// The prompt(s) to tokenize - can be a single string or array of strings
    pub prompt: StringOrArray,

    /// Whether to add special tokens (BOS/EOS) during tokenization
    #[serde(default = "default_true")]
    pub add_special_tokens: bool,

    /// Echo back the prompt in addition to the completion
    #[serde(default)]
    pub echo: Option<bool>,

    /// The suffix that comes after a completion of inserted text
    pub suffix: Option<String>,

    /// Number of prompt logprobs to return
    pub prompt_logprobs: Option<i32>,

    /// Cache salt for prompt caching
    pub cache_salt: Option<String>,
}

/// Response schema for the /v1/chat/completions/render and /v1/completions/render endpoints
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct RenderResponse {
    /// The resulting token IDs
    pub token_ids: Vec<u32>,

    /// Number of tokens
    pub count: usize,
}

// ============================================================================
// Helper Types
// ============================================================================

/// String or array of strings (for flexible input)
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
#[schemars(rename = "TokenizeStringOrArray")]
#[serde(untagged)]
pub enum StringOrArray {
    Single(String),
    Array(Vec<String>),
}

impl StringOrArray {
    /// Check if this is a batch (array) input
    pub fn is_batch(&self) -> bool {
        matches!(self, StringOrArray::Array(_))
    }

    /// Get all strings as a slice (converts single to vec)
    pub fn as_strings(&self) -> Vec<&str> {
        match self {
            StringOrArray::Single(s) => vec![s.as_str()],
            StringOrArray::Array(arr) => arr.iter().map(|s| s.as_str()).collect(),
        }
    }
}

// ============================================================================
// Default Functions
// ============================================================================

fn default_model_name() -> String {
    UNKNOWN_MODEL_ID.to_string()
}

fn default_true() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_request_single() {
        let json = r#"{"prompt": "Hello world"}"#;
        let req: TokenizeRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "unknown");
        assert!(matches!(req.prompt, StringOrArray::Single(_)));
    }

    #[test]
    fn test_tokenize_request_batch() {
        let json = r#"{"model": "llama", "prompt": ["Hello", "World"]}"#;
        let req: TokenizeRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "llama");
        assert!(matches!(req.prompt, StringOrArray::Array(_)));
    }

    #[test]
    fn test_detokenize_request_single() {
        let json = r#"{"tokens": [1, 2, 3]}"#;
        let req: DetokenizeRequest = serde_json::from_str(json).unwrap();
        assert!(matches!(req.tokens, TokensInput::Single(_)));
        assert!(req.skip_special_tokens);
    }

    #[test]
    fn test_detokenize_request_batch() {
        let json = r#"{"tokens": [[1, 2], [3, 4, 5]], "skip_special_tokens": false}"#;
        let req: DetokenizeRequest = serde_json::from_str(json).unwrap();
        assert!(matches!(req.tokens, TokensInput::Batch(_)));
        assert!(!req.skip_special_tokens);
    }

    #[test]
    fn test_tokenize_response_single() {
        let resp = TokenizeResponse {
            tokens: TokensResult::Single(vec![1, 2, 3]),
            count: CountResult::Single(3),
            char_count: CountResult::Single(11),
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("[1,2,3]"));
        assert!(json.contains("\"count\":3"));
        assert!(json.contains("\"char_count\":11"));
    }

    #[test]
    fn test_tokenize_response_batch() {
        let resp = TokenizeResponse {
            tokens: TokensResult::Batch(vec![vec![1, 2], vec![3, 4, 5]]),
            count: CountResult::Batch(vec![2, 3]),
            char_count: CountResult::Batch(vec![5, 5]),
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("[[1,2],[3,4,5]]"));
        assert!(json.contains("[2,3]"));
    }
}
