use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use validator::Validate;

use super::common::{default_true, GenerationRequest, StringOrArray, UsageInfo};

fn default_rerank_object() -> String {
    "rerank".to_string()
}

/// TODO: Create timestamp should not be in protocol layer
fn current_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_else(|_| std::time::Duration::from_secs(0))
        .as_secs() as i64
}

// ============================================================================
// Rerank API
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize, Validate, schemars::JsonSchema)]
#[validate(schema(function = "validate_rerank_request"))]
pub struct RerankRequest {
    /// The query text to rank documents against
    #[validate(custom(function = "validate_query"))]
    pub query: String,

    /// List of documents to be ranked
    #[validate(custom(function = "validate_documents"))]
    pub documents: Vec<String>,

    /// Model to use for reranking
    pub model: String,

    /// Maximum number of documents to return (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = 1))]
    pub top_k: Option<usize>,

    /// Whether to return documents in addition to scores
    #[serde(default = "default_true")]
    pub return_documents: bool,

    // SGLang specific extensions
    /// Request ID for tracking
    pub rid: Option<StringOrArray>,

    /// User identifier
    pub user: Option<String>,
}

impl GenerationRequest for RerankRequest {
    fn get_model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn is_stream(&self) -> bool {
        false // Reranking doesn't support streaming
    }

    fn extract_text_for_routing(&self) -> String {
        self.query.clone()
    }
}

impl super::validated::Normalizable for RerankRequest {
    // Use default no-op normalization
}

// ============================================================================
// Validation Functions
// ============================================================================

/// Validates that the query is not empty
fn validate_query(query: &str) -> Result<(), validator::ValidationError> {
    if query.trim().is_empty() {
        return Err(validator::ValidationError::new("query cannot be empty"));
    }
    Ok(())
}

/// Validates that the documents list is not empty
fn validate_documents(documents: &[String]) -> Result<(), validator::ValidationError> {
    if documents.is_empty() {
        return Err(validator::ValidationError::new(
            "documents list cannot be empty",
        ));
    }
    Ok(())
}

/// Schema-level validation for cross-field dependencies
#[expect(
    clippy::unnecessary_wraps,
    reason = "validator crate requires Result return type"
)]
fn validate_rerank_request(req: &RerankRequest) -> Result<(), validator::ValidationError> {
    // Validate top_k if specified
    if let Some(k) = req.top_k {
        if k > req.documents.len() {
            // This is allowed but we log a warning
            tracing::warn!(
                "top_k ({}) is greater than number of documents ({})",
                k,
                req.documents.len()
            );
        }
    }
    Ok(())
}

impl RerankRequest {
    /// Get the effective top_k value
    pub fn effective_top_k(&self) -> usize {
        self.top_k.unwrap_or(self.documents.len())
    }
}

/// Individual rerank result
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct RerankResult {
    /// Relevance score for the document
    pub score: f32,

    /// The document text (if return_documents was true)
    pub document: Option<String>,

    /// Original index of the document in the request
    pub index: usize,

    /// Additional metadata about the ranking
    pub meta_info: Option<HashMap<String, Value>>,
}

/// Rerank response containing sorted results
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct RerankResponse {
    /// Ranked results sorted by score (highest first)
    pub results: Vec<RerankResult>,

    /// Model used for reranking
    pub model: String,

    /// Usage information
    pub usage: Option<UsageInfo>,

    /// Response object type
    #[serde(default = "default_rerank_object")]
    pub object: String,

    /// Response ID
    pub id: Option<StringOrArray>,

    /// Creation timestamp
    pub created: i64,
}

impl RerankResponse {
    /// Create a new RerankResponse with the given results and model
    pub fn new(
        results: Vec<RerankResult>,
        model: String,
        request_id: Option<StringOrArray>,
    ) -> Self {
        RerankResponse {
            results,
            model,
            usage: None,
            object: default_rerank_object(),
            id: request_id,
            created: current_timestamp(),
        }
    }

    /// Apply top_k limit to results
    pub fn apply_top_k(&mut self, k: usize) {
        self.results.truncate(k);
    }

    /// Drop documents from results (when return_documents is false)
    pub fn drop_documents(&mut self) {
        for result in &mut self.results {
            result.document = None;
        }
    }
}

/// V1 API compatibility format for rerank requests
/// Matches Python's V1RerankReqInput
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct V1RerankReqInput {
    pub query: String,
    pub documents: Vec<String>,
}

/// Convert V1RerankReqInput to RerankRequest
impl From<V1RerankReqInput> for RerankRequest {
    fn from(v1: V1RerankReqInput) -> Self {
        RerankRequest {
            query: v1.query,
            documents: v1.documents,
            model: super::UNKNOWN_MODEL_ID.to_string(),
            top_k: None,
            return_documents: true,
            rid: None,
            user: None,
        }
    }
}

// ============================================================================
// Score API (vLLM /v1/score)
// ============================================================================

/// vLLM-compatible score request for cross-encoder reranker models.
///
/// Matches the vLLM `/v1/score` request schema which uses `text_1`/`text_2`
/// pairs rather than the classic `query`/`documents` style.
///
/// # Example
/// ```json
/// {
///   "model": "modernbert-reranker",
///   "text_1": "What is the capital of France?",
///   "text_2": ["Paris is the capital.", "London is in England."]
/// }
/// ```
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct ScoreRequest {
    /// The model to use for scoring
    pub model: String,

    /// The query/source text (single string)
    pub text_1: String,

    /// The document(s) to score against the query.
    /// Can be a single string or a list of strings.
    pub text_2: StringOrVec,

    /// Optional encoding format for the response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,

    /// Whether to truncate the input
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncate_prompt_tokens: Option<u32>,
}

impl ScoreRequest {
    /// Return text_2 as a slice of string references for routing/hashing.
    pub fn texts(&self) -> Vec<&str> {
        match &self.text_2 {
            StringOrVec::Single(s) => vec![s.as_str()],
            StringOrVec::Array(v) => v.iter().map(String::as_str).collect(),
        }
    }
}

impl GenerationRequest for ScoreRequest {
    fn get_model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn is_stream(&self) -> bool {
        false // Score endpoint never streams
    }

    fn extract_text_for_routing(&self) -> String {
        self.text_1.clone()
    }
}

/// `text_2` field: either a single string or an array.
///
/// vLLM accepts both forms; we deserialize and normalize internally.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum StringOrVec {
    Single(String),
    Array(Vec<String>),
}

impl StringOrVec {
    /// Convert into an owned `Vec<String>` regardless of variant.
    pub fn into_vec(self) -> Vec<String> {
        match self {
            Self::Single(s) => vec![s],
            Self::Array(v) => v,
        }
    }

    /// Return the number of texts.
    pub fn len(&self) -> usize {
        match self {
            Self::Single(_) => 1,
            Self::Array(v) => v.len(),
        }
    }

    /// Return true if empty.
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Single(_) => false,
            Self::Array(v) => v.is_empty(),
        }
    }
}

/// An individual score result from the vLLM score API.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ScoreData {
    /// Always `"score"` (vLLM compat)
    pub object: String,
    /// The relevance score as a float
    pub score: f64,
    /// 0-based index of this text in `text_2`
    pub index: usize,
}

/// Response from the vLLM `/v1/score` endpoint.
///
/// Mirrors the structure returned by vLLM's `ScoringResponse`.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ScoreResponse {
    /// Unique identifier for this score response
    pub id: String,
    /// Always `"list"`
    pub object: String,
    /// Unix timestamp (seconds) when the response was created
    pub created: i64,
    /// The scored results, one per input in `text_2`
    pub data: Vec<ScoreData>,
    /// The model that produced the scores
    pub model: String,
    /// Usage information (if provided by backend)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<UsageInfo>,
}
