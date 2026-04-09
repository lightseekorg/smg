use axum::http::HeaderMap;

// Placeholder header names for memory controls.
// These names are under active discussion and may change in a follow-up PR.
const MEMORY_POLICY_HEADER: &str = "x-smg-memory-policy";
const MEMORY_LTM_STORE_ENABLED_HEADER: &str = "x-smg-memory-ltm-store-enabled";
const MEMORY_SUBJECT_ID_HEADER: &str = "x-smg-memory-subject-id";
const MEMORY_RECALL_METHOD_HEADER: &str = "x-smg-memory-recall-method";
const MEMORY_EMBEDDING_MODEL_HEADER: &str = "x-smg-memory-embedding-model";
const MEMORY_EXTRACTION_MODEL_HEADER: &str = "x-smg-memory-extraction-model";

#[derive(Debug, Clone, Default, PartialEq, Eq)]
/// Normalized view of memory-related request headers.
pub struct MemoryHeaderView {
    /// Raw policy header value (for example `store_only` or `store_and_recall`).
    pub policy: Option<String>,
    /// Header override for store enablement intent.
    pub ltm_store_enabled: Option<String>,
    /// Subject identifier used for memory scoping.
    pub subject_id: Option<String>,
    /// Optional recall strategy hint.
    pub recall_method: Option<String>,
    /// Optional embedding model override for memory operations.
    pub embedding_model: Option<String>,
    /// Optional extraction model override for memory operations.
    pub extraction_model: Option<String>,
}

impl MemoryHeaderView {
    /// Extract memory-related headers from the HTTP request.
    pub fn from_http_headers(headers: &HeaderMap) -> Self {
        Self {
            policy: extract_header(headers, MEMORY_POLICY_HEADER),
            ltm_store_enabled: extract_header(headers, MEMORY_LTM_STORE_ENABLED_HEADER),
            subject_id: extract_header(headers, MEMORY_SUBJECT_ID_HEADER),
            recall_method: extract_header(headers, MEMORY_RECALL_METHOD_HEADER),
            embedding_model: extract_header(headers, MEMORY_EMBEDDING_MODEL_HEADER),
            extraction_model: extract_header(headers, MEMORY_EXTRACTION_MODEL_HEADER),
        }
    }
}

/// Read one header as a trimmed non-empty string.
fn extract_header(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}
