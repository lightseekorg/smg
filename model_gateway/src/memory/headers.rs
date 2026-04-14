use axum::http::HeaderMap;

pub use crate::routers::common::header_utils::{
    MEMORY_EMBEDDING_MODEL_HEADER, MEMORY_EXTRACTION_MODEL_HEADER, MEMORY_LTM_STORE_ENABLED_HEADER,
    MEMORY_POLICY_HEADER, MEMORY_RECALL_METHOD_HEADER, MEMORY_SUBJECT_ID_HEADER,
};

/// Normalized view of memory-related request headers.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MemoryHeaderView {
    pub policy: Option<String>,
    pub ltm_store_enabled: Option<String>,
    pub subject_id: Option<String>,
    pub recall_method: Option<String>,
    pub embedding_model: Option<String>,
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
