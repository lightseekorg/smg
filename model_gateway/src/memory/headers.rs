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
pub struct MemoryHeaderView {
    pub policy: Option<String>,
    pub ltm_store_enabled: Option<String>,
    pub subject_id: Option<String>,
    pub recall_method: Option<String>,
    pub embedding_model: Option<String>,
    pub extraction_model: Option<String>,
}

impl MemoryHeaderView {
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

fn extract_header(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}
