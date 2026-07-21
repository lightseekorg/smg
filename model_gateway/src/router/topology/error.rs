use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum TopologyError {
    #[error("No text content found in request")]
    MissingContent,

    #[error("Invalid JSON structure: {0}")]
    InvalidJson(String),

    #[error("Consistent hash ring is empty")]
    EmptyRing,

    #[error("Worker not found: {0}")]
    WorkerNotFound(String),

    #[error("Invalid UTF-8 sequence")]
    InvalidUtf8,

    #[error("Request is empty")]
    EmptyRequest,

    #[error("Cache capacity exceeded")]
    CacheFull,
}
