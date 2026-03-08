//! LoRA adapter protocol types for the OpenAI-compatible API.
//!
//! This module defines how `lora_path` values in inference requests are
//! interpreted.  The runtime loading logic lives in `model_gateway`.

use thiserror::Error;

/// Classification of a `lora_path` value supplied by the client.
///
/// Rules (checked in order):
/// 1. Starts with `/`                  → [`AdapterUri::LocalPath`] — load from disk on first request
/// 2. Starts with a known remote scheme → [`UnsupportedSchemeError`] — clear 400 for the client
/// 3. Anything else                     → [`AdapterUri::AlreadyLoaded`] — pass the name through to the engine
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdapterUri {
    /// Absolute filesystem path.  SMG loads the adapter into the engine on
    /// first use and caches the result per-worker.
    LocalPath(String),
    /// Adapter name the engine already knows (pre-loaded at engine startup or
    /// via a previous load call).  No loading needed — just rewrite the request.
    AlreadyLoaded(String),
}

/// Returned when `lora_path` contains a URI scheme that is syntactically valid
/// but not yet supported (e.g. `s3://`, `hermes://`).
///
/// The caller should surface this as an HTTP 400 so the client knows the path
/// format is wrong for this version, rather than letting it reach the engine
/// and producing a confusing error.
#[derive(Debug, Error, PartialEq, Eq)]
#[error(
    "unsupported URI scheme '{scheme}' in lora_path — \
     v1 only supports local paths (starting with '/') or pre-loaded adapter names; \
     remote URI downloads (s3://, gs://, hermes://, etc.) are not yet implemented"
)]
pub struct UnsupportedSchemeError {
    pub scheme: String,
}

/// Well-known URI prefixes reserved for future remote download support.
const REMOTE_SCHEMES: &[&str] = &[
    "s3://",
    "gs://",
    "gcs://",
    "az://",
    "abfs://",
    "hermes://",
    "http://",
    "https://",
];

/// Classify a raw `lora_path` string into one of the three cases above.
pub fn classify(lora_path: &str) -> Result<AdapterUri, UnsupportedSchemeError> {
    if lora_path.starts_with('/') {
        return Ok(AdapterUri::LocalPath(lora_path.to_string()));
    }

    for &prefix in REMOTE_SCHEMES {
        if lora_path.starts_with(prefix) {
            let scheme = prefix.trim_end_matches("://").to_string();
            return Err(UnsupportedSchemeError { scheme });
        }
    }

    Ok(AdapterUri::AlreadyLoaded(lora_path.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_path() {
        assert_eq!(
            classify("/tmp/adapters/my-lora"),
            Ok(AdapterUri::LocalPath("/tmp/adapters/my-lora".to_string()))
        );
    }

    #[test]
    fn test_already_loaded() {
        assert_eq!(
            classify("my-lora_a3f9c2eb"),
            Ok(AdapterUri::AlreadyLoaded("my-lora_a3f9c2eb".to_string()))
        );
    }

    #[test]
    fn test_s3_returns_error() {
        let err = classify("s3://my-bucket/adapters/my-lora").unwrap_err();
        assert_eq!(err.scheme, "s3");
    }

    #[test]
    fn test_hermes_returns_error() {
        let err = classify("hermes://myorg/my-adapter@rev123").unwrap_err();
        assert_eq!(err.scheme, "hermes");
    }

    #[test]
    fn test_gcs_returns_error() {
        let err = classify("gs://bucket/path").unwrap_err();
        assert_eq!(err.scheme, "gs");
    }

    #[test]
    fn test_https_returns_error() {
        let err = classify("https://example.com/lora.zip").unwrap_err();
        assert_eq!(err.scheme, "https");
    }
}
