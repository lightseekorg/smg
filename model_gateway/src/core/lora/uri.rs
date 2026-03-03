/// Classification of a `lora_path` value supplied by the client.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdapterUri {
    /// Path already on local disk — skip download, load directly into engine.
    LocalPath(String),
    /// Adapter name that the engine already knows — pass through unchanged.
    AlreadyResolved(String),
}

/// Error returned when a URI scheme is syntactically valid but not yet supported.
///
/// v1 only supports local filesystem paths (starting with `/`) and pre-loaded
/// adapter names. Remote URI schemes (`s3://`, `hermes://`, etc.) are reserved
/// for a future download layer.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error(
    "unsupported URI scheme '{scheme}' in lora_path — \
     v1 only supports local paths (starting with '/') or pre-loaded adapter names; \
     remote URI downloads (s3://, gs://, hermes://, etc.) are not yet implemented"
)]
pub struct UnsupportedSchemeError {
    pub scheme: String,
}

/// Well-known URI prefixes that are reserved for future download support.
/// Encountering these returns a clear error rather than silently passing through.
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

/// Classify a raw `lora_path` string.
///
/// Rules (checked in order):
/// 1. Starts with `/`             → local filesystem path  
/// 2. Starts with a known remote scheme → error (not yet supported)
/// 3. Everything else             → treat as an already-loaded adapter name
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

    Ok(AdapterUri::AlreadyResolved(lora_path.to_string()))
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
    fn test_already_resolved() {
        assert_eq!(
            classify("my-lora_a3f9c2eb"),
            Ok(AdapterUri::AlreadyResolved("my-lora_a3f9c2eb".to_string()))
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
