//! LoRA adapter types for the OpenAI-compatible API.

use serde::{Deserialize, Serialize};

/// Specifies the storage location and identity of a LoRA adapter.
///
/// ## v1 support matrix
/// | `path` format | Status |
/// |---------------|--------|
/// | `/local/path` | ✅ Loaded on first request, cached per worker |
/// | `adapter-name` (no slash) | ✅ Treated as a pre-loaded adapter name |
/// | `s3://...`, `gs://...`, `hermes://...` | 🚧 Planned — returns 400 in v1 |
///
/// ## Wire format
///
/// ```json
/// {
///   "lora_path": {
///     "path": "/models/adapters/finance-v2",
///     "id":   "finance-v2"
///   }
/// }
/// ```
///
/// For simple local-path use cases the field can also be sent as a plain
/// string, which is deserialized as `StorageSpec { path: "…", .. }`:
///
/// ```json
/// { "lora_path": "/models/adapters/finance-v2" }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(from = "StorageSpecInput")]
pub struct StorageSpec {
    /// Storage path.
    ///
    /// | Prefix | Meaning |
    /// |--------|---------|
    /// | `/`    | Local filesystem path |
    /// | none   | Pre-loaded adapter name (passed directly to the engine) |
    /// | `s3://` / `gs://` / `hermes://` | Remote storage — v1: unsupported |
    pub path: String,

    /// Serving-time adapter name.
    ///
    /// When set, SMG uses this name when calling the engine's load API and
    /// when rewriting the inference request.  When absent, a stable name is
    /// derived from the path hash (e.g. `finance-v2_a3f9c2eb`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    /// Storage credentials.
    ///
    /// Intentionally opaque for now — the download layer will interpret this
    /// field when it lands.  Omit entirely to rely on ambient credentials
    /// (IAM role, environment variables, mounted secrets, etc.).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth: Option<serde_json::Value>,
}

impl StorageSpec {
    pub fn from_path(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            id: None,
            auth: None,
        }
    }
}

/// Intermediate type that allows `StorageSpec` to be deserialized from either
/// a plain string or a full object.
#[derive(Deserialize)]
#[serde(untagged)]
enum StorageSpecInput {
    /// Backward-compatible: `"lora_path": "/tmp/my-adapter"`
    Simple(String),
    /// Full spec: `"lora_path": { "path": "...", "id": "...", "auth": {...} }`
    Full {
        path: String,
        #[serde(default)]
        id: Option<String>,
        #[serde(default)]
        auth: Option<serde_json::Value>,
    },
}

impl From<StorageSpecInput> for StorageSpec {
    fn from(input: StorageSpecInput) -> Self {
        match input {
            StorageSpecInput::Simple(path) => Self {
                path,
                id: None,
                auth: None,
            },
            StorageSpecInput::Full { path, id, auth } => Self { path, id, auth },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_plain_string() {
        let json = r#""/tmp/adapters/my-lora""#;
        let spec: StorageSpec = serde_json::from_str(json).unwrap();
        assert_eq!(spec.path, "/tmp/adapters/my-lora");
        assert!(spec.id.is_none());
        assert!(spec.auth.is_none());
    }

    #[test]
    fn test_deserialize_full_object() {
        let json = r#"{"path": "s3://bucket/adapter", "id": "my-adapter"}"#;
        let spec: StorageSpec = serde_json::from_str(json).unwrap();
        assert_eq!(spec.path, "s3://bucket/adapter");
        assert_eq!(spec.id.as_deref(), Some("my-adapter"));
    }

    #[test]
    fn test_deserialize_path_only_object() {
        let json = r#"{"path": "/local/path"}"#;
        let spec: StorageSpec = serde_json::from_str(json).unwrap();
        assert_eq!(spec.path, "/local/path");
        assert!(spec.id.is_none());
    }

    #[test]
    fn test_serialize_omits_none_fields() {
        let spec = StorageSpec::from_path("/tmp/lora");
        let json = serde_json::to_string(&spec).unwrap();
        assert!(!json.contains("id"));
        assert!(!json.contains("auth"));
    }
}
