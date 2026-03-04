use std::sync::Arc;

use dashmap::DashMap;
use reqwest::Client;
use serde_json::Value;
use tokio::sync::Mutex;
use tracing::info;

use crate::config::types::{LoraBackend, LoraConfig};

use openai_protocol::lora::StorageSpec;

use super::{
    engine_client::{EngineClientError, LoraEngineClient},
    uri::{classify, AdapterUri, UnsupportedSchemeError},
};

/// Errors produced by [`WorkerLoraState::resolve`].
///
/// The error kind determines the HTTP status code the router should return:
/// - [`LoraStateError::UnsupportedUri`] → 400 Bad Request (client mistake)
/// - [`LoraStateError::Engine`]         → 502/500 (server-side failure)
#[derive(Debug, thiserror::Error)]
pub enum LoraStateError {
    #[error("unsupported adapter URI: {0}")]
    UnsupportedUri(#[from] UnsupportedSchemeError),
    #[error("engine load/unload failed: {0}")]
    Engine(#[from] EngineClientError),
}

/// Per-worker LoRA adapter lifecycle state.
///
/// Lives on [`BasicWorker`] — every worker owns its own state, so the cache
/// key is simply `local_path → adapter_name` (no `worker_url` dimension needed).
///
/// Responsibilities:
/// - Detect and reject unsupported URI schemes (s3://, hermes://, …).
/// - On first use of a local-path adapter, call the engine's load endpoint.
/// - Cache the loaded adapter name so subsequent requests hit the fast path.
/// - Rewrite the request JSON for the target backend before forwarding.
pub struct WorkerLoraState {
    engine_client: LoraEngineClient,
    backend: LoraBackend,
    worker_url: String,

    /// local_path → adapter_name  (lock-free reads via DashMap sharding)
    loaded: DashMap<String, String>,

    /// Per-path Mutex serialises concurrent first-loads of the same adapter.
    inflight: DashMap<String, Arc<Mutex<()>>>,
}

impl WorkerLoraState {
    /// Create a new state bound to a specific worker URL.
    pub fn new(cfg: &LoraConfig, http: Client, worker_url: impl Into<String>) -> Self {
        let engine_client = LoraEngineClient::new(http, cfg.backend.clone());
        Self {
            engine_client,
            backend: cfg.backend.clone(),
            worker_url: worker_url.into(),
            loaded: DashMap::new(),
            inflight: DashMap::new(),
        }
    }

    /// Resolve and potentially load the LoRA adapter described by `spec`,
    /// then rewrite the relevant fields in `json` for the target backend.
    ///
    /// - Returns [`LoraStateError::UnsupportedUri`] for remote URI schemes
    ///   (s3://, hermes://, …) that are not yet supported.
    /// - Returns [`LoraStateError::Engine`] when the engine rejects the load.
    pub async fn resolve(
        &self,
        spec: &StorageSpec,
        json: &mut Value,
    ) -> Result<(), LoraStateError> {
        let lora_path = &spec.path;

        // Classify the URI — returns Err for unsupported remote schemes.
        let classified = classify(lora_path)?;

        // Already-resolved adapter names: no load needed, but still rewrite
        // (e.g., vLLM needs `model = adapter_name`, not a raw `lora_path`).
        if let AdapterUri::AlreadyResolved(ref name) = classified {
            // Use explicit serving ID if provided, otherwise use the name as-is.
            let effective_name = spec.id.as_deref().unwrap_or(name.as_str());
            rewrite_request(json, effective_name, &self.backend);
            return Ok(());
        }

        // Determine the adapter name: explicit id overrides the hash-derived default.
        let adapter_name = spec
            .id
            .clone()
            .unwrap_or_else(|| make_adapter_name(lora_path));

        // Fast path: adapter already loaded on this worker.
        if let Some(cached) = self.loaded.get(lora_path.as_str()) {
            rewrite_request(json, cached.value(), &self.backend);
            return Ok(());
        }

        // Slow path: serialize concurrent first-loads for the same path.
        let mu = self
            .inflight
            .entry(lora_path.to_string())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone();
        let _guard = mu.lock().await;

        // Re-check after acquiring the lock.
        if let Some(cached) = self.loaded.get(lora_path.as_str()) {
            rewrite_request(json, cached.value(), &self.backend);
            return Ok(());
        }

        info!(
            adapter_name,
            local_path = %lora_path,
            worker_url = %self.worker_url,
            "Loading LoRA adapter into engine"
        );

        let load_result = self
            .engine_client
            .load_adapter(&self.worker_url, &adapter_name, lora_path)
            .await;

        // Always clean up the inflight entry — even on failure — so future
        // requests can retry without the map growing unboundedly.
        self.inflight.remove(lora_path.as_str());

        load_result?;

        self.loaded.insert(lora_path.to_string(), adapter_name.clone());
        rewrite_request(json, &adapter_name, &self.backend);

        Ok(())
    }

    /// Explicitly unload a previously loaded adapter from the engine.
    ///
    /// `local_path` must match the `StorageSpec.path` used during load.
    pub async fn unload(&self, local_path: &str) -> Result<(), LoraStateError> {
        if let Some((_, adapter_name)) = self.loaded.remove(local_path) {
            info!(
                adapter_name,
                local_path,
                worker_url = %self.worker_url,
                "Unloading LoRA adapter from engine"
            );
            self.engine_client
                .unload_adapter(&self.worker_url, &adapter_name)
                .await?;
        }
        Ok(())
    }
}

/// Rewrite the request JSON for the target backend.
///
/// | Backend | Change |
/// |---------|--------|
/// | SGLang  | `lora_path` ← adapter_name (keep `model` unchanged) |
/// | vLLM    | `model` ← adapter_name, remove `lora_path` |
fn rewrite_request(json: &mut Value, adapter_name: &str, backend: &LoraBackend) {
    match backend {
        LoraBackend::Sglang => {
            json["lora_path"] = Value::String(adapter_name.to_string());
        }
        LoraBackend::Vllm => {
            json["model"] = Value::String(adapter_name.to_string());
            if let Some(obj) = json.as_object_mut() {
                obj.remove("lora_path");
            }
        }
    }
}

/// Derive a stable, human-readable adapter name from a local path.
///
/// Example: `/tmp/adapters/my-lora` → `my-lora_a3f9c2eb`
fn make_adapter_name(source_path: &str) -> String {
    let slug = source_path.rsplit('/').next().unwrap_or("adapter");
    let slug = slug.split('@').next().unwrap_or(slug);
    let hash = blake3::hash(source_path.as_bytes());
    let short_hash: String = hash.as_bytes()[..4]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    format!("{slug}_{short_hash}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_adapter_name_basic() {
        let name = make_adapter_name("/tmp/adapters/my-lora");
        assert!(name.starts_with("my-lora_"));
        assert_eq!(name.len(), "my-lora_".len() + 8);
    }

    #[test]
    fn test_make_adapter_name_deterministic() {
        assert_eq!(make_adapter_name("/tmp/foo"), make_adapter_name("/tmp/foo"));
    }

    #[test]
    fn test_rewrite_sglang() {
        let mut json = serde_json::json!({"model": "Llama-3.1-8B", "lora_path": "/tmp/lora"});
        rewrite_request(&mut json, "lora_abc12345", &LoraBackend::Sglang);
        assert_eq!(json["lora_path"], "lora_abc12345");
        assert_eq!(json["model"], "Llama-3.1-8B");
    }

    #[test]
    fn test_rewrite_vllm() {
        let mut json = serde_json::json!({"model": "Llama-3.1-8B", "lora_path": "/tmp/lora"});
        rewrite_request(&mut json, "lora_abc12345", &LoraBackend::Vllm);
        assert_eq!(json["model"], "lora_abc12345");
        assert!(json.get("lora_path").is_none());
    }
}
