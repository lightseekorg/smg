use std::sync::Arc;

use dashmap::DashMap;
use reqwest::Client;
use serde_json::Value;
use thiserror::Error;
use tokio::sync::Mutex;
use tracing::info;

use crate::config::types::{LoraBackend, LoraConfig};

use super::{
    engine_client::{EngineClientError, LoraEngineClient},
    uri::{classify, AdapterUri, UnsupportedSchemeError},
};

/// Errors returned by [`LoraMiddleware::resolve`].
#[derive(Debug, Error)]
pub enum LoraError {
    #[error("engine client error: {0}")]
    Engine(#[from] EngineClientError),
    #[error("unsupported adapter URI: {0}")]
    UnsupportedUri(#[from] UnsupportedSchemeError),
}

/// Central coordinator for LoRA adapter lifecycle.
///
/// Responsibilities:
/// - Accept a raw `lora_path` value from the JSON request body.
/// - If the adapter is a local path: load it into the engine (once per worker, idempotent).
/// - Rewrite the request JSON so the engine sees the correct field name / value.
/// - Cache loaded adapters in memory, keyed by `(worker_url, local_path)`, so that
///   each worker's state is tracked independently.
pub struct LoraMiddleware {
    engine_client: LoraEngineClient,
    backend: LoraBackend,

    /// (worker_url, local_path) → adapter_name
    ///
    /// Keyed per-worker so that loading an adapter into worker A does not
    /// mask the fact that worker B has never received the same load call.
    loaded: DashMap<(String, String), String>,

    /// Per-(worker, path) Mutex prevents duplicate concurrent first-loads.
    inflight: DashMap<(String, String), Arc<Mutex<()>>>,
}

impl LoraMiddleware {
    pub fn new(cfg: &LoraConfig, http: Client) -> Self {
        let engine_client = LoraEngineClient::new(http, cfg.backend.clone());
        Self {
            engine_client,
            backend: cfg.backend.clone(),
            loaded: DashMap::new(),
            inflight: DashMap::new(),
        }
    }

    /// Resolve and potentially load the LoRA adapter referenced in `json`,
    /// then rewrite the relevant fields for the target backend.
    ///
    /// No-ops when `lora_path` is absent from `json`.
    /// Returns `Err(LoraError::UnsupportedUri)` for remote URI schemes that
    /// are not yet supported (s3://, hermes://, etc.).
    pub async fn resolve(
        &self,
        json: &mut Value,
        worker_url: &str,
    ) -> Result<(), LoraError> {
        let lora_path = match json.get("lora_path").and_then(|v| v.as_str()) {
            Some(p) => p.to_string(),
            None => return Ok(()), // not a LoRA request — fast path
        };

        // Classify the URI — returns Err for unsupported remote schemes.
        let classified = classify(&lora_path)?;

        // Already-resolved adapter names don't need loading but still need JSON
        // rewriting for backends that use different field names
        // (e.g., vLLM uses `model`, not `lora_path`).
        if let AdapterUri::AlreadyResolved(ref name) = classified {
            rewrite_request(json, name, &self.backend);
            return Ok(());
        }

        let cache_key = (worker_url.to_string(), lora_path.clone());

        // Fast path: already loaded on this specific worker (lock-free DashMap read).
        if let Some(name) = self.loaded.get(&cache_key) {
            rewrite_request(json, name.value(), &self.backend);
            return Ok(());
        }

        // Slow path: serialize concurrent first-loads for the same (worker, path).
        let mu = self
            .inflight
            .entry(cache_key.clone())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone();
        let _guard = mu.lock().await;

        // Re-check after acquiring lock — another task may have finished.
        if let Some(name) = self.loaded.get(&cache_key) {
            rewrite_request(json, name.value(), &self.backend);
            return Ok(());
        }

        // Attempt to load the adapter into the engine.
        let adapter_name = make_adapter_name(&lora_path);
        info!(
            adapter_name,
            local_path = %lora_path,
            worker_url,
            "Loading LoRA adapter into engine"
        );

        let load_result = self
            .engine_client
            .load_adapter(worker_url, &adapter_name, &lora_path)
            .await;

        // Always remove the inflight entry — even on failure — so future
        // requests can retry without growing the map unboundedly.
        self.inflight.remove(&cache_key);

        // Propagate error after cleanup.
        load_result?;

        // Cache the successful load keyed by (worker, path).
        self.loaded.insert(cache_key, adapter_name.clone());
        rewrite_request(json, &adapter_name, &self.backend);

        Ok(())
    }

    /// Unload a previously loaded adapter from the engine.
    ///
    /// This is an explicit management operation; it is *not* called automatically
    /// after every inference request (no per-request unload in v1).
    pub async fn unload(
        &self,
        local_path: &str,
        worker_url: &str,
    ) -> Result<(), LoraError> {
        let cache_key = (worker_url.to_string(), local_path.to_string());
        if let Some((_, adapter_name)) = self.loaded.remove(&cache_key) {
            info!(
                adapter_name,
                local_path,
                worker_url,
                "Unloading LoRA adapter from engine"
            );
            self.engine_client
                .unload_adapter(worker_url, &adapter_name)
                .await?;
        }
        Ok(())
    }
}

/// Rewrite the request JSON body so the engine receives the correct fields.
///
/// | Backend | What changes |
/// |---------|-------------|
/// | SGLang  | `lora_path` → `adapter_name`, `model` unchanged |
/// | vLLM    | `model` → `adapter_name`, `lora_path` removed |
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
    // Use first 4 bytes (8 hex chars) as a short, stable identifier.
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
    fn test_make_adapter_name_with_revision() {
        let name = make_adapter_name("/tmp/adapters/my-lora@abc123");
        assert!(name.starts_with("my-lora_"));
    }

    #[test]
    fn test_make_adapter_name_deterministic() {
        let a = make_adapter_name("/tmp/foo");
        let b = make_adapter_name("/tmp/foo");
        assert_eq!(a, b);
    }

    #[test]
    fn test_rewrite_sglang() {
        let mut json = serde_json::json!({
            "model": "Llama-3.1-8B",
            "lora_path": "/tmp/my-lora",
        });
        rewrite_request(&mut json, "my-lora_abcd1234", &LoraBackend::Sglang);
        assert_eq!(json["lora_path"], "my-lora_abcd1234");
        assert_eq!(json["model"], "Llama-3.1-8B");
    }

    #[test]
    fn test_rewrite_vllm() {
        let mut json = serde_json::json!({
            "model": "Llama-3.1-8B",
            "lora_path": "/tmp/my-lora",
        });
        rewrite_request(&mut json, "my-lora_abcd1234", &LoraBackend::Vllm);
        assert_eq!(json["model"], "my-lora_abcd1234");
        assert!(json.get("lora_path").is_none());
    }
}
