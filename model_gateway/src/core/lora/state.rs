use std::sync::Arc;

use dashmap::DashMap;
use openai_protocol::{
    lora::{classify, AdapterUri, UnsupportedSchemeError},
    worker::RuntimeType,
};
use reqwest::Client;
use serde_json::{json, Value};
use tokio::sync::Mutex;
use tracing::info;

/// Errors produced by [`WorkerLoraState::resolve`].
///
/// - [`LoraStateError::UnsupportedUri`]     → 400 Bad Request (client mistake)
/// - [`LoraStateError::UnsupportedRuntime`] → 500 Internal (misconfiguration)
/// - [`LoraStateError::EngineLoad`]         → 502 / 500 (server-side failure)
#[derive(Debug, thiserror::Error)]
pub enum LoraStateError {
    #[error("{0}")]
    UnsupportedUri(#[from] UnsupportedSchemeError),
    #[error("runtime type {0:?} does not support LoRA load/unload")]
    UnsupportedRuntime(RuntimeType),
    #[error("engine lora load failed: {0}")]
    EngineLoad(String),
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
///
/// The load / unload endpoints are derived from [`RuntimeType`]:
/// - SGLang:         `/load_lora_adapter`,   `/unload_lora_adapter`
/// - vLLM / TRT-LLM: `/v1/load_lora_adapter`, `/v1/unload_lora_adapter`
pub struct WorkerLoraState {
    http: Client,
    runtime_type: RuntimeType,
    worker_url: String,

    /// local_path → adapter_name  (lock-free reads via DashMap sharding)
    loaded: DashMap<String, String>,

    /// Per-path Mutex serialises concurrent first-loads of the same adapter.
    inflight: DashMap<String, Arc<Mutex<()>>>,
}

impl WorkerLoraState {
    /// Create a new state bound to a specific worker URL and runtime type.
    pub fn new(http: Client, runtime_type: RuntimeType, worker_url: impl Into<String>) -> Self {
        Self {
            http,
            runtime_type,
            worker_url: worker_url.into(),
            loaded: DashMap::new(),
            inflight: DashMap::new(),
        }
    }

    /// Resolve and potentially load the LoRA adapter identified by `lora_path`,
    /// then rewrite the relevant fields in `json` for the target backend.
    ///
    /// - Returns [`LoraStateError::UnsupportedUri`]     for remote URI schemes (s3://, hermes://, …).
    /// - Returns [`LoraStateError::UnsupportedRuntime`] for passthrough backends (External).
    /// - Returns [`LoraStateError::EngineLoad`]         when the engine rejects the load.
    pub async fn resolve(&self, lora_path: &str, json: &mut Value) -> Result<(), LoraStateError> {
        // External is a passthrough backend (OpenAI / Anthropic); it has no
        // load/unload endpoint.  Fail fast with a clear server-side error rather
        // than attempting an HTTP call that will always 404.
        if matches!(
            self.runtime_type,
            RuntimeType::External | RuntimeType::Unspecified
        ) {
            return Err(LoraStateError::UnsupportedRuntime(self.runtime_type));
        }

        // Classify the URI:
        // - AlreadyResolved: a pre-loaded adapter name → rewrite and return
        // - LocalPath: needs loading into the engine
        // - Err: unsupported remote scheme (s3://, hermes://, …) → 400
        match classify(lora_path)? {
            AdapterUri::AlreadyLoaded(name) => {
                rewrite_request(json, &name, self.runtime_type);
                return Ok(());
            }
            AdapterUri::LocalPath(_) => {}
        }

        // Fast path: adapter already loaded on this worker.
        if let Some(cached) = self.loaded.get(lora_path) {
            rewrite_request(json, cached.value(), self.runtime_type);
            return Ok(());
        }

        // Slow path: serialize concurrent first-loads for the same path.
        // Every arriving request for a not-yet-loaded adapter obtains (or reuses)
        // a per-path Mutex, then waits for the lock.  The first task to acquire it
        // does the actual load; subsequent tasks find the adapter in `loaded` on
        // the double-check below and return immediately.
        let mu = self
            .inflight
            .entry(lora_path.to_string())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone();
        let _guard = mu.lock().await;

        // Double-check: another task may have loaded the adapter while we waited.
        if let Some(cached) = self.loaded.get(lora_path) {
            rewrite_request(json, cached.value(), self.runtime_type);
            return Ok(());
        }

        // Compute the stable adapter name only when we're actually going to load.
        let adapter_name = make_adapter_name(lora_path);

        info!(
            adapter_name,
            local_path = %lora_path,
            worker_url = %self.worker_url,
            "Loading LoRA adapter into engine"
        );

        let load_result = self.load_adapter(&adapter_name, lora_path).await;

        // On success: insert into `loaded` BEFORE removing from `inflight`.
        //
        // This closes the race window: if another task arrives between the
        // `inflight.remove` and `loaded.insert` it would see neither map
        // populated, create a fresh mutex, and attempt a duplicate load.
        // By inserting first, any concurrent task will hit the `loaded` fast
        // path and return immediately — even before we finish the cleanup.
        if load_result.is_ok() {
            self.loaded
                .insert(lora_path.to_string(), adapter_name.clone());
        }
        self.inflight.remove(lora_path);
        load_result?;

        rewrite_request(json, &adapter_name, self.runtime_type);

        Ok(())
    }

    /// Explicitly unload a previously loaded adapter from the engine.
    ///
    /// `local_path` must match the value used during load.
    ///
    /// # TODO
    /// Not yet wired to a call site.  Intended for a future LRU eviction policy
    /// or graceful worker shutdown hook that unloads adapters to free GPU memory.
    pub async fn unload(&self, local_path: &str) -> Result<(), LoraStateError> {
        if let Some((_, adapter_name)) = self.loaded.remove(local_path) {
            info!(
                adapter_name,
                local_path,
                worker_url = %self.worker_url,
                "Unloading LoRA adapter from engine"
            );
            self.unload_adapter(&adapter_name).await?;
        }
        Ok(())
    }

    // ── Private HTTP helpers ──────────────────────────────────────────────────

    async fn load_adapter(
        &self,
        adapter_name: &str,
        local_path: &str,
    ) -> Result<(), LoraStateError> {
        let endpoint = self.load_endpoint()?;
        let body = json!({
            "lora_name": adapter_name,
            "lora_path": local_path,
        });

        let resp = self
            .http
            .post(&endpoint)
            .json(&body)
            .send()
            .await
            .map_err(|e| LoraStateError::EngineLoad(e.to_string()))?;

        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(LoraStateError::EngineLoad(text));
        }
        Ok(())
    }

    async fn unload_adapter(&self, adapter_name: &str) -> Result<(), LoraStateError> {
        let endpoint = self.unload_endpoint()?;
        let body = json!({ "lora_name": adapter_name });

        let resp = self
            .http
            .post(&endpoint)
            .json(&body)
            .send()
            .await
            .map_err(|e| LoraStateError::EngineLoad(e.to_string()))?;

        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(LoraStateError::EngineLoad(text));
        }
        Ok(())
    }

    fn load_endpoint(&self) -> Result<String, LoraStateError> {
        match self.runtime_type {
            RuntimeType::Vllm | RuntimeType::Trtllm => {
                Ok(format!("{}/v1/load_lora_adapter", self.worker_url))
            }
            RuntimeType::Sglang => Ok(format!("{}/load_lora_adapter", self.worker_url)),
            RuntimeType::External | RuntimeType::Unspecified => {
                Err(LoraStateError::UnsupportedRuntime(self.runtime_type))
            }
        }
    }

    fn unload_endpoint(&self) -> Result<String, LoraStateError> {
        match self.runtime_type {
            RuntimeType::Vllm | RuntimeType::Trtllm => {
                Ok(format!("{}/v1/unload_lora_adapter", self.worker_url))
            }
            RuntimeType::Sglang => Ok(format!("{}/unload_lora_adapter", self.worker_url)),
            RuntimeType::External | RuntimeType::Unspecified => {
                Err(LoraStateError::UnsupportedRuntime(self.runtime_type))
            }
        }
    }
}

/// Rewrite the request JSON for the target runtime.
///
/// | Runtime        | Change |
/// |----------------|--------|
/// | SGLang         | `lora_path` ← adapter_name (keep `model` unchanged) |
/// | vLLM / TRT-LLM | `model` ← adapter_name, remove `lora_path` |
fn rewrite_request(json: &mut Value, adapter_name: &str, runtime: RuntimeType) {
    match runtime {
        RuntimeType::Vllm | RuntimeType::Trtllm => {
            json["model"] = Value::String(adapter_name.to_string());
            if let Some(obj) = json.as_object_mut() {
                obj.remove("lora_path");
            }
        }
        RuntimeType::Sglang => {
            json["lora_path"] = Value::String(adapter_name.to_string());
        }
        // Guarded upstream in resolve(); these branches are never reached in practice.
        RuntimeType::External | RuntimeType::Unspecified => {}
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
        rewrite_request(&mut json, "lora_abc12345", RuntimeType::Sglang);
        assert_eq!(json["lora_path"], "lora_abc12345");
        assert_eq!(json["model"], "Llama-3.1-8B");
    }

    #[test]
    fn test_rewrite_vllm() {
        let mut json = serde_json::json!({"model": "Llama-3.1-8B", "lora_path": "/tmp/lora"});
        rewrite_request(&mut json, "lora_abc12345", RuntimeType::Vllm);
        assert_eq!(json["model"], "lora_abc12345");
        assert!(json.get("lora_path").is_none());
    }
}
