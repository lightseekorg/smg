use reqwest::Client;
use serde_json::json;
use thiserror::Error;

use crate::config::types::LoraBackend;

/// Errors returned by [`LoraEngineClient`].
#[derive(Debug, Error)]
pub enum EngineClientError {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("engine rejected load_adapter: {0}")]
    LoadFailed(String),
    #[error("engine rejected unload_adapter: {0}")]
    UnloadFailed(String),
}

/// Sends load / unload requests to the LLM engine backend.
///
/// Backend differences are encapsulated here; callers always use the same API.
pub struct LoraEngineClient {
    http: Client,
    backend: LoraBackend,
}

impl LoraEngineClient {
    pub fn new(http: Client, backend: LoraBackend) -> Self {
        Self { http, backend }
    }

    /// Ask the engine to load a LoRA adapter.
    ///
    /// - SGLang: `POST <engine>/load_lora_adapter`
    /// - vLLM:   `POST <engine>/v1/load_lora_adapter`
    pub async fn load_adapter(
        &self,
        engine_url: &str,
        adapter_name: &str,
        local_path: &str,
    ) -> Result<(), EngineClientError> {
        let endpoint = match self.backend {
            LoraBackend::Sglang => format!("{engine_url}/load_lora_adapter"),
            LoraBackend::Vllm => format!("{engine_url}/v1/load_lora_adapter"),
        };
        let body = json!({
            "lora_name": adapter_name,
            "lora_path": local_path,
        });

        let resp = self.http.post(&endpoint).json(&body).send().await?;
        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(EngineClientError::LoadFailed(text));
        }
        Ok(())
    }

    /// Ask the engine to unload a previously loaded LoRA adapter.
    ///
    /// - SGLang: `POST <engine>/unload_lora_adapter`
    /// - vLLM:   `POST <engine>/v1/unload_lora_adapter`
    pub async fn unload_adapter(
        &self,
        engine_url: &str,
        adapter_name: &str,
    ) -> Result<(), EngineClientError> {
        let endpoint = match self.backend {
            LoraBackend::Sglang => format!("{engine_url}/unload_lora_adapter"),
            LoraBackend::Vllm => format!("{engine_url}/v1/unload_lora_adapter"),
        };
        let body = json!({ "lora_name": adapter_name });

        let resp = self.http.post(&endpoint).json(&body).send().await?;
        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(EngineClientError::UnloadFailed(text));
        }
        Ok(())
    }
}
