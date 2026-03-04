//! Workflow step that loads LoRA adapters into the engine at worker startup.

use async_trait::async_trait;
use openai_protocol::worker::LoraStorage;
use serde_json::json;
use tracing::{info, warn};
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowResult};

use crate::core::{steps::workflow_data::LocalWorkerWorkflowData, worker::RuntimeType};

/// Load declared LoRA adapters into the engine after the worker is registered.
///
/// For each [`LoraSpec`] in `WorkerSpec.loras`:
/// 1. Resolve the local path (download for remote storage — v1: remote
///    variants log a warning and are skipped until download support lands).
/// 2. Call the engine's load endpoint (derived from `runtime_type`):
///    - SGLang / unknown: `POST {url}/load_lora_adapter`
///    - vLLM / TRT-LLM:  `POST {url}/v1/load_lora_adapter`
///
/// This step runs **before** activation so adapters are ready before the
/// worker starts serving requests.  Individual adapter failures are non-fatal
/// and are logged as warnings — the worker still activates.
pub struct LoadLoraAdaptersStep;

#[async_trait]
impl StepExecutor<LocalWorkerWorkflowData> for LoadLoraAdaptersStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<LocalWorkerWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let loras = &context.data.config.loras;

        if loras.is_empty() {
            return Ok(StepResult::Continue);
        }

        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| wfaas::WorkflowError::ContextValueNotFound("app_context".to_string()))?;

        // Derive load endpoint from the detected runtime type.
        let runtime = context
            .data
            .detected_runtime_type
            .as_deref()
            .and_then(|s| s.parse::<RuntimeType>().ok())
            .unwrap_or_default();

        let base_url = super::http_base_url(&context.data.config.url);

        let load_endpoint = match runtime {
            RuntimeType::Vllm | RuntimeType::Trtllm => {
                format!("{base_url}/v1/load_lora_adapter")
            }
            _ => {
                // SGLang and unknown runtimes use the non-versioned path.
                format!("{base_url}/load_lora_adapter")
            }
        };

        for lora in loras {
            // --- Step 1: Resolve local path -----------------------------------
            let local_path = match resolve_local_path(&lora.storage) {
                Ok(p) => p,
                Err(reason) => {
                    warn!(
                        lora_id = %lora.id,
                        backend = %lora.storage.backend_name(),
                        %reason,
                        "Skipping LoRA adapter: remote storage not yet supported in v1"
                    );
                    continue;
                }
            };

            // --- Step 2: Call engine load API ---------------------------------
            let payload = json!({
                "lora_name": lora.id,
                "lora_path": local_path,
            });

            info!(
                lora_id    = %lora.id,
                local_path = %local_path,
                endpoint   = %load_endpoint,
                "Loading LoRA adapter at worker startup"
            );

            match app_context
                .client
                .post(&load_endpoint)
                .json(&payload)
                .send()
                .await
            {
                Ok(resp) if resp.status().is_success() => {
                    info!(lora_id = %lora.id, "LoRA adapter loaded successfully");
                }
                Ok(resp) => {
                    let status = resp.status();
                    let body = resp.text().await.unwrap_or_default();
                    warn!(
                        lora_id = %lora.id,
                        %status,
                        body = %body,
                        "Engine rejected LoRA adapter load; continuing without this adapter"
                    );
                }
                Err(e) => {
                    warn!(
                        lora_id = %lora.id,
                        error = %e,
                        "Failed to reach engine for LoRA load; continuing without this adapter"
                    );
                }
            }
        }

        Ok(StepResult::Continue)
    }
}

/// Resolve the adapter's local filesystem path.
///
/// - `Local`: returns the configured path directly.
/// - Remote variants (`S3`, `Oci`, `Gcs`): returns `Err` with a descriptive
///   message; download support will be added in a future version.
fn resolve_local_path(storage: &LoraStorage) -> Result<String, String> {
    match storage {
        LoraStorage::Local { path } => Ok(path.clone()),
        LoraStorage::S3 { bucket, key, .. } => Err(format!(
            "s3://{bucket}/{key} — S3 download not yet implemented; \
             pre-download the adapter to a local path and use type=local"
        )),
        LoraStorage::Oci { namespace, bucket, object, .. } => Err(format!(
            "oci://{namespace}/{bucket}/{object} — OCI download not yet implemented; \
             pre-download the adapter to a local path and use type=local"
        )),
        LoraStorage::Gcs { bucket, object, .. } => Err(format!(
            "gs://{bucket}/{object} — GCS download not yet implemented; \
             pre-download the adapter to a local path and use type=local"
        )),
    }
}
