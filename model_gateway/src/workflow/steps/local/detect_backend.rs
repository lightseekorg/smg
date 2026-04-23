//! Backend runtime detection step.
//!
//! Detects the runtime type (sglang, vllm, trtllm, mlx, tokenspeed) for both HTTP
//! and gRPC workers.
//! - HTTP: probes `/v1/models` (owned_by field), falls back to unique endpoints.
//! - gRPC: tries sglang → vllm → trtllm → mlx health checks sequentially. If the
//!   SGLang handshake wins, we follow up with `GetServerInfo` to disambiguate
//!   a real SGLang scheduler from a TokenSpeed scheduler (they share the wire
//!   proto) — the TokenSpeed servicer stamps a runtime marker into its
//!   `server_args` struct that the gateway reads here.

use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use smg_grpc_client::SglangSchedulerClient;
use tracing::debug;
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use super::discover_metadata::ModelsResponse;
use crate::{
    worker::ConnectionMode,
    workflow::{
        data::{WorkerKind, WorkerWorkflowData},
        steps::util::{do_grpc_health_check, grpc_base_url, http_base_url},
    },
};

/// Key the TokenSpeed gRPC servicer stamps into its ``server_args`` struct on
/// ``GetServerInfo``. Kept in sync with ``BACKEND_RUNTIME_MARKER_KEY`` in
/// ``grpc_servicer/smg_grpc_servicer/tokenspeed/servicer.py``.
const TOKENSPEED_RUNTIME_MARKER_KEY: &str = "__ts_backend_runtime__";
const TOKENSPEED_RUNTIME_MARKER_VALUE: &str = "tokenspeed";

// ─── gRPC backend detection ────────────────────────────────────────────────

/// Detect gRPC backend by trying runtime-specific health checks sequentially.
///
/// If `runtime_hint` is provided (from explicit config), tries that first.
/// Otherwise tries sglang → vllm → trtllm → mlx.
async fn detect_grpc_backend(
    url: &str,
    timeout_secs: u64,
    runtime_hint: Option<&str>,
) -> Result<String, String> {
    let grpc_url = grpc_base_url(url);

    // If we have a hint, try it first (fast path)
    if let Some(hint) = runtime_hint {
        if do_grpc_health_check(&grpc_url, timeout_secs, hint)
            .await
            .is_ok()
        {
            return Ok(maybe_promote_sglang_to_tokenspeed(&grpc_url, timeout_secs, hint).await);
        }
    }

    // Try each runtime sequentially (most common first), skipping the hint we already tried
    for runtime in &["sglang", "vllm", "trtllm", "mlx"] {
        if Some(*runtime) == runtime_hint {
            continue;
        }
        if do_grpc_health_check(&grpc_url, timeout_secs, runtime)
            .await
            .is_ok()
        {
            return Ok(maybe_promote_sglang_to_tokenspeed(&grpc_url, timeout_secs, runtime).await);
        }
    }

    Err(format!(
        "gRPC backend detection failed for {url} (tried sglang, vllm, trtllm, mlx)"
    ))
}

/// If the SGLang handshake succeeded, follow up with ``GetServerInfo`` and
/// check for the TokenSpeed runtime marker in ``server_args``. A real SGLang
/// scheduler never stamps that key, so its presence reliably distinguishes a
/// TokenSpeed worker that happens to speak the same wire proto. Any probe
/// failure here falls back to ``"sglang"`` — worst case the worker gets
/// labeled as SGLang in metrics, generation still works because the two
/// engines share the proto.
async fn maybe_promote_sglang_to_tokenspeed(
    grpc_url: &str,
    timeout_secs: u64,
    detected: &str,
) -> String {
    if detected != "sglang" {
        return detected.to_string();
    }

    let connect_future = SglangSchedulerClient::connect(grpc_url);
    let client = match tokio::time::timeout(Duration::from_secs(timeout_secs), connect_future).await
    {
        Ok(Ok(c)) => c,
        Ok(Err(e)) => {
            debug!("tokenspeed-promotion: gRPC reconnect to {grpc_url} failed: {e}");
            return "sglang".to_string();
        }
        Err(_) => {
            debug!("tokenspeed-promotion: gRPC reconnect to {grpc_url} timed out");
            return "sglang".to_string();
        }
    };

    let info_future = client.get_server_info();
    let info = match tokio::time::timeout(Duration::from_secs(timeout_secs), info_future).await {
        Ok(Ok(i)) => i,
        Ok(Err(status)) => {
            debug!("tokenspeed-promotion: get_server_info on {grpc_url} failed: {status}");
            return "sglang".to_string();
        }
        Err(_) => {
            debug!("tokenspeed-promotion: get_server_info on {grpc_url} timed out");
            return "sglang".to_string();
        }
    };

    let is_tokenspeed = info
        .server_args
        .as_ref()
        .and_then(|s| s.fields.get(TOKENSPEED_RUNTIME_MARKER_KEY))
        .and_then(|v| v.kind.as_ref())
        .is_some_and(|kind| match kind {
            prost_types::value::Kind::StringValue(s) => s == TOKENSPEED_RUNTIME_MARKER_VALUE,
            _ => false,
        });

    if is_tokenspeed {
        debug!("tokenspeed-promotion: {grpc_url} identified as tokenspeed worker");
        "tokenspeed".to_string()
    } else {
        "sglang".to_string()
    }
}

// ─── HTTP backend detection ────────────────────────────────────────────────

/// Detect HTTP backend by checking `/v1/models` `owned_by` field.
async fn detect_via_models_endpoint(
    url: &str,
    timeout_secs: u64,
    client: &Client,
    api_key: Option<&str>,
) -> Result<String, String> {
    let models_url = format!("{}/v1/models", http_base_url(url));

    let mut req = client
        .get(&models_url)
        .timeout(Duration::from_secs(timeout_secs));
    if let Some(key) = api_key {
        req = req.bearer_auth(key);
    }

    let response = req
        .send()
        .await
        .map_err(|e| format!("Failed to reach {models_url}: {e}"))?;

    if !response.status().is_success() {
        return Err(format!(
            "/v1/models returned status {} from {}",
            response.status(),
            models_url
        ));
    }

    let models: ModelsResponse = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse /v1/models response: {e}"))?;

    let first_model = models
        .data
        .first()
        .ok_or_else(|| format!("/v1/models returned empty data array from {models_url}"))?;

    match first_model.owned_by.as_deref() {
        Some("sglang") => Ok("sglang".to_string()),
        Some("vllm") => Ok("vllm".to_string()),
        other => Err(format!("Unrecognized owned_by value: {other:?}")),
    }
}

/// Probe vLLM's `/version` endpoint.
async fn try_vllm_version(
    url: &str,
    timeout_secs: u64,
    client: &Client,
    api_key: Option<&str>,
) -> Result<(), String> {
    let version_url = format!("{}/version", http_base_url(url));

    let mut req = client
        .get(&version_url)
        .timeout(Duration::from_secs(timeout_secs));
    if let Some(key) = api_key {
        req = req.bearer_auth(key);
    }

    let response = req
        .send()
        .await
        .map_err(|e| format!("Failed to reach {version_url}: {e}"))?;

    if !response.status().is_success() {
        return Err(format!("/version returned {}", response.status()));
    }

    Ok(())
}

/// Probe SGLang's `/server_info` endpoint.
async fn try_sglang_server_info(
    url: &str,
    timeout_secs: u64,
    client: &Client,
    api_key: Option<&str>,
) -> Result<(), String> {
    let info_url = format!("{}/server_info", http_base_url(url));

    let mut req = client
        .get(&info_url)
        .timeout(Duration::from_secs(timeout_secs));
    if let Some(key) = api_key {
        req = req.bearer_auth(key);
    }

    let response = req
        .send()
        .await
        .map_err(|e| format!("Failed to reach {info_url}: {e}"))?;

    if !response.status().is_success() {
        return Err(format!("/server_info returned {}", response.status()));
    }

    Ok(())
}

/// Detect HTTP backend runtime type.
///
/// Strategy:
/// 1. Primary: `GET /v1/models` → check `owned_by` field
/// 2. Fallback: probe `/version` (vLLM) and `/server_info` (SGLang) in parallel
async fn detect_http_backend(
    url: &str,
    timeout_secs: u64,
    client: &Client,
    api_key: Option<&str>,
) -> Result<String, String> {
    // Strategy 1: /v1/models owned_by
    match detect_via_models_endpoint(url, timeout_secs, client, api_key).await {
        Ok(runtime) => {
            debug!("Detected HTTP backend via /v1/models owned_by: {}", runtime);
            return Ok(runtime);
        }
        Err(e) => {
            debug!(
                "Could not detect backend via /v1/models, trying fallback: {}",
                e
            );
        }
    }

    // Strategy 2: probe unique endpoints in parallel.
    // /version is unique to vLLM. /server_info is NOT unique to SGLang — vLLM can
    // also expose it. So /version takes priority: if it succeeds, it's definitely vLLM
    // regardless of whether /server_info also succeeds. We only conclude SGLang if
    // /server_info succeeds and /version does not.
    let (vllm_result, sglang_result) = tokio::join!(
        try_vllm_version(url, timeout_secs, client, api_key),
        try_sglang_server_info(url, timeout_secs, client, api_key),
    );

    if vllm_result.is_ok() {
        if sglang_result.is_ok() {
            debug!(
                "Both /version and /server_info succeeded for {}; /version is vLLM-specific, detecting as vllm",
                url
            );
        }
        return Ok("vllm".to_string());
    }
    if sglang_result.is_ok() {
        debug!("Detected HTTP backend via /server_info (no /version): sglang");
        return Ok("sglang".to_string());
    }

    Err(format!(
        "Could not detect HTTP backend for {url} (tried /v1/models, /version, /server_info)"
    ))
}

// ─── Step implementation ───────────────────────────────────────────────────

/// Step 2: Detect backend runtime type (sglang, vllm, trtllm, mlx).
///
/// Runs after `detect_connection_mode` and before `discover_metadata`.
/// Sets `detected_runtime_type` in workflow data for all downstream steps.
pub struct DetectBackendStep;

#[async_trait]
impl StepExecutor<WorkerWorkflowData> for DetectBackendStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        if context.data.worker_kind != Some(WorkerKind::Local) {
            return Ok(StepResult::Skip);
        }

        let config = &context.data.config;
        let connection_mode =
            context.data.connection_mode.as_ref().ok_or_else(|| {
                WorkflowError::ContextValueNotFound("connection_mode".to_string())
            })?;
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;

        let timeout = config
            .health
            .timeout_secs
            .unwrap_or(app_context.router_config.health_check.timeout_secs);

        // If runtime_type is explicitly configured, use it and skip detection
        let config_runtime = config.runtime_type;
        if config_runtime.is_specified() {
            debug!(
                "Using explicitly configured runtime type: {} for {}",
                config_runtime, config.url
            );
            context.data.detected_runtime_type = Some(config_runtime.to_string());
            return Ok(StepResult::Success);
        }

        debug!(
            "Detecting backend for {} ({:?})",
            config.url, connection_mode
        );

        let detected = match connection_mode {
            ConnectionMode::Http => {
                let client = &app_context.client;
                detect_http_backend(&config.url, timeout, client, config.api_key.as_deref())
                    .await
                    .map_err(|e| WorkflowError::StepFailed {
                        step_id: wfaas::StepId::new("detect_backend"),
                        message: format!("HTTP backend detection failed for {}: {}", config.url, e),
                    })?
            }
            ConnectionMode::Grpc => detect_grpc_backend(&config.url, timeout, None)
                .await
                .map_err(|e| WorkflowError::StepFailed {
                    step_id: wfaas::StepId::new("detect_backend"),
                    message: format!("gRPC backend detection failed for {}: {}", config.url, e),
                })?,
        };

        debug!(
            "Detected backend: {} for {} ({:?})",
            detected, config.url, connection_mode
        );
        context.data.detected_runtime_type = Some(detected);
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        true
    }
}
