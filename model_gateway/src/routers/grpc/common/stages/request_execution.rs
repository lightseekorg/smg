//! Request execution stage: Execute gRPC requests (single or dual dispatch)

use std::time::Instant;

use async_trait::async_trait;
use axum::response::Response;
use tracing::{debug, error, info_span, Instrument};

use super::PipelineStage;
use crate::{
    observability::metrics::{metrics_labels, Metrics},
    routers::{
        error,
        grpc::{
            context::{
                ClientSelection, ExecutionResult, LoadGuards, PdTiming, RequestContext,
                WorkerSelection,
            },
            proto_wrapper::{
                ProtoEmbedRequest, ProtoGenerateRequest, ProtoRequest, ProtoResponseVariant,
                ProtoStream,
            },
            utils::tonic_ext::{TonicResultExt, TonicStatusExt},
        },
    },
    worker::{RuntimeType, DEFAULT_BOOTSTRAP_PORT, MOONCAKE_CONNECTOR, NIXL_CONNECTOR},
};

type StreamResult = Result<ProtoStream, tonic::Status>;

/// KV-transfer params tagged onto the NIXL prefill leg so the engine pins its
/// KV blocks and returns the handoff params for the decode worker.
const NIXL_PREFILL_KV_PARAMS: &str = r#"{"do_remote_decode":true,"do_remote_prefill":false}"#;

/// PD KV-transfer behavior derived from prefill worker metadata.
#[derive(Debug, Clone, PartialEq)]
enum KvConnectorMode {
    /// MooncakeConnector: mint a transfer_id, tag both legs, synthesize decode
    /// params from worker metadata; legacy host/port injection when the
    /// servicer predates kv_engine_id reporting (or DP runs without a pinned rank).
    Mooncake {
        host: String,
        port: u32,
        engine_id: Option<String>,
    },
    /// NixlConnector: tag prefill with do_remote_decode, relay returned params to decode.
    Nixl,
    /// Unknown/absent connector: relay returned params opportunistically.
    Passthrough,
}

impl KvConnectorMode {
    fn metrics_label(&self) -> &'static str {
        match self {
            Self::Mooncake { .. } => metrics_labels::KV_CONNECTOR_MOONCAKE,
            Self::Nixl => metrics_labels::KV_CONNECTOR_NIXL,
            Self::Passthrough => metrics_labels::KV_CONNECTOR_PASSTHROUGH,
        }
    }
}

fn kv_connector_mode(
    kv_connector: Option<&str>,
    bootstrap_host: &str,
    bootstrap_port: Option<u16>,
    kv_engine_id: Option<&str>,
) -> KvConnectorMode {
    match kv_connector {
        Some(MOONCAKE_CONNECTOR) => KvConnectorMode::Mooncake {
            host: bootstrap_host.to_string(),
            port: u32::from(bootstrap_port.unwrap_or(DEFAULT_BOOTSTRAP_PORT)),
            // Empty means unknown (forces the legacy fallback)
            engine_id: kv_engine_id.filter(|s| !s.is_empty()).map(str::to_string),
        },
        Some(NIXL_CONNECTOR) => KvConnectorMode::Nixl,
        _ => KvConnectorMode::Passthrough,
    }
}

/// Connector id of the engine core serving the prefill leg. With DP the cores
/// suffix the configured id as `{base}_dp{rank}`, so minting needs a pinned
/// rank; unpinned DP>1 yields None (no mint — decode recomputes locally).
fn effective_kv_engine_id(
    base: Option<&str>,
    dp_size: Option<usize>,
    dp_rank: Option<usize>,
) -> Option<String> {
    let base = base.filter(|s| !s.is_empty())?;
    if dp_size.unwrap_or(1) > 1 {
        dp_rank.map(|rank| format!("{base}_dp{rank}"))
    } else {
        Some(base.to_string())
    }
}

/// Prefill-leg params for Mooncake: the engine pins blocks under the minted id.
fn mooncake_prefill_params(transfer_id: &str) -> String {
    serde_json::json!({
        "do_remote_decode": true,
        "do_remote_prefill": false,
        "transfer_id": transfer_id,
    })
    .to_string()
}

/// Decode-leg params for Mooncake, synthesized from prefill worker metadata
/// (the engine returns nothing to relay; the connector is push-based).
fn mooncake_decode_params(transfer_id: &str, engine_id: &str, host: &str, port: u32) -> String {
    serde_json::json!({
        "do_remote_decode": false,
        "do_remote_prefill": true,
        "transfer_id": transfer_id,
        "remote_engine_id": engine_id,
        "remote_bootstrap_addr": format!("http://{host}:{port}"),
    })
    .to_string()
}

/// Request execution stage: Execute gRPC requests (single or dual dispatch)
pub(crate) struct RequestExecutionStage {
    mode: ExecutionMode,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ExecutionMode {
    /// Regular mode: single worker execution
    Single,
    /// PD mode: dual dispatch to prefill + decode workers
    DualDispatch,
    /// EPD mode: encode was already dispatched by the encode stage; this runs
    /// the prefill+decode leg as a parallel dual dispatch, bypassing the
    /// runtime-type PD gate (TokenSpeed EPD is its only user).
    Epd,
}

impl RequestExecutionStage {
    pub fn new(mode: ExecutionMode) -> Self {
        Self { mode }
    }
}

#[async_trait]
impl PipelineStage for RequestExecutionStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let proto_request = ctx.state.proto_request.take().ok_or_else(|| {
            error!(
                function = "RequestExecutionStage::execute",
                "Proto request not built"
            );
            error::internal_error("proto_request_not_built", "Proto request not built")
        })?;

        let clients = ctx.state.clients.as_mut().ok_or_else(|| {
            error!(
                function = "RequestExecutionStage::execute",
                "Client acquisition not completed"
            );
            error::internal_error(
                "client_acquisition_not_completed",
                "Client acquisition not completed",
            )
        })?;

        // Create load guards for worker load tracking (increment load when created)
        // They will be automatically dropped (and decrement load) when RequestContext is dropped
        let workers = ctx.state.workers.as_ref().ok_or_else(|| {
            error!(
                function = "RequestExecutionStage::execute",
                "Worker selection not completed"
            );
            error::internal_error(
                "worker_selection_not_completed",
                "Worker selection not completed",
            )
        })?;

        ctx.state.load_guards = Some(LoadGuards::new(workers, ctx.input.headers.as_ref()));

        // Extract dispatch metadata for tracing span
        let dispatch = ctx.state.dispatch.as_ref();
        let request_id = dispatch.map(|d| d.request_id.as_str()).unwrap_or("unknown");
        let model = dispatch.map(|d| d.model.as_str()).unwrap_or("unknown");
        let request_type = match &proto_request {
            ProtoRequest::Generate(_) => "generate",
            ProtoRequest::Embed(_) => "embed",
        };

        // Create OTEL span for gRPC request execution
        let span = info_span!(
            target: "smg::otel-trace",
            "grpc_execute",
            request_type,
            request_id = %request_id,
            model = %model,
            mode = ?self.mode,
        );

        let result = async {
            match proto_request {
                ProtoRequest::Generate(req) => match self.mode {
                    ExecutionMode::Single => self.execute_single(req, clients, workers).await,
                    ExecutionMode::DualDispatch => {
                        // Dispatch based on runtime type:
                        // - SGLang: parallel dual dispatch with bootstrap metadata
                        // - vLLM: sequential prefill-then-decode with kv_transfer_params relay
                        let runtime_type = workers.pd_runtime_type();
                        match runtime_type {
                            Some(RuntimeType::Vllm) => {
                                self.execute_sequential_pd(req, clients, workers, model)
                                    .await
                            }
                            Some(RuntimeType::Sglang) => {
                                self.execute_dual_dispatch(req, clients, workers).await
                            }
                            Some(RuntimeType::Trtllm)
                            | Some(RuntimeType::Mlx)
                            | Some(RuntimeType::TokenSpeed)
                            | Some(RuntimeType::External)
                            | Some(RuntimeType::Unspecified) => {
                                error!(
                                    function = "RequestExecutionStage::execute",
                                    runtime_type = ?runtime_type,
                                    "Runtime does not support PD disaggregated mode"
                                );
                                Err(error::bad_request(
                                    "runtime_pd_not_supported",
                                    "This runtime does not support PD disaggregated mode",
                                ))
                            }
                            None => {
                                error!(
                                    function = "RequestExecutionStage::execute",
                                    "PD mode requires dual worker selection"
                                );
                                Err(error::internal_error(
                                    "pd_mode_requires_dual_workers",
                                    "PD mode requires dual worker selection",
                                ))
                            }
                        }
                    }
                    ExecutionMode::Epd => {
                        // Encode was already dispatched by the EncodeStage; run the
                        // prefill+decode leg as a parallel dual dispatch (prefill
                        // carries the encode handshake + metadata-only mm). This
                        // bypasses the runtime-type PD gate above; TokenSpeed EPD
                        // uses the same Mooncake-bootstrap dual dispatch as SGLang.
                        self.execute_dual_dispatch(req, clients, workers).await
                    }
                },
                ProtoRequest::Embed(req) => self.execute_single_embed(req, clients, workers).await,
            }
        }
        .instrument(span)
        .await?;

        // Store result in context for ResponseProcessingStage
        ctx.state.response.execution_result = Some(result);
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "RequestExecution"
    }
}

impl RequestExecutionStage {
    async fn execute_single(
        &self,
        mut proto_request: ProtoGenerateRequest,
        clients: &mut ClientSelection,
        workers: &WorkerSelection,
    ) -> Result<ExecutionResult, Response> {
        let client = clients.single_mut().ok_or_else(|| {
            error!(
                function = "execute_single",
                "Expected single client but got dual"
            );
            error::internal_error(
                "expected_single_client_got_dual",
                "Expected single client but got dual",
            )
        })?;

        if let Some(rank) = workers.single().and_then(|w| w.dp_rank()) {
            proto_request.set_data_parallel_rank(rank as i32);
        }

        let result = client.generate(proto_request).await;
        workers.record_outcome(result.cb_status_code());

        let stream = result.map_err(|e| {
            error!(function = "execute_single", error = %e, "Failed to start generation");
            e.to_http_error(
                "start_generation_failed",
                format!("Failed to start generation: {}", e.message()),
            )
        })?;

        Ok(ExecutionResult::Single { stream })
    }

    async fn execute_single_embed(
        &self,
        proto_request: ProtoEmbedRequest,
        clients: &mut ClientSelection,
        workers: &WorkerSelection,
    ) -> Result<ExecutionResult, Response> {
        let client = clients.single_mut().ok_or_else(|| {
            error!(
                function = "execute_single_embed",
                "Expected single client but got dual"
            );
            error::internal_error(
                "expected_single_client_got_dual",
                "Expected single client but got dual",
            )
        })?;

        let result = client.embed(proto_request).await;
        workers.record_outcome(result.cb_status_code());

        let complete = result.map_err(|e| {
            error!(function = "execute_single_embed", error = %e, "Failed to start embedding");
            e.to_http_error(
                "start_embedding_failed",
                format!("Failed to start embedding: {}", e.message()),
            )
        })?;

        Ok(ExecutionResult::Embedding { response: complete })
    }

    async fn execute_dual_dispatch(
        &self,
        proto_request: ProtoGenerateRequest,
        clients: &mut ClientSelection,
        workers: &WorkerSelection,
    ) -> Result<ExecutionResult, Response> {
        let runtime = workers.pd_runtime_type().map(|r| r.as_str()).unwrap_or("");
        let (prefill_client, decode_client) = clients.dual_mut().ok_or_else(|| {
            error!(
                function = "execute_dual_dispatch",
                "Expected dual clients but got single"
            );
            error::internal_error(
                "expected_dual_clients_got_single",
                "Expected dual clients but got single",
            )
        })?;

        let mut prefill_request = proto_request.clone_inner();
        // Drop only the heavy pixel tensors from the decode request (~40MB saved),
        // but KEEP the lightweight mm metadata (placeholders, im_token_id, grid_thw).
        // The decode engine needs the placeholder offsets to pad_value-disambiguate
        // image tokens; without it the bare image-placeholder tokens compare equal
        // across images and the decode prefix cache reuses a previous request's
        // image KV (reads the wrong image). grid_thw is kept so the decode computes
        // the same MRoPE positions/delta the prefill did (consistent with the
        // transferred mrope delta). Non-TokenSpeed backends no-op here.
        let mut decode_request = proto_request;
        decode_request.clear_mm_pixel_values();
        if let Some(rank) = workers.prefill_worker().and_then(|w| w.dp_rank()) {
            prefill_request.set_data_parallel_rank(rank as i32);
        }
        if let Some(rank) = workers.decode_worker().and_then(|w| w.dp_rank()) {
            decode_request.set_data_parallel_rank(rank as i32);
        }

        // `generate` only establishes the prefill stream here (SMG does not drain
        // it on this path), so prefill duration cannot be measured — only TTFT,
        // recorded at the first decode token in streaming. prefill_start anchors it.
        let prefill_start = Instant::now();
        let (prefill_result, decode_result): (StreamResult, StreamResult) = tokio::join!(
            prefill_client.generate(prefill_request),
            decode_client.generate(decode_request)
        );

        // Record circuit breaker outcomes (client errors don't count as failures)
        workers.record_dual_outcomes(
            prefill_result.cb_status_code(),
            decode_result.cb_status_code(),
        );

        // Handle prefill result
        let prefill_stream = prefill_result.map_err(|e| {
            Metrics::record_worker_error(
                metrics_labels::WORKER_PREFILL,
                metrics_labels::CONNECTION_GRPC,
                metrics_labels::ERROR_BACKEND,
            );
            error!(function = "execute_dual_dispatch", error = %e, "Prefill worker failed to start");
            e.to_http_error("prefill_worker_failed_to_start", format!("Prefill worker failed to start: {}", e.message()))
        })?;

        // Handle decode result
        let decode_stream = decode_result.map_err(|e| {
            Metrics::record_worker_error(
                metrics_labels::WORKER_DECODE,
                metrics_labels::CONNECTION_GRPC,
                metrics_labels::ERROR_BACKEND,
            );
            error!(function = "execute_dual_dispatch", error = %e, "Decode worker failed to start");
            e.to_http_error(
                "decode_worker_failed_to_start",
                format!("Decode worker failed to start: {}", e.message()),
            )
        })?;

        Ok(ExecutionResult::Dual {
            prefill: prefill_stream,
            decode: Box::new(decode_stream),
            pd_timing: PdTiming {
                prefill_start,
                runtime,
            },
        })
    }

    /// Execute vLLM PD: send to prefill with max_tokens=1 first, wait for completion,
    /// then send original request to decode.
    ///
    /// For Mooncake: injects bootstrap_host/port from prefill worker metadata into
    /// the decode request. For NIXL: tags the prefill request with do_remote_decode,
    /// then relays the kv_transfer_params returned by the prefill engine to decode.
    async fn execute_sequential_pd(
        &self,
        proto_request: ProtoGenerateRequest,
        clients: &mut ClientSelection,
        workers: &WorkerSelection,
        model: &str,
    ) -> Result<ExecutionResult, Response> {
        let runtime = workers.pd_runtime_type().map(|r| r.as_str()).unwrap_or("");
        let (prefill_client, decode_client) = clients.dual_mut().ok_or_else(|| {
            error!(
                function = "execute_sequential_pd",
                "Expected dual clients but got single"
            );
            error::internal_error(
                "expected_dual_clients_got_single",
                "Expected dual clients but got single",
            )
        })?;

        let mode = workers
            .prefill_worker()
            .map(|w| {
                let meta = w.metadata();
                // Discovered dp_size matters even without --dp-aware expansion:
                // a DP>1 engine behind an unexpanded worker must not be minted for
                let dp_size = w
                    .dp_size()
                    .or_else(|| meta.spec.labels.get("dp_size").and_then(|s| s.parse().ok()));
                let engine_id =
                    effective_kv_engine_id(meta.spec.kv_engine_id.as_deref(), dp_size, w.dp_rank());
                kv_connector_mode(
                    meta.spec.kv_connector.as_deref(),
                    &meta.spec.bootstrap_host,
                    meta.spec.bootstrap_port,
                    engine_id.as_deref(),
                )
            })
            .unwrap_or(KvConnectorMode::Passthrough);

        // Recorded on the success path (after decode established) so failed
        // requests don't pollute success metrics; captured here before use of mode.
        let kv_connector_label = mode.metrics_label();

        match &mode {
            KvConnectorMode::Mooncake {
                host,
                port,
                engine_id,
            } => debug!(
                bootstrap_host = %host,
                bootstrap_port = port,
                engine_id_known = engine_id.is_some(),
                "vLLM PD (Mooncake): will inject kv_transfer_params into decode request"
            ),
            KvConnectorMode::Nixl => debug!(
                "vLLM PD (NIXL): will tag prefill with do_remote_decode and relay returned kv_transfer_params to decode"
            ),
            KvConnectorMode::Passthrough => {
                // Warn once: PD without a discovered connector usually means GetServerInfo
                // lacks kv fields or labels.kv_connector is missing in worker config
                static WARN_ONCE: std::sync::Once = std::sync::Once::new();
                WARN_ONCE.call_once(|| {
                    tracing::warn!(
                        "vLLM PD: no kv_connector detected on prefill worker; KV transfer params \
                         will only be relayed if the engine returns them"
                    );
                });
            }
        }

        // The KV handoff is single-consumer: with n>1 each fan-out child on decode
        // would pull, and the first completion frees the prefill blocks under its
        // siblings (same hazard for NIXL and Mooncake)
        let relay_kv_params = proto_request.sampling_n() <= 1;

        // Mooncake is push-based: the engine returns nothing, so the router mints
        // the transfer correlation id and synthesizes decode params from metadata
        let mooncake_transfer_id = match &mode {
            KvConnectorMode::Mooncake {
                engine_id: Some(_), ..
            } if relay_kv_params => Some(format!("xfer-{}", uuid::Uuid::now_v7())),
            _ => None,
        };

        // Clone request and sanitize sampling (max_tokens=1, n=1), stream=false for prefill
        let mut prefill_request = proto_request.clone_inner();
        prefill_request.sanitize_sampling_for_prefill(1);
        prefill_request.set_stream(false);
        if let Some(rank) = workers.prefill_worker().and_then(|w| w.dp_rank()) {
            prefill_request.set_data_parallel_rank(rank as i32);
        }
        if mode == KvConnectorMode::Nixl {
            if relay_kv_params {
                prefill_request.set_kv_transfer_params_json(NIXL_PREFILL_KV_PARAMS.to_string());
            } else {
                debug!(
                    request_id = %prefill_request.request_id(),
                    "vLLM PD (NIXL): n>1 request, skipping kv_transfer_params relay \
                     (decode recomputes the prompt locally)"
                );
            }
        }
        if let Some(ref transfer_id) = mooncake_transfer_id {
            prefill_request.set_kv_transfer_params_json(mooncake_prefill_params(transfer_id));
        }

        debug!(
            request_id = %prefill_request.request_id(),
            "vLLM PD: sending prefill request (max_tokens=1)"
        );

        // Send to prefill, wait for completion
        let prefill_start = Instant::now();
        let mut prefill_stream = prefill_client
            .generate(prefill_request)
            .await
            .map_err(|e| {
                workers.record_outcome_prefill(e.http_status().as_u16());
                Metrics::record_worker_error(
                    metrics_labels::WORKER_PREFILL,
                    metrics_labels::CONNECTION_GRPC,
                    metrics_labels::ERROR_BACKEND,
                );
                error!(function = "execute_sequential_pd", error = %e, "Prefill worker failed to start");
                e.to_http_error("prefill_worker_failed_to_start", format!("Prefill worker failed to start: {}", e.message()))
            })?;

        // Drain prefill response, harvesting connector params from the Complete frame
        let mut prefill_kv_params: Option<String> = None;
        while let Some(result) = prefill_stream.next().await {
            match result {
                Ok(response) => {
                    if let ProtoResponseVariant::Complete(complete) = response.into_response() {
                        if let Some(json) = complete.kv_transfer_params_json() {
                            prefill_kv_params = Some(json.to_owned());
                        }
                    }
                }
                Err(e) => {
                    workers.record_outcome_prefill(e.http_status().as_u16());
                    Metrics::record_worker_error(
                        metrics_labels::WORKER_PREFILL,
                        metrics_labels::CONNECTION_GRPC,
                        metrics_labels::ERROR_BACKEND,
                    );
                    error!(function = "execute_sequential_pd", error = %e, "Prefill stream error");
                    return Err(e.to_http_error(
                        "prefill_stream_error",
                        format!("Prefill stream error: {}", e.message()),
                    ));
                }
            }
        }
        prefill_stream.mark_completed();
        workers.record_outcome_prefill(200);
        // Captured at drain; recorded below only once decode is established.
        let prefill_duration = prefill_start.elapsed();

        // KV-transfer window: prefill drain complete to decode send complete.
        let kv_window_start = Instant::now();

        debug!("vLLM PD: prefill completed, sending decode request");

        // Decode reuses proto_request as-is; same request_id as the prefill leg is
        // load-bearing for NIXL P/D correlation on vLLM < 0.13
        let mut decode_request = proto_request;
        if let Some(rank) = workers.decode_worker().and_then(|w| w.dp_rank()) {
            decode_request.set_data_parallel_rank(rank as i32);
        }
        match (&mode, prefill_kv_params) {
            // Modern Mooncake: synthesized params under the minted transfer_id
            (
                KvConnectorMode::Mooncake {
                    host,
                    port,
                    engine_id: Some(engine_id),
                },
                _,
            ) if mooncake_transfer_id.is_some() => {
                let transfer_id = mooncake_transfer_id.as_deref().unwrap_or_default();
                debug!(
                    request_id = %decode_request.request_id(),
                    transfer_id = %transfer_id,
                    "vLLM PD (Mooncake): injecting minted kv_transfer_params into decode request"
                );
                decode_request.set_kv_transfer_params_json(mooncake_decode_params(
                    transfer_id,
                    engine_id,
                    host,
                    *port,
                ));
            }
            // Legacy Mooncake (no engine_id discovered, or n>1): typed host/port injection
            (KvConnectorMode::Mooncake { host, port, .. }, _) => {
                debug!(
                    remote_host = %host,
                    remote_port = port,
                    "vLLM PD: injecting kv_transfer_params into decode request"
                );
                decode_request.set_kv_transfer_params(host.clone(), *port);
            }
            (KvConnectorMode::Nixl | KvConnectorMode::Passthrough, Some(json))
                if relay_kv_params =>
            {
                debug!(
                    request_id = %decode_request.request_id(),
                    params_len = json.len(),
                    "vLLM PD: relaying prefill kv_transfer_params to decode request"
                );
                decode_request.set_kv_transfer_params_json(json);
            }
            (KvConnectorMode::Nixl, None) if relay_kv_params => {
                Metrics::record_pd_kv_transfer_failure();
                tracing::warn!(
                    request_id = %decode_request.request_id(),
                    "vLLM PD (NIXL): prefill returned no kv_transfer_params; decode will \
                     recompute the prompt locally (outdated smg-grpc-servicer or missing \
                     kv-transfer-config?)"
                );
            }
            _ => {}
        }

        // Send request to decode
        let decode_stream = decode_client.generate(decode_request).await.map_err(|e| {
            workers.record_outcome_decode(e.http_status().as_u16());
            Metrics::record_worker_error(
                metrics_labels::WORKER_DECODE,
                metrics_labels::CONNECTION_GRPC,
                metrics_labels::ERROR_BACKEND,
            );
            error!(function = "execute_sequential_pd", error = %e, "Decode worker failed to start");
            e.to_http_error(
                "decode_worker_failed_to_start",
                format!("Decode worker failed to start: {}", e.message()),
            )
        })?;

        workers.record_outcome_decode(200);
        // Decode established: record the success-only PD metrics here.
        Metrics::record_pd_kv_connector_mode(kv_connector_label);
        Metrics::record_pd_prefill_duration(
            metrics_labels::BACKEND_PD,
            model,
            runtime,
            prefill_duration,
        );
        Metrics::record_pd_kv_transfer_duration(
            metrics_labels::BACKEND_PD,
            model,
            runtime,
            kv_window_start.elapsed(),
        );

        Ok(ExecutionResult::Single {
            stream: decode_stream,
        })
    }
}

#[cfg(test)]
mod tests {
    use smg_grpc_client::vllm_proto as vllm;

    use super::*;

    #[test]
    fn kv_connector_mode_mooncake_uses_bootstrap_metadata() {
        let mode = kv_connector_mode(
            Some(MOONCAKE_CONNECTOR),
            "prefill-host",
            Some(9090),
            Some("engine-1"),
        );
        assert_eq!(
            mode,
            KvConnectorMode::Mooncake {
                host: "prefill-host".to_string(),
                port: 9090,
                engine_id: Some("engine-1".to_string()),
            }
        );
    }

    #[test]
    fn kv_connector_mode_mooncake_defaults_port_and_tolerates_missing_engine_id() {
        let mode = kv_connector_mode(Some(MOONCAKE_CONNECTOR), "prefill-host", None, None);
        assert_eq!(
            mode,
            KvConnectorMode::Mooncake {
                host: "prefill-host".to_string(),
                port: u32::from(DEFAULT_BOOTSTRAP_PORT),
                engine_id: None,
            }
        );
    }

    #[test]
    fn kv_connector_mode_mooncake_empty_engine_id_means_legacy() {
        let mode = kv_connector_mode(Some(MOONCAKE_CONNECTOR), "host", Some(9090), Some(""));
        assert_eq!(
            mode,
            KvConnectorMode::Mooncake {
                host: "host".to_string(),
                port: 9090,
                engine_id: None,
            }
        );
    }

    #[test]
    fn kv_connector_mode_nixl() {
        assert_eq!(
            kv_connector_mode(Some(NIXL_CONNECTOR), "ignored", Some(9090), None),
            KvConnectorMode::Nixl
        );
    }

    #[test]
    fn kv_connector_mode_unknown_or_missing_is_passthrough() {
        assert_eq!(
            kv_connector_mode(Some("LMCacheConnector"), "host", None, None),
            KvConnectorMode::Passthrough
        );
        assert_eq!(
            kv_connector_mode(None, "host", None, None),
            KvConnectorMode::Passthrough
        );
    }

    #[test]
    fn mooncake_prefill_params_carry_transfer_id() {
        let value: serde_json::Value =
            serde_json::from_str(&mooncake_prefill_params("xfer-abc")).unwrap();
        assert_eq!(value["do_remote_decode"], true);
        assert_eq!(value["do_remote_prefill"], false);
        assert_eq!(value["transfer_id"], "xfer-abc");
        assert_eq!(value.as_object().unwrap().len(), 3);
    }

    #[test]
    fn mooncake_decode_params_synthesize_full_handoff() {
        let value: serde_json::Value = serde_json::from_str(&mooncake_decode_params(
            "xfer-abc", "engine-1", "10.0.0.1", 8998,
        ))
        .unwrap();
        assert_eq!(value["do_remote_decode"], false);
        assert_eq!(value["do_remote_prefill"], true);
        assert_eq!(value["transfer_id"], "xfer-abc");
        assert_eq!(value["remote_engine_id"], "engine-1");
        assert_eq!(value["remote_bootstrap_addr"], "http://10.0.0.1:8998");
        assert_eq!(value.as_object().unwrap().len(), 5);
    }

    #[test]
    fn nixl_prefill_kv_params_is_valid_json() {
        let value: serde_json::Value = serde_json::from_str(NIXL_PREFILL_KV_PARAMS).unwrap();
        assert_eq!(value["do_remote_decode"], true);
        assert_eq!(value["do_remote_prefill"], false);
        assert_eq!(value.as_object().unwrap().len(), 2);
    }

    #[test]
    fn sanitize_sampling_for_prefill_forces_length_capped_finish() {
        let mut request = ProtoGenerateRequest::Vllm(Box::new(vllm::GenerateRequest {
            sampling_params: Some(vllm::SamplingParams {
                max_tokens: Some(128),
                min_tokens: 16,
                n: 4,
                stop: vec!["</s>".to_string()],
                stop_token_ids: vec![2],
                ignore_eos: false,
                ..Default::default()
            }),
            ..Default::default()
        }));
        request.sanitize_sampling_for_prefill(1);
        let ProtoGenerateRequest::Vllm(req) = request else {
            panic!("expected vLLM request");
        };
        let params = req.sampling_params.unwrap();
        assert_eq!(params.max_tokens, Some(1));
        assert_eq!(params.min_tokens, 0);
        assert_eq!(params.n, 1);
        assert!(params.stop.is_empty());
        assert!(params.stop_token_ids.is_empty());
        assert!(params.ignore_eos);
    }

    #[test]
    fn sampling_n_defaults_to_one() {
        let unset = ProtoGenerateRequest::Vllm(Box::default());
        assert_eq!(unset.sampling_n(), 1);

        let zero = ProtoGenerateRequest::Vllm(Box::new(vllm::GenerateRequest {
            sampling_params: Some(vllm::SamplingParams {
                n: 0,
                ..Default::default()
            }),
            ..Default::default()
        }));
        assert_eq!(zero.sampling_n(), 1);

        let four = ProtoGenerateRequest::Vllm(Box::new(vllm::GenerateRequest {
            sampling_params: Some(vllm::SamplingParams {
                n: 4,
                ..Default::default()
            }),
            ..Default::default()
        }));
        assert_eq!(four.sampling_n(), 4);
    }

    #[test]
    fn kv_transfer_params_json_request_roundtrip() {
        let mut request = ProtoGenerateRequest::Vllm(Box::default());
        request.set_kv_transfer_params_json(NIXL_PREFILL_KV_PARAMS.to_string());
        let ProtoGenerateRequest::Vllm(req) = request else {
            panic!("expected vLLM request");
        };
        assert_eq!(
            req.kv_transfer_params_json.as_deref(),
            Some(NIXL_PREFILL_KV_PARAMS)
        );
    }

    #[test]
    fn kv_transfer_params_json_complete_accessor_filters_empty() {
        use crate::routers::grpc::proto_wrapper::ProtoGenerateComplete;

        let complete = ProtoGenerateComplete::Vllm(vllm::GenerateComplete {
            kv_transfer_params_json: Some(r#"{"do_remote_prefill":true}"#.to_string()),
            ..Default::default()
        });
        assert_eq!(
            complete.kv_transfer_params_json(),
            Some(r#"{"do_remote_prefill":true}"#)
        );

        let empty = ProtoGenerateComplete::Vllm(vllm::GenerateComplete {
            kv_transfer_params_json: Some(String::new()),
            ..Default::default()
        });
        assert_eq!(empty.kv_transfer_params_json(), None);

        let unset = ProtoGenerateComplete::Vllm(vllm::GenerateComplete::default());
        assert_eq!(unset.kv_transfer_params_json(), None);
    }

    #[test]
    fn effective_engine_id_passthrough_when_no_dp() {
        assert_eq!(
            effective_kv_engine_id(Some("eng"), None, None).as_deref(),
            Some("eng")
        );
        assert_eq!(
            effective_kv_engine_id(Some("eng"), Some(1), None).as_deref(),
            Some("eng")
        );
    }

    #[test]
    fn effective_engine_id_suffixes_pinned_dp_rank() {
        assert_eq!(
            effective_kv_engine_id(Some("eng"), Some(2), Some(1)).as_deref(),
            Some("eng_dp1")
        );
        assert_eq!(
            effective_kv_engine_id(Some("eng"), Some(2), Some(0)).as_deref(),
            Some("eng_dp0")
        );
    }

    #[test]
    fn effective_engine_id_none_for_unpinned_dp() {
        assert_eq!(effective_kv_engine_id(Some("eng"), Some(2), None), None);
    }

    #[test]
    fn effective_engine_id_none_for_missing_or_empty_base() {
        assert_eq!(effective_kv_engine_id(None, Some(2), Some(0)), None);
        assert_eq!(effective_kv_engine_id(Some(""), None, None), None);
    }
}
