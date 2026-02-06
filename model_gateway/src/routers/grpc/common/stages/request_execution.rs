//! Request execution stage: Execute gRPC requests (single or dual dispatch)

use async_trait::async_trait;
use axum::response::Response;
use tracing::{debug, error, info_span, Instrument};

use super::PipelineStage;
use crate::{
    core::{RuntimeType, DEFAULT_BOOTSTRAP_PORT, MOONCAKE_CONNECTOR},
    routers::{
        error,
        grpc::{
            context::{
                ClientSelection, ExecutionResult, LoadGuards, RequestContext, WorkerSelection,
            },
            proto_wrapper::{
                ProtoEmbedRequest, ProtoEmbedResponseVariant, ProtoGenerateRequest, ProtoRequest,
                ProtoStream,
            },
        },
    },
};

type StreamResult = Result<ProtoStream, Box<dyn std::error::Error + Send + Sync>>;

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
        let request_id = ctx
            .state
            .dispatch
            .as_ref()
            .map(|d| d.request_id.as_str())
            .unwrap_or("unknown");
        let model = ctx
            .state
            .dispatch
            .as_ref()
            .map(|d| d.model.as_str())
            .unwrap_or("unknown");

        // Create OTEL span for gRPC request execution
        let span = info_span!(
            target: "smg::otel-trace",
            "grpc_generate",
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
                        // - vLLM: sequential prefill-then-decode (NIXL handles KV transfer)
                        match workers.pd_runtime_type() {
                            Some(RuntimeType::Vllm) => {
                                self.execute_sequential_pd(req, clients, workers).await
                            }
                            Some(RuntimeType::Sglang) => {
                                self.execute_dual_dispatch(req, clients, workers).await
                            }
                            Some(RuntimeType::Trtllm) | Some(RuntimeType::External) => {
                                error!(
                                    function = "RequestExecutionStage::execute",
                                    runtime_type = ?workers.pd_runtime_type(),
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
                },
                ProtoRequest::Embed(req) => self.execute_single_embed(req, clients).await,
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
        proto_request: ProtoGenerateRequest,
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

        let result = client.generate(proto_request).await;

        // Record circuit breaker outcome
        workers.record_outcome(result.is_ok());

        let stream = result.map_err(|e| {
            error!(
                function = "execute_single",
                error = %e,
                "Failed to start generation"
            );
            error::internal_error(
                "start_generation_failed",
                format!("Failed to start generation: {}", e),
            )
        })?;

        Ok(ExecutionResult::Single { stream })
    }

    async fn execute_single_embed(
        &self,
        proto_request: ProtoEmbedRequest,
        clients: &mut ClientSelection,
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

        let response = client.embed(proto_request).await.map_err(|e| {
            error!(
                function = "execute_single_embed",
                error = %e,
                "Failed to start embedding"
            );
            error::internal_error(
                "start_embedding_failed",
                format!("Failed to start embedding: {}", e),
            )
        })?;

        match response.into_response() {
            ProtoEmbedResponseVariant::Complete(complete) => {
                Ok(ExecutionResult::Embedding { response: complete })
            }
            ProtoEmbedResponseVariant::Error(e) => {
                error!(
                    function = "execute_single_embed",
                    error = %e.message(),
                    "Embedding execution failed"
                );
                Err(error::internal_error(
                    "embedding_execution_failed",
                    e.message().to_string(),
                ))
            }
            ProtoEmbedResponseVariant::None => {
                error!(
                    function = "execute_single_embed",
                    "Embedding execution returned no response"
                );
                Err(error::internal_error(
                    "embedding_no_response",
                    "Embedding execution returned no response",
                ))
            }
        }
    }

    async fn execute_dual_dispatch(
        &self,
        proto_request: ProtoGenerateRequest,
        clients: &mut ClientSelection,
        workers: &WorkerSelection,
    ) -> Result<ExecutionResult, Response> {
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

        let prefill_request = proto_request.clone_inner();
        let decode_request = proto_request;

        let (prefill_result, decode_result): (StreamResult, StreamResult) = tokio::join!(
            prefill_client.generate(prefill_request),
            decode_client.generate(decode_request)
        );

        // Record circuit breaker outcomes for each worker individually
        workers.record_dual_outcomes(prefill_result.is_ok(), decode_result.is_ok());

        // Handle prefill result
        let prefill_stream = match prefill_result {
            Ok(s) => s,
            Err(e) => {
                error!(
                    function = "execute_dual_dispatch",
                    error = %e,
                    "Prefill worker failed to start"
                );
                return Err(error::internal_error(
                    "prefill_worker_failed_to_start",
                    format!("Prefill worker failed to start: {}", e),
                ));
            }
        };

        // Handle decode result
        let decode_stream = match decode_result {
            Ok(s) => s,
            Err(e) => {
                error!(
                    function = "execute_dual_dispatch",
                    error = %e,
                    "Decode worker failed to start"
                );
                return Err(error::internal_error(
                    "decode_worker_failed_to_start",
                    format!("Decode worker failed to start: {}", e),
                ));
            }
        };

        Ok(ExecutionResult::Dual {
            prefill: prefill_stream,
            decode: Box::new(decode_stream),
        })
    }

    /// Execute vLLM PD: send to prefill with max_tokens=1 first, wait for completion,
    /// then send original request to decode. NIXL/Mooncake handles KV cache transfer.
    ///
    /// For Mooncake: uses bootstrap_host/port from prefill worker metadata to inject
    /// kv_transfer_params into decode request so decode knows where to fetch KV cache.
    /// For NIXL: no kv_transfer_params needed (uses prompt prefix matching).
    async fn execute_sequential_pd(
        &self,
        proto_request: ProtoGenerateRequest,
        clients: &mut ClientSelection,
        workers: &WorkerSelection,
    ) -> Result<ExecutionResult, Response> {
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

        // Get bootstrap info from prefill worker metadata (only for Mooncake PD)
        // NIXL uses prefix matching and doesn't need kv_transfer_params
        let kv_transfer_params: Option<(String, u32)> = workers
            .prefill_worker()
            .map(|w| w.metadata())
            .filter(|meta| meta.kv_connector.as_deref() == Some(MOONCAKE_CONNECTOR))
            .map(|meta| {
                let port = meta.bootstrap_port.unwrap_or(DEFAULT_BOOTSTRAP_PORT);
                (meta.bootstrap_host.clone(), port as u32)
            });

        if let Some((ref host, port)) = kv_transfer_params {
            debug!(
                bootstrap_host = %host,
                bootstrap_port = port,
                "vLLM PD (Mooncake): will inject kv_transfer_params into decode request"
            );
        } else {
            // Log at info level since this could indicate misconfiguration if user expects Mooncake
            // NIXL doesn't need kv_transfer_params (uses automatic prefix matching)
            // If user expects Mooncake but kv_connector wasn't discovered, they can manually set
            // labels: { "kv_connector": "MooncakeConnector" } in worker config
            let has_kv_connector = workers
                .prefill_worker()
                .map(|w| w.metadata().kv_connector.is_some())
                .unwrap_or(false);
            if has_kv_connector {
                debug!("vLLM PD (NIXL): using automatic prefix matching for KV transfer");
            } else {
                debug!(
                    "vLLM PD: no kv_connector detected (server may not support GetServerInfo kv fields). \
                     Assuming NIXL mode. For Mooncake, set labels.kv_connector=MooncakeConnector in worker config"
                );
            }
        }

        // Clone request and set max_tokens=1, stream=false for prefill
        let mut prefill_request = proto_request.clone_inner();
        prefill_request.set_max_tokens_for_prefill(1);
        prefill_request.set_stream(false);

        debug!(
            request_id = %prefill_request.request_id(),
            "vLLM PD: sending prefill request (max_tokens=1)"
        );

        // Send to prefill, wait for completion
        let mut prefill_stream = prefill_client
            .generate(prefill_request)
            .await
            .map_err(|e| {
                workers.record_outcome_prefill(false);
                error!(
                    function = "execute_sequential_pd",
                    error = %e,
                    "Prefill worker failed to start"
                );
                error::internal_error(
                    "prefill_worker_failed_to_start",
                    format!("Prefill worker failed to start: {}", e),
                )
            })?;

        // Drain prefill response (we just need to wait for completion)
        while let Some(result) = prefill_stream.next().await {
            match result {
                Ok(_response) => {
                    // Just consume the response, we use bootstrap info from worker metadata
                }
                Err(e) => {
                    workers.record_outcome_prefill(false);
                    error!(
                        function = "execute_sequential_pd",
                        error = %e,
                        "Prefill stream error"
                    );
                    return Err(error::internal_error(
                        "prefill_stream_error",
                        format!("Prefill stream error: {}", e),
                    ));
                }
            }
        }
        prefill_stream.mark_completed();
        workers.record_outcome_prefill(true);

        debug!("vLLM PD: prefill completed, sending decode request");

        // Clone original request and inject kv_transfer_params if present (Mooncake)
        let mut decode_request = proto_request;
        if let Some((remote_host, remote_port)) = kv_transfer_params {
            debug!(
                remote_host = %remote_host,
                remote_port = remote_port,
                "vLLM PD: injecting kv_transfer_params into decode request"
            );
            decode_request.set_kv_transfer_params(remote_host, remote_port);
        }

        // Send request to decode
        let decode_stream = decode_client.generate(decode_request).await.map_err(|e| {
            workers.record_outcome_decode(false);
            error!(
                function = "execute_sequential_pd",
                error = %e,
                "Decode worker failed to start"
            );
            error::internal_error(
                "decode_worker_failed_to_start",
                format!("Decode worker failed to start: {}", e),
            )
        })?;

        workers.record_outcome_decode(true);

        Ok(ExecutionResult::Single {
            stream: decode_stream,
        })
    }
}
