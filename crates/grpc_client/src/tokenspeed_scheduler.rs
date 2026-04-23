//! gRPC client for the TokenSpeed scheduler service.
//!
//! Wire-level identity (package ``tokenspeed.grpc.scheduler``, service
//! ``TokenSpeedScheduler``) is distinct from SGLang — see
//! ``proto/tokenspeed_scheduler.proto``. Message types are currently
//! imported from the SGLang proto (the two backends happen to share their
//! tokenized-request and sampling-param shapes today), so this wrapper
//! reuses ``crate::sglang_scheduler::proto`` for all request / response
//! types and only implements the thin layer that dispatches to the
//! ``TokenSpeedScheduler`` service. When the two protocols diverge, the
//! shared message types should be split into this module (or a
//! ``tokenspeed_proto`` submodule) and the call sites updated.
//!
//! Also reuses ``SglangSchedulerClient``'s request-builder helpers
//! (``build_generate_request_from_chat``, ``build_grpc_sampling_params_*``,
//! etc.) — those are pure conversions from openai-protocol types to the
//! shared proto types, so duplicating them would be busy-work.

use std::{sync::Arc, time::Duration};

use openai_protocol::{
    chat::ChatCompletionRequest, completion::CompletionRequest, generate::GenerateRequest,
    messages::CreateMessageRequest, responses::ResponsesRequest,
};
use tonic::{transport::Channel, Request};
use tracing::{debug, warn};

use crate::{
    sglang_scheduler::{proto, AbortDispatcher, AbortOnDropStream, SglangSchedulerClient},
    BoxedTraceInjector, NoopTraceInjector,
};

#[expect(clippy::allow_attributes)]
pub mod tokenspeed_proto {
    #![allow(clippy::all, clippy::absolute_paths, unused_qualifications)]
    // ``build.rs`` writes the tokenspeed-specific generated stub to a
    // dedicated sub-OUT_DIR so it doesn't overwrite the SGLang stub (see
    // the pass-3 ``out_dir`` override there). ``include_proto!`` only
    // looks in the top-level ``OUT_DIR``, so we ``include!`` the file by
    // explicit path instead.
    include!(concat!(
        env!("OUT_DIR"),
        "/tokenspeed/tokenspeed.grpc.scheduler.rs"
    ));
}

/// gRPC client for the TokenSpeed scheduler.
///
/// Thin wrapper around the tonic-generated ``TokenSpeedSchedulerClient``
/// stub. RPC methods mirror [`crate::SglangSchedulerClient`] and accept /
/// return the shared ``crate::sglang_scheduler::proto`` message types.
#[derive(Clone)]
pub struct TokenSpeedSchedulerClient {
    client: tokenspeed_proto::token_speed_scheduler_client::TokenSpeedSchedulerClient<Channel>,
    trace_injector: BoxedTraceInjector,
}

impl TokenSpeedSchedulerClient {
    pub async fn connect(endpoint: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Self::connect_with_trace_injector(endpoint, Arc::new(NoopTraceInjector)).await
    }

    pub async fn connect_with_trace_injector(
        endpoint: &str,
        trace_injector: BoxedTraceInjector,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        debug!("Connecting to TokenSpeed scheduler at {}", endpoint);

        let http_endpoint = if let Some(addr) = endpoint.strip_prefix("grpc://") {
            format!("http://{addr}")
        } else {
            endpoint.to_string()
        };

        // Same channel knobs as SglangSchedulerClient — these are independent
        // of the service being called and proven load-appropriate in prod.
        let channel = Channel::from_shared(http_endpoint)?
            .http2_keep_alive_interval(Duration::from_secs(30))
            .keep_alive_timeout(Duration::from_secs(10))
            .keep_alive_while_idle(true)
            .tcp_keepalive(Some(Duration::from_secs(60)))
            .tcp_nodelay(true)
            .http2_adaptive_window(true)
            .initial_stream_window_size(Some(16 * 1024 * 1024))
            .initial_connection_window_size(Some(32 * 1024 * 1024))
            .connect()
            .await?;

        let client =
            tokenspeed_proto::token_speed_scheduler_client::TokenSpeedSchedulerClient::new(channel);

        Ok(Self {
            client,
            trace_injector,
        })
    }

    #[must_use]
    pub fn with_trace_injector(mut self, trace_injector: BoxedTraceInjector) -> Self {
        self.trace_injector = trace_injector;
        self
    }

    pub async fn generate(
        &self,
        req: proto::GenerateRequest,
    ) -> Result<AbortOnDropStream, tonic::Status> {
        let request_id = req.request_id.clone();
        let mut client = self.client.clone();
        let mut request = Request::new(req);

        if let Err(e) = self.trace_injector.inject(request.metadata_mut()) {
            warn!("Failed to inject trace context: {}", e);
        }

        let response = client.generate(request).await?;

        Ok(AbortOnDropStream::new(
            response.into_inner(),
            request_id,
            tokenspeed_abort_dispatcher(self.clone()),
        ))
    }

    pub async fn embed(
        &self,
        req: proto::EmbedRequest,
    ) -> Result<proto::EmbedResponse, tonic::Status> {
        let mut client = self.client.clone();
        let mut request = Request::new(req);

        if let Err(e) = self.trace_injector.inject(request.metadata_mut()) {
            warn!("Failed to inject trace context: {}", e);
        }

        let response = client.embed(request).await?;
        Ok(response.into_inner())
    }

    pub async fn health_check(&self) -> Result<proto::HealthCheckResponse, tonic::Status> {
        debug!("Sending health check request");
        let request = Request::new(proto::HealthCheckRequest {});
        let mut client = self.client.clone();
        let response = client.health_check(request).await?;
        Ok(response.into_inner())
    }

    pub async fn abort_request(
        &self,
        request_id: String,
        reason: String,
    ) -> Result<(), tonic::Status> {
        debug!(
            "Sending abort request for {} (reason: {})",
            request_id, reason
        );
        let request = Request::new(proto::AbortRequest {
            request_id: request_id.clone(),
            reason,
        });
        let mut client = self.client.clone();
        let response = client.abort(request).await?;
        debug!(
            "Abort response for {}: success={}, message={}",
            request_id,
            response.get_ref().success,
            response.get_ref().message
        );
        Ok(())
    }

    pub async fn get_model_info(&self) -> Result<proto::GetModelInfoResponse, tonic::Status> {
        let request = Request::new(proto::GetModelInfoRequest {});
        let mut client = self.client.clone();
        let response = client.get_model_info(request).await?;
        Ok(response.into_inner())
    }

    pub async fn get_server_info(&self) -> Result<proto::GetServerInfoResponse, tonic::Status> {
        let request = Request::new(proto::GetServerInfoRequest {});
        let mut client = self.client.clone();
        let response = client.get_server_info(request).await?;
        Ok(response.into_inner())
    }

    pub async fn get_loads(
        &self,
        include: Vec<String>,
    ) -> Result<proto::GetLoadsResponse, tonic::Status> {
        let request = Request::new(proto::GetLoadsRequest {
            dp_rank: None,
            include,
        });
        let mut client = self.client.clone();
        let response = client.get_loads(request).await?;
        Ok(response.into_inner())
    }

    crate::impl_get_tokenizer!();
    crate::impl_subscribe_kv_events!();

    // ── Request builders ──────────────────────────────────────────────
    //
    // TokenSpeed reuses SGLang's on-wire message types today (see
    // ``proto/tokenspeed_scheduler.proto`` — messages are imported from
    // ``sglang_scheduler.proto``). These builders delegate to the SGLang
    // sampling-param helpers (shared ``pub(crate)`` in ``sglang_scheduler.rs``)
    // and construct the same ``proto::GenerateRequest`` / ``proto::EmbedRequest``
    // shape. When the two backends diverge, copy-and-specialize rather
    // than adding conditional branches in SGLang's builders.

    #[expect(
        clippy::unused_self,
        reason = "receiver kept for API parity with SglangSchedulerClient"
    )]
    pub fn build_embed_request(
        &self,
        request_id: String,
        original_text: Option<String>,
        token_ids: Vec<u32>,
    ) -> proto::EmbedRequest {
        proto::EmbedRequest {
            request_id,
            tokenized: Some(proto::TokenizedInput {
                original_text: original_text.unwrap_or_default(),
                input_ids: token_ids,
            }),
            ..Default::default()
        }
    }

    #[expect(
        clippy::unused_self,
        reason = "receiver kept for API parity with SglangSchedulerClient"
    )]
    pub fn build_generate_request_from_chat(
        &self,
        request_id: String,
        body: &ChatCompletionRequest,
        processed_text: String,
        token_ids: Vec<u32>,
        multimodal_inputs: Option<proto::MultimodalInputs>,
        tool_call_constraint: Option<(String, String)>,
    ) -> Result<proto::GenerateRequest, String> {
        let sampling_params = SglangSchedulerClient::build_grpc_sampling_params_from_chat(
            body,
            tool_call_constraint,
        )?;
        Ok(proto::GenerateRequest {
            request_id,
            tokenized: Some(proto::TokenizedInput {
                original_text: processed_text,
                input_ids: token_ids,
            }),
            mm_inputs: multimodal_inputs,
            sampling_params: Some(sampling_params),
            return_logprob: body.logprobs,
            logprob_start_len: -1,
            top_logprobs_num: body.top_logprobs.unwrap_or(0) as i32,
            return_hidden_states: body.return_hidden_states,
            stream: body.stream,
            ..Default::default()
        })
    }

    #[expect(
        clippy::unused_self,
        reason = "receiver kept for API parity with SglangSchedulerClient"
    )]
    pub fn build_plain_generate_request(
        &self,
        request_id: String,
        body: &GenerateRequest,
        original_text: Option<String>,
        token_ids: Vec<u32>,
    ) -> Result<proto::GenerateRequest, String> {
        let sampling_params =
            SglangSchedulerClient::build_sampling_params_from_plain(body.sampling_params.as_ref())?;
        Ok(proto::GenerateRequest {
            request_id,
            tokenized: Some(proto::TokenizedInput {
                original_text: original_text.unwrap_or_default(),
                input_ids: token_ids,
            }),
            sampling_params: Some(sampling_params),
            return_logprob: body.return_logprob.unwrap_or(false),
            logprob_start_len: body.logprob_start_len.unwrap_or(-1),
            top_logprobs_num: body.top_logprobs_num.unwrap_or(0),
            token_ids_logprob: body.token_ids_logprob.clone().unwrap_or_default(),
            return_hidden_states: body.return_hidden_states,
            stream: body.stream,
            log_metrics: body.log_metrics,
            ..Default::default()
        })
    }

    #[expect(
        clippy::unused_self,
        reason = "receiver kept for API parity with SglangSchedulerClient"
    )]
    pub fn build_generate_request_from_responses(
        &self,
        request_id: String,
        body: &ResponsesRequest,
        processed_text: String,
        token_ids: Vec<u32>,
        constraint: Option<(String, String)>,
    ) -> Result<proto::GenerateRequest, String> {
        let sampling_params =
            SglangSchedulerClient::build_grpc_sampling_params_from_responses(body, constraint)?;
        Ok(proto::GenerateRequest {
            request_id,
            tokenized: Some(proto::TokenizedInput {
                original_text: processed_text,
                input_ids: token_ids,
            }),
            sampling_params: Some(sampling_params),
            stream: body.stream.unwrap_or(false),
            ..Default::default()
        })
    }

    #[expect(
        clippy::unused_self,
        reason = "receiver kept for API parity with SglangSchedulerClient"
    )]
    pub fn build_generate_request_from_messages(
        &self,
        request_id: String,
        body: &CreateMessageRequest,
        processed_text: String,
        token_ids: Vec<u32>,
        multimodal_inputs: Option<proto::MultimodalInputs>,
        tool_call_constraint: Option<(String, String)>,
    ) -> Result<proto::GenerateRequest, String> {
        let sampling_params = SglangSchedulerClient::build_grpc_sampling_params_from_messages(
            body,
            tool_call_constraint,
        )?;
        Ok(proto::GenerateRequest {
            request_id,
            tokenized: Some(proto::TokenizedInput {
                original_text: processed_text,
                input_ids: token_ids,
            }),
            mm_inputs: multimodal_inputs,
            sampling_params: Some(sampling_params),
            stream: body.stream.unwrap_or(false),
            ..Default::default()
        })
    }

    #[expect(
        clippy::unused_self,
        reason = "receiver kept for API parity with SglangSchedulerClient"
    )]
    pub fn build_generate_request_from_completion(
        &self,
        request_id: String,
        body: &CompletionRequest,
        original_text: String,
        token_ids: Vec<u32>,
    ) -> Result<proto::GenerateRequest, String> {
        let sampling_params =
            SglangSchedulerClient::build_grpc_sampling_params_from_completion(body)?;
        Ok(proto::GenerateRequest {
            request_id,
            tokenized: Some(proto::TokenizedInput {
                original_text,
                input_ids: token_ids,
            }),
            sampling_params: Some(sampling_params),
            return_logprob: body.logprobs.is_some(),
            logprob_start_len: -1,
            top_logprobs_num: body.logprobs.unwrap_or(0) as i32,
            stream: body.stream,
            ..Default::default()
        })
    }
}

/// Abort dispatcher for TokenSpeed streams — sends the abort RPC over the
/// TokenSpeed service instead of SGLang's. Structurally identical to
/// ``sglang_scheduler::sglang_abort_dispatcher``; kept separate because the
/// two methods dispatch to different tonic stubs.
fn tokenspeed_abort_dispatcher(client: TokenSpeedSchedulerClient) -> AbortDispatcher {
    Arc::new(move |request_id: String| {
        let client = client.clone();
        let request_id_for_log = request_id.clone();
        #[expect(
            clippy::disallowed_methods,
            reason = "fire-and-forget abort on Drop is intentional"
        )]
        tokio::spawn(async move {
            if let Err(e) = client
                .abort_request(request_id, "Stream dropped".to_string())
                .await
            {
                warn!(
                    "Failed to send abort on drop for request {}: {}",
                    request_id_for_log, e
                );
            }
        });
    })
}
