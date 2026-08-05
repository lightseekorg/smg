//! Backend client: the polymorphism point over a worker's transport.
//!
//! A worker's backend is reached either via gRPC (the [`GrpcClient`] multiplexer
//! over SGLang/vLLM/TRT-LLM/MLX/TokenSpeed) or via a direct ZMQ connection to a
//! same-host engine — vLLM EngineCore or TokenSpeed ([`ZmqEngineClient`]).
//! `BackendClient` keeps those
//! first-class siblings — `GrpcClient` stays pure gRPC — while the execution
//! pipeline (which works against [`ProtoStream`]/[`ProtoGenerateRequest`]) is
//! shared unchanged.

use openai_protocol::{
    chat::ChatCompletionRequest, completion::CompletionRequest, generate::GenerateRequest,
    messages::CreateMessageRequest, worker::WorkerLoadResponse,
};
use smg_grpc_client::{
    common_proto, tokenizer_bundle::StreamBundle, SglangSchedulerClient, TokenSpeedSchedulerClient,
    VllmEngineClient,
};

use crate::{
    routers::grpc::{
        client::{
            GenerateRequestBuildOptions, GrpcClient, HealthCheckResponse, ModelInfo, ServerInfo,
        },
        proto_wrapper::{
            finish_tokenspeed_request, finish_vllm_request, ProtoEmbedComplete, ProtoEmbedRequest,
            ProtoGenerateRequest, ProtoStream,
        },
        zmq_client::ZmqEngineClient,
        MultimodalData,
    },
    worker::RuntimeType,
};

/// A backend connection: gRPC (any engine) or direct ZMQ (vLLM EngineCore or
/// TokenSpeed).
#[derive(Clone)]
pub enum BackendClient {
    Grpc(GrpcClient),
    Zmq(ZmqEngineClient),
}

impl BackendClient {
    /// Runtime type backing this client.
    pub fn runtime_type(&self) -> RuntimeType {
        match self {
            Self::Grpc(client) => client.runtime_type(),
            Self::Zmq(client) => client.runtime(),
        }
    }

    /// True if this is a direct-ZMQ backend (the engine receives token ids only
    /// and cannot match string stops itself).
    pub fn is_zmq(&self) -> bool {
        matches!(self, Self::Zmq(_))
    }

    /// Local liveness. gRPC has no cheap local flag (it uses a health RPC), so
    /// this reports `true` for gRPC; ZMQ reflects its connection liveness.
    pub fn is_alive(&self) -> bool {
        match self {
            Self::Grpc(_) => true,
            Self::Zmq(client) => client.is_alive(),
        }
    }

    /// Mutable SGLang client accessor. Only valid for a gRPC-SGLang backend;
    /// callers guard with a runtime/sglang check.
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees an SGLang gRPC backend"
    )]
    pub fn as_sglang_mut(&mut self) -> &mut SglangSchedulerClient {
        match self {
            Self::Grpc(client) => client.as_sglang_mut(),
            Self::Zmq(_) => panic!("Expected SGLang client, got ZMQ backend"),
        }
    }

    pub async fn health_check(&self) -> Result<HealthCheckResponse, tonic::Status> {
        match self {
            Self::Grpc(client) => client.health_check().await,
            Self::Zmq(client) => {
                let resp = client.health_check();
                Ok(HealthCheckResponse {
                    healthy: resp.healthy,
                    message: resp.message,
                })
            }
        }
    }

    pub async fn get_model_info(&self) -> Result<ModelInfo, tonic::Status> {
        match self {
            Self::Grpc(client) => client.get_model_info().await,
            Self::Zmq(client) => Ok(client.get_model_info()),
        }
    }

    pub async fn get_server_info(&self) -> Result<ServerInfo, tonic::Status> {
        match self {
            Self::Grpc(client) => client.get_server_info().await,
            Self::Zmq(client) => Ok(client.get_server_info()),
        }
    }

    pub async fn get_loads(&self) -> Result<WorkerLoadResponse, tonic::Status> {
        match self {
            Self::Grpc(client) => client.get_loads().await,
            Self::Zmq(client) => Ok(client.get_loads()),
        }
    }

    pub async fn flush_cache(
        &self,
        timeout_s: f32,
    ) -> Result<common_proto::FlushCacheResponse, tonic::Status> {
        match self {
            Self::Grpc(client) => client.flush_cache(timeout_s).await,
            Self::Zmq(_) => Err(tonic::Status::unimplemented(
                "FlushCache not supported over ZMQ",
            )),
        }
    }

    pub async fn start_profile(
        &self,
        req: common_proto::StartProfileRequest,
    ) -> Result<common_proto::ProfileResponse, tonic::Status> {
        match self {
            Self::Grpc(client) => client.start_profile(req).await,
            Self::Zmq(_) => Err(tonic::Status::unimplemented(
                "StartProfile not supported over ZMQ",
            )),
        }
    }

    pub async fn stop_profile(&self) -> Result<common_proto::ProfileResponse, tonic::Status> {
        match self {
            Self::Grpc(client) => client.stop_profile().await,
            Self::Zmq(_) => Err(tonic::Status::unimplemented(
                "StopProfile not supported over ZMQ",
            )),
        }
    }

    pub async fn subscribe_kv_events(
        &self,
        start_seq: u64,
    ) -> Result<tonic::Streaming<common_proto::KvEventBatch>, tonic::Status> {
        match self {
            Self::Grpc(client) => client.subscribe_kv_events(start_seq).await,
            Self::Zmq(_) => Err(tonic::Status::unimplemented(
                "SubscribeKvEvents not supported over ZMQ",
            )),
        }
    }

    pub async fn get_tokenizer(
        &self,
    ) -> Result<StreamBundle, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            Self::Grpc(client) => client.get_tokenizer().await,
            // EngineCore does not serve tokenizer artifacts over ZMQ; the
            // tokenizer is configured at worker registration instead.
            Self::Zmq(_) => Err("ZMQ backend does not serve a tokenizer bundle".into()),
        }
    }

    pub async fn generate(
        &mut self,
        req: ProtoGenerateRequest,
    ) -> Result<ProtoStream, tonic::Status> {
        match self {
            Self::Grpc(client) => client.generate(req).await,
            Self::Zmq(client) => Ok(ProtoStream::Zmq(client.generate(req).await?)),
        }
    }

    pub async fn embed(
        &mut self,
        req: ProtoEmbedRequest,
    ) -> Result<ProtoEmbedComplete, tonic::Status> {
        match self {
            Self::Grpc(client) => client.embed(req).await,
            Self::Zmq(_) => Err(tonic::Status::unimplemented(
                "ZMQ backend does not support embedding yet",
            )),
        }
    }

    #[expect(
        clippy::unreachable,
        reason = "assembly stage guarantees matching MultimodalData variant for each backend"
    )]
    pub fn build_chat_request(
        &self,
        request_id: String,
        body: &ChatCompletionRequest,
        processed_text: String,
        token_ids: Vec<u32>,
        options: GenerateRequestBuildOptions,
    ) -> Result<ProtoGenerateRequest, String> {
        match self {
            Self::Grpc(client) => {
                client.build_chat_request(request_id, body, processed_text, token_ids, options)
            }
            // A ZMQ backend speaks vLLM EngineCore or TokenSpeed directly; build
            // the native request for its runtime, mirroring the gRPC per-engine
            // dispatch in `GrpcClient::build_chat_request`.
            Self::Zmq(client) => match client.runtime() {
                RuntimeType::TokenSpeed => {
                    let tokenspeed_mm = options.multimodal_inputs.map(|mm| match mm {
                        MultimodalData::TokenSpeed(data) => data.into_proto(true),
                        _ => unreachable!("caller guarantees matching variant"),
                    });
                    finish_tokenspeed_request(tokenspeed_mm, |mm| {
                        TokenSpeedSchedulerClient::build_generate_request_from_chat(
                            request_id,
                            body,
                            processed_text,
                            token_ids,
                            mm,
                            options.tool_constraints,
                        )
                    })
                }
                _ => {
                    let vllm_mm = options.multimodal_inputs.map(|mm| match mm {
                        MultimodalData::Vllm(data) => data.into_proto(),
                        _ => unreachable!("caller guarantees matching variant"),
                    });
                    finish_vllm_request(vllm_mm, |mm| {
                        VllmEngineClient::build_generate_request_from_chat(
                            request_id,
                            body,
                            processed_text,
                            token_ids,
                            mm,
                            options.tool_constraints,
                        )
                    })
                }
            },
        }
    }

    #[expect(
        clippy::unreachable,
        reason = "assembly stage guarantees matching MultimodalData variant for each backend"
    )]
    pub fn build_messages_request(
        &self,
        request_id: String,
        body: &CreateMessageRequest,
        processed_text: String,
        token_ids: Vec<u32>,
        options: GenerateRequestBuildOptions,
    ) -> Result<ProtoGenerateRequest, String> {
        match self {
            Self::Grpc(client) => {
                client.build_messages_request(request_id, body, processed_text, token_ids, options)
            }
            // Mirrors the gRPC per-engine dispatch: build the request natively for
            // the ZMQ backend's runtime (vLLM EngineCore or TokenSpeed).
            Self::Zmq(client) => match client.runtime() {
                RuntimeType::TokenSpeed => {
                    let tokenspeed_mm = options.multimodal_inputs.map(|mm| match mm {
                        MultimodalData::TokenSpeed(data) => data.into_proto(true),
                        _ => unreachable!("caller guarantees matching variant"),
                    });
                    finish_tokenspeed_request(tokenspeed_mm, |mm| {
                        TokenSpeedSchedulerClient::build_generate_request_from_messages(
                            request_id,
                            body,
                            processed_text,
                            token_ids,
                            mm,
                            options.tool_constraints,
                        )
                    })
                }
                _ => {
                    let vllm_mm = options.multimodal_inputs.map(|mm| match mm {
                        MultimodalData::Vllm(data) => data.into_proto(),
                        _ => unreachable!("caller guarantees matching variant"),
                    });
                    finish_vllm_request(vllm_mm, |mm| {
                        VllmEngineClient::build_generate_request_from_messages(
                            request_id,
                            body,
                            processed_text,
                            token_ids,
                            mm,
                            options.tool_constraints,
                        )
                    })
                }
            },
        }
    }

    pub fn build_completion_request(
        &self,
        request_id: String,
        body: &CompletionRequest,
        original_text: String,
        token_ids: Vec<u32>,
    ) -> Result<ProtoGenerateRequest, String> {
        match self {
            Self::Grpc(client) => {
                client.build_completion_request(request_id, body, original_text, token_ids)
            }
            Self::Zmq(client) => match client.runtime() {
                RuntimeType::TokenSpeed => {
                    let req = TokenSpeedSchedulerClient::build_generate_request_from_completion(
                        request_id,
                        body,
                        original_text,
                        token_ids,
                    )?;
                    Ok(ProtoGenerateRequest::TokenSpeed(Box::new(req)))
                }
                _ => {
                    let req = VllmEngineClient::build_generate_request_from_completion(
                        request_id,
                        body,
                        original_text,
                        token_ids,
                    )?;
                    Ok(ProtoGenerateRequest::Vllm(Box::new(req)))
                }
            },
        }
    }

    pub fn build_generate_request(
        &self,
        request_id: String,
        body: &GenerateRequest,
        original_text: Option<String>,
        token_ids: Vec<u32>,
    ) -> Result<ProtoGenerateRequest, String> {
        match self {
            Self::Grpc(client) => {
                client.build_generate_request(request_id, body, original_text, token_ids)
            }
            Self::Zmq(client) => match client.runtime() {
                RuntimeType::TokenSpeed => {
                    let req = TokenSpeedSchedulerClient::build_plain_generate_request(
                        request_id,
                        body,
                        original_text,
                        token_ids,
                    )?;
                    Ok(ProtoGenerateRequest::TokenSpeed(Box::new(req)))
                }
                _ => {
                    let req = VllmEngineClient::build_plain_generate_request(
                        request_id,
                        body,
                        original_text,
                        token_ids,
                    )?;
                    Ok(ProtoGenerateRequest::Vllm(Box::new(req)))
                }
            },
        }
    }
}
