//! Backend client: the polymorphism point over a worker's transport.
//!
//! A worker's backend is reached via the [`GrpcClient`] multiplexer (over
//! SGLang/vLLM/TRT-LLM/MLX/TokenSpeed). `BackendClient` is the seam the
//! execution pipeline works against so that additional transports (a direct
//! ZMQ connection to a same-host vLLM EngineCore) can be added as sibling
//! variants without the pipeline — which operates on [`ProtoStream`] /
//! [`ProtoGenerateRequest`] — having to change.
//!
//! Today it wraps only gRPC; every method delegates to the inner client, so
//! this is a behavior-neutral seam.

use openai_protocol::{
    chat::ChatCompletionRequest, completion::CompletionRequest, generate::GenerateRequest,
    messages::CreateMessageRequest, worker::WorkerLoadResponse,
};
use smg_grpc_client::{common_proto, tokenizer_bundle::StreamBundle, SglangSchedulerClient};

use crate::routers::grpc::{
    client::{GenerateRequestBuildOptions, GrpcClient, HealthCheckResponse, ModelInfo, ServerInfo},
    proto_wrapper::{ProtoEmbedComplete, ProtoEmbedRequest, ProtoGenerateRequest, ProtoStream},
};

/// A worker's backend connection. Currently gRPC-only; a direct-ZMQ variant is
/// added alongside `Grpc` in a follow-up.
#[derive(Clone)]
pub enum BackendClient {
    Grpc(GrpcClient),
}

impl BackendClient {
    /// Runtime type backing this client.
    pub fn runtime_type(&self) -> crate::worker::RuntimeType {
        match self {
            Self::Grpc(client) => client.runtime_type(),
        }
    }

    /// True if this backend speaks the vLLM protocol.
    pub fn is_vllm(&self) -> bool {
        match self {
            Self::Grpc(client) => client.is_vllm(),
        }
    }

    /// Mutable SGLang client accessor. Only valid for a gRPC-SGLang backend;
    /// callers guard with a runtime/sglang check.
    pub fn as_sglang_mut(&mut self) -> &mut SglangSchedulerClient {
        match self {
            Self::Grpc(client) => client.as_sglang_mut(),
        }
    }

    pub async fn health_check(&self) -> Result<HealthCheckResponse, tonic::Status> {
        match self {
            Self::Grpc(client) => client.health_check().await,
        }
    }

    pub async fn get_model_info(&self) -> Result<ModelInfo, tonic::Status> {
        match self {
            Self::Grpc(client) => client.get_model_info().await,
        }
    }

    pub async fn get_server_info(&self) -> Result<ServerInfo, tonic::Status> {
        match self {
            Self::Grpc(client) => client.get_server_info().await,
        }
    }

    pub async fn get_loads(&self) -> Result<WorkerLoadResponse, tonic::Status> {
        match self {
            Self::Grpc(client) => client.get_loads().await,
        }
    }

    pub async fn flush_cache(
        &self,
        timeout_s: f32,
    ) -> Result<common_proto::FlushCacheResponse, tonic::Status> {
        match self {
            Self::Grpc(client) => client.flush_cache(timeout_s).await,
        }
    }

    pub async fn start_profile(
        &self,
        req: common_proto::StartProfileRequest,
    ) -> Result<common_proto::ProfileResponse, tonic::Status> {
        match self {
            Self::Grpc(client) => client.start_profile(req).await,
        }
    }

    pub async fn stop_profile(&self) -> Result<common_proto::ProfileResponse, tonic::Status> {
        match self {
            Self::Grpc(client) => client.stop_profile().await,
        }
    }

    pub async fn subscribe_kv_events(
        &self,
        start_seq: u64,
    ) -> Result<tonic::Streaming<common_proto::KvEventBatch>, tonic::Status> {
        match self {
            Self::Grpc(client) => client.subscribe_kv_events(start_seq).await,
        }
    }

    pub async fn get_tokenizer(
        &self,
    ) -> Result<StreamBundle, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            Self::Grpc(client) => client.get_tokenizer().await,
        }
    }

    pub async fn generate(
        &mut self,
        req: ProtoGenerateRequest,
    ) -> Result<ProtoStream, tonic::Status> {
        match self {
            Self::Grpc(client) => client.generate(req).await,
        }
    }

    pub async fn embed(
        &mut self,
        req: ProtoEmbedRequest,
    ) -> Result<ProtoEmbedComplete, tonic::Status> {
        match self {
            Self::Grpc(client) => client.embed(req).await,
        }
    }

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
        }
    }

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
        }
    }
}
