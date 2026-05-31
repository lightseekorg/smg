//! Unified gRPC client wrapper for SGLang, vLLM, and TensorRT-LLM backends

use std::collections::HashMap;

use openai_protocol::{
    chat::ChatCompletionRequest, completion::CompletionRequest, generate::GenerateRequest,
    messages::CreateMessageRequest, worker::WorkerLoadResponse,
};
use serde_json::Value;
use smg_grpc_client::{
    tokenizer_bundle, tokenizer_bundle::StreamBundle, MlxEngineClient, SglangSchedulerClient,
    TokenSpeedSchedulerClient, TrtllmServiceClient, VllmEngineClient,
};

use crate::routers::grpc::{
    proto_wrapper::{ProtoEmbedComplete, ProtoEmbedRequest, ProtoGenerateRequest, ProtoStream},
    MultimodalData,
};

/// Health check response (common across backends)
#[derive(Debug, Clone)]
pub struct HealthCheckResponse {
    pub healthy: bool,
    pub message: String,
}

/// Wraps the per-backend gRPC clients. RPCs absent on a backend's wire
/// return `Status::unimplemented`.
#[derive(Clone)]
pub enum GrpcClient {
    Sglang(SglangSchedulerClient),
    Vllm(VllmEngineClient),
    Trtllm(TrtllmServiceClient),
    Mlx(MlxEngineClient),
    TokenSpeed(TokenSpeedSchedulerClient),
}

impl GrpcClient {
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_sglang() check"
    )]
    pub fn as_sglang(&self) -> &SglangSchedulerClient {
        match self {
            Self::Sglang(client) => client,
            _ => panic!("Expected SGLang client"),
        }
    }

    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_sglang() check"
    )]
    pub fn as_sglang_mut(&mut self) -> &mut SglangSchedulerClient {
        match self {
            Self::Sglang(client) => client,
            _ => panic!("Expected SGLang client"),
        }
    }

    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_vllm() check"
    )]
    pub fn as_vllm(&self) -> &VllmEngineClient {
        match self {
            Self::Vllm(client) => client,
            _ => panic!("Expected vLLM client"),
        }
    }

    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_vllm() check"
    )]
    pub fn as_vllm_mut(&mut self) -> &mut VllmEngineClient {
        match self {
            Self::Vllm(client) => client,
            _ => panic!("Expected vLLM client"),
        }
    }

    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_trtllm() check"
    )]
    pub fn as_trtllm(&self) -> &TrtllmServiceClient {
        match self {
            Self::Trtllm(client) => client,
            _ => panic!("Expected TensorRT-LLM client"),
        }
    }

    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_trtllm() check"
    )]
    pub fn as_trtllm_mut(&mut self) -> &mut TrtllmServiceClient {
        match self {
            Self::Trtllm(client) => client,
            _ => panic!("Expected TensorRT-LLM client"),
        }
    }

    pub fn is_sglang(&self) -> bool {
        matches!(self, Self::Sglang(_))
    }

    pub fn is_vllm(&self) -> bool {
        matches!(self, Self::Vllm(_))
    }

    pub fn is_trtllm(&self) -> bool {
        matches!(self, Self::Trtllm(_))
    }

    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_mlx() check"
    )]
    pub fn as_mlx(&self) -> &MlxEngineClient {
        match self {
            Self::Mlx(client) => client,
            _ => panic!("Expected MLX client"),
        }
    }

    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_mlx() check"
    )]
    pub fn as_mlx_mut(&mut self) -> &mut MlxEngineClient {
        match self {
            Self::Mlx(client) => client,
            _ => panic!("Expected MLX client"),
        }
    }

    pub fn is_mlx(&self) -> bool {
        matches!(self, Self::Mlx(_))
    }

    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_tokenspeed() check"
    )]
    pub fn as_tokenspeed(&self) -> &TokenSpeedSchedulerClient {
        match self {
            Self::TokenSpeed(client) => client,
            _ => panic!("Expected TokenSpeed client"),
        }
    }

    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_tokenspeed() check"
    )]
    pub fn as_tokenspeed_mut(&mut self) -> &mut TokenSpeedSchedulerClient {
        match self {
            Self::TokenSpeed(client) => client,
            _ => panic!("Expected TokenSpeed client"),
        }
    }

    pub fn is_tokenspeed(&self) -> bool {
        matches!(self, Self::TokenSpeed(_))
    }

    pub async fn connect(
        url: &str,
        runtime_type: &str,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        match runtime_type {
            "sglang" => Ok(Self::Sglang(SglangSchedulerClient::connect(url).await?)),
            "vllm" => Ok(Self::Vllm(VllmEngineClient::connect(url).await?)),
            "trtllm" | "tensorrt-llm" => Ok(Self::Trtllm(TrtllmServiceClient::connect(url).await?)),
            "mlx" => Ok(Self::Mlx(MlxEngineClient::connect(url).await?)),
            "tokenspeed" => Ok(Self::TokenSpeed(
                TokenSpeedSchedulerClient::connect(url).await?,
            )),
            _ => Err(format!("Unknown runtime type: {runtime_type}").into()),
        }
    }

    pub async fn health_check(&self) -> Result<HealthCheckResponse, tonic::Status> {
        match self {
            Self::Sglang(client) => {
                let resp = client.health_check().await?;
                Ok(HealthCheckResponse {
                    healthy: resp.healthy,
                    message: resp.message,
                })
            }
            Self::Vllm(client) => {
                let resp = client.health_check().await?;
                Ok(HealthCheckResponse {
                    healthy: resp.healthy,
                    message: resp.message,
                })
            }
            Self::Trtllm(client) => {
                let resp = client.health_check().await?;
                let healthy = resp.status.to_lowercase().contains("ok")
                    || resp.status.to_lowercase().contains("healthy");
                Ok(HealthCheckResponse {
                    healthy,
                    message: resp.status,
                })
            }
            Self::Mlx(client) => {
                let resp = client.health_check().await?;
                Ok(HealthCheckResponse {
                    healthy: resp.healthy,
                    message: resp.message,
                })
            }
            Self::TokenSpeed(client) => {
                let resp = client.health_check().await?;
                Ok(HealthCheckResponse {
                    healthy: resp.healthy,
                    message: resp.message,
                })
            }
        }
    }

    pub async fn get_model_info(&self) -> Result<ModelInfo, tonic::Status> {
        match self {
            Self::Sglang(client) => Ok(ModelInfo::Sglang(Box::new(client.get_model_info().await?))),
            Self::Vllm(client) => Ok(ModelInfo::Vllm(client.get_model_info().await?)),
            Self::Trtllm(client) => Ok(ModelInfo::Trtllm(client.get_model_info().await?)),
            Self::Mlx(client) => Ok(ModelInfo::Mlx(client.get_model_info().await?)),
            Self::TokenSpeed(client) => Ok(ModelInfo::TokenSpeed(Box::new(
                client.get_model_info().await?,
            ))),
        }
    }

    /// Get the full load response from the backend.
    /// Returns `Unimplemented` for backends without scheduler load metrics.
    pub async fn get_loads(&self) -> Result<WorkerLoadResponse, tonic::Status> {
        match self {
            Self::Sglang(client) => {
                let resp = client.get_loads(vec!["core".to_string()]).await?;
                Ok(WorkerLoadResponse::from(resp))
            }
            Self::TokenSpeed(client) => {
                let resp = client.get_loads(vec!["core".to_string()]).await?;
                Ok(WorkerLoadResponse::from(resp))
            }
            _ => Err(tonic::Status::unimplemented(
                "GetLoads RPC not supported for this backend",
            )),
        }
    }

    /// Subscribe to KV cache events. Returns `Unimplemented` on backends
    /// without KV-event streaming.
    pub async fn subscribe_kv_events(
        &self,
        start_seq: u64,
    ) -> Result<tonic::Streaming<smg_grpc_client::common_proto::KvEventBatch>, tonic::Status> {
        match self {
            Self::Sglang(client) => client.subscribe_kv_events(start_seq).await,
            Self::Vllm(client) => client.subscribe_kv_events(start_seq).await,
            Self::Trtllm(client) => client.subscribe_kv_events(start_seq).await,
            Self::Mlx(_) => Err(tonic::Status::unimplemented(
                "SubscribeKvEvents RPC not supported for MLX backend",
            )),
            Self::TokenSpeed(_) => Err(tonic::Status::unimplemented(
                "SubscribeKvEvents RPC not supported for TokenSpeed backend",
            )),
        }
    }

    pub async fn get_server_info(&self) -> Result<ServerInfo, tonic::Status> {
        match self {
            Self::Sglang(client) => Ok(ServerInfo::Sglang(Box::new(
                client.get_server_info().await?,
            ))),
            Self::Vllm(client) => Ok(ServerInfo::Vllm(client.get_server_info().await?)),
            Self::Trtllm(client) => Ok(ServerInfo::Trtllm(client.get_server_info().await?)),
            Self::Mlx(client) => Ok(ServerInfo::Mlx(client.get_server_info().await?)),
            Self::TokenSpeed(client) => Ok(ServerInfo::TokenSpeed(Box::new(
                client.get_server_info().await?,
            ))),
        }
    }

    /// Fetch tokenizer bundle from backend runtime and validate integrity/safety.
    pub async fn get_tokenizer(
        &self,
    ) -> Result<StreamBundle, Box<dyn std::error::Error + Send + Sync>> {
        let bundle = match self {
            Self::Sglang(client) => client.get_tokenizer().await,
            Self::Vllm(client) => client.get_tokenizer().await,
            Self::Trtllm(client) => client.get_tokenizer().await,
            Self::Mlx(client) => client.get_tokenizer().await,
            Self::TokenSpeed(_) => {
                return Err(Box::new(tonic::Status::unimplemented(
                    "TokenSpeed backend does not support GetTokenizer RPC",
                )));
            }
        }?;

        tokenizer_bundle::validate_bundle_sha256(&bundle).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Tokenizer bundle SHA256 validation failed: {e}"),
            )
        })?;

        Ok(bundle)
    }

    /// Generate streaming response from request
    ///
    /// Dispatches to the appropriate backend client and wraps the result in ProtoStream.
    /// Returns `tonic::Status` on error so callers can inspect the gRPC status code directly.
    pub async fn generate(
        &mut self,
        req: ProtoGenerateRequest,
    ) -> Result<ProtoStream, tonic::Status> {
        match (self, req) {
            (Self::Sglang(client), ProtoGenerateRequest::Sglang(boxed_req)) => {
                let stream = client.generate(*boxed_req).await?;
                Ok(ProtoStream::Sglang(stream))
            }
            (Self::Vllm(client), ProtoGenerateRequest::Vllm(boxed_req)) => {
                let stream = client.generate(*boxed_req).await?;
                Ok(ProtoStream::Vllm(stream))
            }
            (Self::Trtllm(client), ProtoGenerateRequest::Trtllm(boxed_req)) => {
                let stream = client.generate(*boxed_req).await?;
                Ok(ProtoStream::Trtllm(stream))
            }
            (Self::Mlx(client), ProtoGenerateRequest::Mlx(boxed_req)) => {
                let stream = client.generate(*boxed_req).await?;
                Ok(ProtoStream::Mlx(stream))
            }
            (Self::TokenSpeed(client), ProtoGenerateRequest::TokenSpeed(boxed_req)) => {
                let stream = client.generate(*boxed_req).await?;
                Ok(ProtoStream::TokenSpeed(stream))
            }
            #[expect(
                clippy::panic,
                reason = "client and request types are always matched by construction in the pipeline"
            )]
            _ => panic!("Mismatched client and request types"),
        }
    }

    pub async fn embed(
        &mut self,
        req: ProtoEmbedRequest,
    ) -> Result<ProtoEmbedComplete, tonic::Status> {
        match (self, req) {
            (Self::Sglang(client), ProtoEmbedRequest::Sglang(boxed_req)) => {
                let resp = client.embed(*boxed_req).await?;
                Ok(ProtoEmbedComplete::Sglang(resp))
            }
            (Self::Vllm(client), ProtoEmbedRequest::Vllm(boxed_req)) => {
                let resp = client.embed(*boxed_req).await?;
                Ok(ProtoEmbedComplete::Vllm(resp))
            }
            (Self::TokenSpeed(_), _) => Err(tonic::Status::unimplemented(
                "TokenSpeed backend does not support embedding",
            )),
            (Self::Mlx(_), _) => Err(tonic::Status::unimplemented(
                "MLX backend does not support embedding",
            )),
            #[expect(
                clippy::panic,
                reason = "client and request types are always matched by construction in the pipeline"
            )]
            _ => panic!("Mismatched client and request types or unsupported embedding backend"),
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
        multimodal_inputs: Option<MultimodalData>,
        tool_constraints: Option<(String, String)>,
    ) -> Result<ProtoGenerateRequest, String> {
        match self {
            Self::Sglang(client) => {
                let sglang_mm = multimodal_inputs.map(|mm| match mm {
                    MultimodalData::Sglang(data) => data.into_proto(),
                    _ => unreachable!("caller guarantees matching variant"),
                });
                let req = client.build_generate_request_from_chat(
                    request_id,
                    body,
                    processed_text,
                    token_ids,
                    sglang_mm,
                    tool_constraints,
                )?;
                Ok(ProtoGenerateRequest::Sglang(Box::new(req)))
            }
            Self::Vllm(client) => {
                let vllm_mm = multimodal_inputs.map(|mm| match mm {
                    MultimodalData::Vllm(data) => data.into_proto(),
                    _ => unreachable!("caller guarantees matching variant"),
                });
                let req = client.build_generate_request_from_chat(
                    request_id,
                    body,
                    processed_text,
                    token_ids,
                    vllm_mm,
                    tool_constraints,
                )?;
                Ok(ProtoGenerateRequest::Vllm(Box::new(req)))
            }
            Self::Trtllm(client) => {
                let trtllm_mm = multimodal_inputs.map(|mm| match mm {
                    MultimodalData::Trtllm(data) => data.into_proto(),
                    _ => unreachable!("caller guarantees matching variant"),
                });
                let req = client.build_generate_request_from_chat(
                    request_id,
                    body,
                    processed_text,
                    token_ids,
                    trtllm_mm,
                    tool_constraints,
                )?;
                Ok(ProtoGenerateRequest::Trtllm(Box::new(req)))
            }
            // MLX: caller stage rejects multimodal before reaching this path.
            Self::Mlx(client) => {
                let req = client.build_generate_request_from_chat(
                    request_id,
                    body,
                    processed_text,
                    token_ids,
                    tool_constraints,
                )?;
                Ok(ProtoGenerateRequest::Mlx(Box::new(req)))
            }
            Self::TokenSpeed(client) => {
                let tokenspeed_mm = multimodal_inputs.map(|mm| match mm {
                    MultimodalData::TokenSpeed(data) => data.into_proto(),
                    _ => unreachable!("caller guarantees matching variant"),
                });
                let req = client.build_generate_request_from_chat(
                    request_id,
                    body,
                    processed_text,
                    token_ids,
                    tokenspeed_mm,
                    tool_constraints,
                )?;
                Ok(ProtoGenerateRequest::TokenSpeed(Box::new(req)))
            }
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
        multimodal_inputs: Option<MultimodalData>,
        tool_constraints: Option<(String, String)>,
    ) -> Result<ProtoGenerateRequest, String> {
        match self {
            Self::Sglang(client) => {
                let sglang_mm = multimodal_inputs.map(|mm| match mm {
                    MultimodalData::Sglang(data) => data.into_proto(),
                    _ => unreachable!("caller guarantees matching variant"),
                });
                let req = client.build_generate_request_from_messages(
                    request_id,
                    body,
                    processed_text,
                    token_ids,
                    sglang_mm,
                    tool_constraints,
                )?;
                Ok(ProtoGenerateRequest::Sglang(Box::new(req)))
            }
            Self::Vllm(client) => {
                let vllm_mm = multimodal_inputs.map(|mm| match mm {
                    MultimodalData::Vllm(data) => data.into_proto(),
                    _ => unreachable!("caller guarantees matching variant"),
                });
                let req = client.build_generate_request_from_messages(
                    request_id,
                    body,
                    processed_text,
                    token_ids,
                    vllm_mm,
                    tool_constraints,
                )?;
                Ok(ProtoGenerateRequest::Vllm(Box::new(req)))
            }
            Self::Trtllm(client) => {
                let trtllm_mm = multimodal_inputs.map(|mm| match mm {
                    MultimodalData::Trtllm(data) => data.into_proto(),
                    _ => unreachable!("caller guarantees matching variant"),
                });
                let req = client.build_generate_request_from_messages(
                    request_id,
                    body,
                    processed_text,
                    token_ids,
                    trtllm_mm,
                    tool_constraints,
                )?;
                Ok(ProtoGenerateRequest::Trtllm(Box::new(req)))
            }
            // MLX: caller stage rejects multimodal before reaching this path.
            Self::Mlx(client) => {
                let req = client.build_generate_request_from_messages(
                    request_id,
                    body,
                    processed_text,
                    token_ids,
                    tool_constraints,
                )?;
                Ok(ProtoGenerateRequest::Mlx(Box::new(req)))
            }
            Self::TokenSpeed(client) => {
                let tokenspeed_mm = multimodal_inputs.map(|mm| match mm {
                    MultimodalData::TokenSpeed(data) => data.into_proto(),
                    _ => unreachable!("caller guarantees matching variant"),
                });
                let req = client.build_generate_request_from_messages(
                    request_id,
                    body,
                    processed_text,
                    token_ids,
                    tokenspeed_mm,
                    tool_constraints,
                )?;
                Ok(ProtoGenerateRequest::TokenSpeed(Box::new(req)))
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
            Self::Sglang(client) => {
                let req = client.build_generate_request_from_completion(
                    request_id,
                    body,
                    original_text,
                    token_ids,
                )?;
                Ok(ProtoGenerateRequest::Sglang(Box::new(req)))
            }
            Self::Vllm(client) => {
                let req = client.build_generate_request_from_completion(
                    request_id,
                    body,
                    original_text,
                    token_ids,
                )?;
                Ok(ProtoGenerateRequest::Vllm(Box::new(req)))
            }
            Self::Trtllm(client) => {
                let req = client.build_generate_request_from_completion(
                    request_id,
                    body,
                    original_text,
                    token_ids,
                )?;
                Ok(ProtoGenerateRequest::Trtllm(Box::new(req)))
            }
            Self::Mlx(client) => {
                let req = client.build_generate_request_from_completion(
                    request_id,
                    body,
                    original_text,
                    token_ids,
                )?;
                Ok(ProtoGenerateRequest::Mlx(Box::new(req)))
            }
            Self::TokenSpeed(client) => {
                let req = client.build_generate_request_from_completion(
                    request_id,
                    body,
                    original_text,
                    token_ids,
                )?;
                Ok(ProtoGenerateRequest::TokenSpeed(Box::new(req)))
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
            Self::Sglang(client) => {
                let req = client.build_plain_generate_request(
                    request_id,
                    body,
                    original_text,
                    token_ids,
                )?;
                Ok(ProtoGenerateRequest::Sglang(Box::new(req)))
            }
            Self::Vllm(client) => {
                let req = client.build_plain_generate_request(
                    request_id,
                    body,
                    original_text,
                    token_ids,
                )?;
                Ok(ProtoGenerateRequest::Vllm(Box::new(req)))
            }
            Self::Trtllm(client) => {
                let req = client.build_plain_generate_request(
                    request_id,
                    body,
                    original_text,
                    token_ids,
                )?;
                Ok(ProtoGenerateRequest::Trtllm(Box::new(req)))
            }
            Self::Mlx(client) => {
                let req = client.build_plain_generate_request(
                    request_id,
                    body,
                    original_text,
                    token_ids,
                )?;
                Ok(ProtoGenerateRequest::Mlx(Box::new(req)))
            }
            Self::TokenSpeed(client) => {
                let req = client.build_plain_generate_request(
                    request_id,
                    body,
                    original_text,
                    token_ids,
                )?;
                Ok(ProtoGenerateRequest::TokenSpeed(Box::new(req)))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Metadata wrappers
// ---------------------------------------------------------------------------

pub enum ModelInfo {
    Sglang(Box<smg_grpc_client::sglang_proto::GetModelInfoResponse>),
    Vllm(smg_grpc_client::vllm_proto::GetModelInfoResponse),
    Trtllm(smg_grpc_client::trtllm_proto::GetModelInfoResponse),
    Mlx(smg_grpc_client::mlx_proto::GetModelInfoResponse),
    TokenSpeed(Box<smg_grpc_client::tokenspeed_proto::GetModelInfoResponse>),
}

pub enum ServerInfo {
    Sglang(Box<smg_grpc_client::sglang_proto::GetServerInfoResponse>),
    Vllm(smg_grpc_client::vllm_proto::GetServerInfoResponse),
    Trtllm(smg_grpc_client::trtllm_proto::GetServerInfoResponse),
    Mlx(smg_grpc_client::mlx_proto::GetServerInfoResponse),
    TokenSpeed(Box<smg_grpc_client::tokenspeed_proto::GetServerInfoResponse>),
}

impl ModelInfo {
    pub fn to_labels(&self) -> HashMap<String, String> {
        match self {
            ModelInfo::Sglang(info) => flat_labels(info),
            ModelInfo::Vllm(info) => flat_labels(info),
            ModelInfo::Trtllm(info) => flat_labels(info),
            ModelInfo::Mlx(info) => flat_labels(info),
            ModelInfo::TokenSpeed(info) => flat_labels(info),
        }
    }
}

impl ServerInfo {
    /// Convert to labels. SGLang needs special handling because its `server_args`
    /// is a `prost_types::Struct` (not Serialize). vLLM/TRT-LLM are plain structs.
    pub fn to_labels(&self) -> HashMap<String, String> {
        match self {
            ServerInfo::Sglang(info) => {
                let mut labels = HashMap::new();
                if let Some(ref args) = info.server_args {
                    pick_prost_fields(&mut labels, args, SGLANG_GRPC_KEYS);
                }
                if !info.sglang_version.is_empty() {
                    labels.insert("version".to_string(), info.sglang_version.clone());
                }
                labels
            }
            ServerInfo::Vllm(info) => flat_labels(info),
            ServerInfo::Trtllm(info) => flat_labels(info),
            ServerInfo::Mlx(info) => flat_labels(info),
            ServerInfo::TokenSpeed(info) => {
                let mut labels = HashMap::new();
                if let Some(ref args) = info.server_args {
                    pick_prost_fields(&mut labels, args, TOKENSPEED_GRPC_KEYS);
                }
                if !info.tokenspeed_version.is_empty() {
                    labels.insert("version".to_string(), info.tokenspeed_version.clone());
                }
                labels
            }
        }
    }

    /// Convert to JSON. For the potentially very large `server_args` protobuf
    /// `Struct`: filter it down to the curated `*_GRPC_KEYS` subset
    pub fn to_public_json(&self) -> Value {
        fn prost_number_to_json(number: f64) -> Value {
            const MAX_SAFE_INTEGER_F64: f64 = 9_007_199_254_740_991.0;
            const MIN_SAFE_INTEGER_F64: f64 = -9_007_199_254_740_991.0;

            if number.is_finite()
                && number.fract() == 0.0
                && (MIN_SAFE_INTEGER_F64..=MAX_SAFE_INTEGER_F64).contains(&number)
            {
                return Value::from(number as i64);
            }

            serde_json::Number::from_f64(number)
                .map(Value::Number)
                .unwrap_or(Value::Null)
        }

        fn prost_value_to_json(value: &prost_types::Value) -> Value {
            match value.kind.as_ref() {
                Some(prost_types::value::Kind::NullValue(_)) | None => Value::Null,
                Some(prost_types::value::Kind::NumberValue(number)) => prost_number_to_json(*number),
                Some(prost_types::value::Kind::StringValue(text)) => Value::String(text.clone()),
                Some(prost_types::value::Kind::BoolValue(flag)) => Value::Bool(*flag),
                Some(prost_types::value::Kind::StructValue(struct_value)) => {
                    Value::Object(struct_to_json_map(struct_value))
                }
                Some(prost_types::value::Kind::ListValue(list_value)) => Value::Array(
                    list_value
                        .values
                        .iter()
                        .map(prost_value_to_json)
                        .collect::<Vec<_>>(),
                ),
            }
        }

        fn struct_to_json_map(struct_value: &prost_types::Struct) -> serde_json::Map<String, Value> {
            struct_value
                .fields
                .iter()
                .map(|(key, value)| (key.clone(), prost_value_to_json(value)))
                .collect()
        }

        fn filter_struct_json(
            struct_value: Option<&prost_types::Struct>,
            keys: &[&str],
        ) -> Value {
            let Some(struct_value) = struct_value else {
                return Value::Null;
            };

            let mut map = serde_json::Map::new();
            for key in keys {
                if let Some(value) = struct_value.fields.get(*key) {
                    map.insert((*key).to_string(), prost_value_to_json(value));
                }
            }
            Value::Object(map)
        }

        fn optional_struct_json(struct_value: Option<&prost_types::Struct>) -> Value {
            struct_value
                .map(|struct_value| Value::Object(struct_to_json_map(struct_value)))
                .unwrap_or(Value::Null)
        }

        fn optional_timestamp_json(timestamp: Option<&prost_types::Timestamp>) -> Value {
            timestamp
                .map(|timestamp| {
                    serde_json::json!({
                        "seconds": timestamp.seconds,
                        "nanos": timestamp.nanos,
                    })
                })
                .unwrap_or(Value::Null)
        }

        match self {
            ServerInfo::Sglang(info) => {
                let mut payload = serde_json::Map::new();
                payload.insert(
                    "server_args".to_string(),
                    filter_struct_json(info.server_args.as_ref(), SGLANG_GRPC_KEYS),
                );
                payload.insert(
                    "scheduler_info".to_string(),
                    optional_struct_json(info.scheduler_info.as_ref()),
                );
                payload.insert("active_requests".to_string(), Value::from(info.active_requests));
                payload.insert("is_paused".to_string(), Value::Bool(info.is_paused));
                if let Some(number) = serde_json::Number::from_f64(info.last_receive_timestamp) {
                    payload.insert("last_receive_timestamp".to_string(), Value::Number(number));
                }
                if let Some(number) = serde_json::Number::from_f64(info.uptime_seconds) {
                    payload.insert("uptime_seconds".to_string(), Value::Number(number));
                }
                payload.insert(
                    "max_total_num_tokens".to_string(),
                    Value::from(info.max_total_num_tokens),
                );
                payload.insert(
                    "sglang_version".to_string(),
                    Value::String(info.sglang_version.clone()),
                );
                payload.insert(
                    "server_type".to_string(),
                    Value::String(info.server_type.clone()),
                );
                payload.insert(
                    "start_time".to_string(),
                    optional_timestamp_json(info.start_time.as_ref()),
                );
                Value::Object(payload)
            }
            ServerInfo::Vllm(info) => serde_json::to_value(info)
                .unwrap_or_else(|_| Value::Object(serde_json::Map::new())),
            ServerInfo::Trtllm(info) => serde_json::to_value(info)
                .unwrap_or_else(|_| Value::Object(serde_json::Map::new())),
            ServerInfo::Mlx(info) => serde_json::to_value(info)
                .unwrap_or_else(|_| Value::Object(serde_json::Map::new())),
            ServerInfo::TokenSpeed(info) => {
                let mut payload = serde_json::Map::new();
                payload.insert(
                    "server_args".to_string(),
                    filter_struct_json(info.server_args.as_ref(), TOKENSPEED_GRPC_KEYS),
                );
                payload.insert(
                    "scheduler_info".to_string(),
                    optional_struct_json(info.scheduler_info.as_ref()),
                );
                payload.insert("active_requests".to_string(), Value::from(info.active_requests));
                payload.insert("is_paused".to_string(), Value::Bool(info.is_paused));
                if let Some(number) = serde_json::Number::from_f64(info.uptime_seconds) {
                    payload.insert("uptime_seconds".to_string(), Value::Number(number));
                }
                payload.insert(
                    "max_total_num_tokens".to_string(),
                    Value::from(info.max_total_num_tokens),
                );
                payload.insert(
                    "tokenspeed_version".to_string(),
                    Value::String(info.tokenspeed_version.clone()),
                );
                payload.insert(
                    "start_time".to_string(),
                    optional_timestamp_json(info.start_time.as_ref()),
                );
                Value::Object(payload)
            }
        }
    }
}

/// Keys worth extracting from SGLang gRPC `server_args` (which contains the full config).
const SGLANG_GRPC_KEYS: &[&str] = &[
    "model_path",
    "served_model_name",
    "tokenizer_path",
    "tp_size",
    "dp_size",
    "pp_size",
    "context_length",
    "max_total_tokens",
    "max_running_requests",
    "load_balance_method",
    "disaggregation_mode",
    "is_embedding",
    "vocab_size",
    "weight_version",
];

/// Keys worth extracting from TokenSpeed gRPC `server_args` (post-rename: bare
/// names, not `_path` variants — TokenSpeed dropped the legacy suffixes).
const TOKENSPEED_GRPC_KEYS: &[&str] = &[
    "model",
    "served_model_name",
    "tokenizer",
    "tp_size",
    "dp_size",
    "pp_size",
    "context_length",
    "max_total_tokens",
    "max_running_requests",
    "load_balance_method",
    "is_embedding",
    "vocab_size",
    "weight_version",
];

// ---------------------------------------------------------------------------
// Label helpers
// ---------------------------------------------------------------------------

/// Serialize to flat label map, skipping nulls/zeros/empty.
///
/// Booleans are emitted as `"true"` / `"false"` so downstream consumers
/// (e.g. `is_generation == "false"` for embedding detection) work correctly.
pub(crate) fn flat_labels<T: serde::Serialize>(value: &T) -> HashMap<String, String> {
    let mut labels = HashMap::new();
    if let Ok(Value::Object(obj)) = serde_json::to_value(value) {
        for (key, val) in obj {
            match val {
                Value::String(s) if !s.is_empty() && s != "null" => {
                    labels.insert(key, s);
                }
                Value::Number(n) if n.as_f64().is_some_and(|v| v != 0.0) => {
                    // Format integers without decimal point
                    let formatted = n
                        .as_i64()
                        .map(|i| i.to_string())
                        .unwrap_or_else(|| n.to_string());
                    labels.insert(key, formatted);
                }
                Value::Bool(b) => {
                    labels.insert(key, b.to_string());
                }
                Value::Array(arr) if !arr.is_empty() => {
                    if let Ok(s) = serde_json::to_string(&arr) {
                        labels.insert(key, s);
                    }
                }
                _ => {}
            }
        }
    }
    labels
}

/// Pick specific keys from a `prost_types::Struct`.
fn pick_prost_fields(labels: &mut HashMap<String, String>, s: &prost_types::Struct, keys: &[&str]) {
    for key in keys {
        if let Some(val) = s.fields.get(*key) {
            if let Some(ref kind) = val.kind {
                match kind {
                    prost_types::value::Kind::StringValue(s) if !s.is_empty() && s != "null" => {
                        labels.insert((*key).to_string(), s.clone());
                    }
                    prost_types::value::Kind::NumberValue(n) if *n != 0.0 => {
                        let formatted = if *n == (*n as i64) as f64 {
                            (*n as i64).to_string()
                        } else {
                            n.to_string()
                        };
                        labels.insert((*key).to_string(), formatted);
                    }
                    prost_types::value::Kind::BoolValue(b) => {
                        labels.insert((*key).to_string(), b.to_string());
                    }
                    _ => {}
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use smg_grpc_client::{sglang_proto, tokenspeed_proto};

    use super::ServerInfo;

    fn string_value(value: &str) -> prost_types::Value {
        prost_types::Value {
            kind: Some(prost_types::value::Kind::StringValue(value.to_string())),
        }
    }

    fn number_value(value: f64) -> prost_types::Value {
        prost_types::Value {
            kind: Some(prost_types::value::Kind::NumberValue(value)),
        }
    }

    #[test]
    fn to_public_json_sglang_keeps_pb_shape_and_filters_server_args() {
        let server_info = ServerInfo::Sglang(Box::new(sglang_proto::GetServerInfoResponse {
            server_args: Some(prost_types::Struct {
                fields: BTreeMap::from([
                    ("model_path".to_string(), string_value("Qwen/Qwen3-8B")),
                    ("tp_size".to_string(), number_value(2.0)),
                    ("host".to_string(), string_value("127.0.0.1")),
                ]),
            }),
            scheduler_info: Some(prost_types::Struct {
                fields: BTreeMap::from([
                    ("max_total_num_tokens".to_string(), number_value(16384.0)),
                    ("policy".to_string(), string_value("lpm")),
                ]),
            }),
            active_requests: 3,
            is_paused: true,
            last_receive_timestamp: 12.5,
            uptime_seconds: 7.25,
            sglang_version: "0.4.0".to_string(),
            server_type: "grpc".to_string(),
            start_time: Some(prost_types::Timestamp {
                seconds: 100,
                nanos: 5,
            }),
            max_total_num_tokens: 8192,
            ..Default::default()
        }));

        let json = server_info.to_public_json();

        assert_eq!(json["server_args"]["model_path"], "Qwen/Qwen3-8B");
        assert_eq!(json["server_args"]["tp_size"], 2);
        assert!(json["server_args"].get("host").is_none());
        assert_eq!(json["scheduler_info"]["policy"], "lpm");
        assert_eq!(json["scheduler_info"]["max_total_num_tokens"], 16384);
        assert_eq!(json["active_requests"], 3);
        assert_eq!(json["is_paused"], true);
        assert_eq!(json["last_receive_timestamp"], 12.5);
        assert_eq!(json["uptime_seconds"], 7.25);
        assert_eq!(json["sglang_version"], "0.4.0");
        assert_eq!(json["server_type"], "grpc");
        assert_eq!(json["start_time"]["seconds"], 100);
        assert_eq!(json["max_total_num_tokens"], 8192);
    }

    #[test]
    fn to_public_json_tokenspeed_keeps_pb_shape_and_filters_server_args() {
        let server_info =
            ServerInfo::TokenSpeed(Box::new(tokenspeed_proto::GetServerInfoResponse {
                server_args: Some(prost_types::Struct {
                    fields: BTreeMap::from([
                        ("model".to_string(), string_value("Qwen/Qwen3-8B")),
                        ("tokenizer".to_string(), string_value("tok-path")),
                        ("dp_size".to_string(), number_value(4.0)),
                        ("host".to_string(), string_value("127.0.0.1")),
                    ]),
                }),
                scheduler_info: Some(prost_types::Struct {
                    fields: BTreeMap::from([(
                        "max_total_num_tokens".to_string(),
                        number_value(32768.0),
                    )]),
                }),
                active_requests: 1,
                is_paused: false,
                uptime_seconds: 4.5,
                max_total_num_tokens: 4096,
                tokenspeed_version: "0.1.0".to_string(),
                start_time: Some(prost_types::Timestamp {
                    seconds: 200,
                    nanos: 9,
                }),
                ..Default::default()
            }));

        let json = server_info.to_public_json();

        assert_eq!(json["server_args"]["model"], "Qwen/Qwen3-8B");
        assert_eq!(json["server_args"]["tokenizer"], "tok-path");
        assert_eq!(json["server_args"]["dp_size"], 4);
        assert!(json["server_args"].get("host").is_none());
        assert_eq!(json["scheduler_info"]["max_total_num_tokens"], 32768);
        assert_eq!(json["active_requests"], 1);
        assert_eq!(json["is_paused"], false);
        assert_eq!(json["uptime_seconds"], 4.5);
        assert_eq!(json["tokenspeed_version"], "0.1.0");
        assert_eq!(json["start_time"]["seconds"], 200);
        assert_eq!(json["max_total_num_tokens"], 4096);
    }
}
