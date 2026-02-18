//! Unified gRPC client wrapper for SGLang, vLLM, and TensorRT-LLM backends

use std::collections::HashMap;

use llm_tokenizer::bundle::TokenizerBundle;
use openai_protocol::{chat::ChatCompletionRequest, generate::GenerateRequest};
use smg_grpc_client::{
    sglang_proto::MultimodalInputs, SglangSchedulerClient, TrtllmServiceClient, VllmEngineClient,
};

use crate::routers::grpc::proto_wrapper::{
    ProtoEmbedRequest, ProtoEmbedResponse, ProtoGenerateRequest, ProtoStream,
};

/// Health check response (common across backends)
#[derive(Debug, Clone)]
pub struct HealthCheckResponse {
    pub healthy: bool,
    pub message: String,
}

/// Polymorphic gRPC client that wraps SGLang, vLLM, or TensorRT-LLM
#[derive(Clone)]
pub enum GrpcClient {
    Sglang(SglangSchedulerClient),
    Vllm(VllmEngineClient),
    Trtllm(TrtllmServiceClient),
}

impl GrpcClient {
    pub fn as_sglang(&self) -> &SglangSchedulerClient {
        match self {
            Self::Sglang(client) => client,
            _ => panic!("Expected SGLang client"),
        }
    }

    pub fn as_sglang_mut(&mut self) -> &mut SglangSchedulerClient {
        match self {
            Self::Sglang(client) => client,
            _ => panic!("Expected SGLang client"),
        }
    }

    pub fn as_vllm(&self) -> &VllmEngineClient {
        match self {
            Self::Vllm(client) => client,
            _ => panic!("Expected vLLM client"),
        }
    }

    pub fn as_vllm_mut(&mut self) -> &mut VllmEngineClient {
        match self {
            Self::Vllm(client) => client,
            _ => panic!("Expected vLLM client"),
        }
    }

    pub fn as_trtllm(&self) -> &TrtllmServiceClient {
        match self {
            Self::Trtllm(client) => client,
            _ => panic!("Expected TensorRT-LLM client"),
        }
    }

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

    pub async fn connect(
        url: &str,
        runtime_type: &str,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        match runtime_type {
            "sglang" => Ok(Self::Sglang(SglangSchedulerClient::connect(url).await?)),
            "vllm" => Ok(Self::Vllm(VllmEngineClient::connect(url).await?)),
            "trtllm" | "tensorrt-llm" => Ok(Self::Trtllm(TrtllmServiceClient::connect(url).await?)),
            _ => Err(format!("Unknown runtime type: {}", runtime_type).into()),
        }
    }

    pub async fn health_check(
        &self,
    ) -> Result<HealthCheckResponse, Box<dyn std::error::Error + Send + Sync>> {
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
        }
    }

    pub async fn get_model_info(
        &self,
    ) -> Result<ModelInfo, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            Self::Sglang(client) => Ok(ModelInfo::Sglang(Box::new(client.get_model_info().await?))),
            Self::Vllm(client) => Ok(ModelInfo::Vllm(client.get_model_info().await?)),
            Self::Trtllm(client) => Ok(ModelInfo::Trtllm(client.get_model_info().await?)),
        }
    }

    pub async fn get_server_info(
        &self,
    ) -> Result<ServerInfo, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            Self::Sglang(client) => Ok(ServerInfo::Sglang(Box::new(
                client.get_server_info().await?,
            ))),
            Self::Vllm(client) => Ok(ServerInfo::Vllm(client.get_server_info().await?)),
            Self::Trtllm(client) => Ok(ServerInfo::Trtllm(client.get_server_info().await?)),
        }
    }

    /// Fetch tokenizer bundle from backend runtime when supported.
    pub async fn get_tokenizer(
        &self,
    ) -> Result<TokenizerBundle, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            Self::Sglang(client) => client.get_tokenizer().await,
            Self::Vllm(client) => client.get_tokenizer().await,
            Self::Trtllm(client) => client.get_tokenizer().await,
        }
    }

    /// Generate streaming response from request
    ///
    /// Dispatches to the appropriate backend client and wraps the result in ProtoStream
    pub async fn generate(
        &mut self,
        req: ProtoGenerateRequest,
    ) -> Result<ProtoStream, Box<dyn std::error::Error + Send + Sync>> {
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
            _ => panic!("Mismatched client and request types"),
        }
    }

    pub async fn embed(
        &mut self,
        req: ProtoEmbedRequest,
    ) -> Result<ProtoEmbedResponse, Box<dyn std::error::Error + Send + Sync>> {
        match (self, req) {
            (Self::Sglang(client), ProtoEmbedRequest::Sglang(boxed_req)) => {
                let resp = client.embed(*boxed_req).await?;
                Ok(ProtoEmbedResponse::Sglang(resp))
            }
            _ => panic!("Mismatched client and request types or unsupported embedding backend"),
        }
    }

    pub fn build_chat_request(
        &self,
        request_id: String,
        body: &ChatCompletionRequest,
        processed_text: String,
        token_ids: Vec<u32>,
        multimodal_inputs: Option<MultimodalInputs>,
        tool_constraints: Option<(String, String)>,
    ) -> Result<ProtoGenerateRequest, String> {
        match self {
            Self::Sglang(client) => {
                let req = client.build_generate_request_from_chat(
                    request_id,
                    body,
                    processed_text,
                    token_ids,
                    multimodal_inputs,
                    tool_constraints,
                )?;
                Ok(ProtoGenerateRequest::Sglang(Box::new(req)))
            }
            Self::Vllm(client) => {
                let req = client.build_generate_request_from_chat(
                    request_id,
                    body,
                    processed_text,
                    token_ids,
                    tool_constraints,
                )?;
                Ok(ProtoGenerateRequest::Vllm(Box::new(req)))
            }
            Self::Trtllm(client) => {
                let req = client.build_generate_request_from_chat(
                    request_id,
                    body,
                    processed_text,
                    token_ids,
                    tool_constraints,
                )?;
                Ok(ProtoGenerateRequest::Trtllm(Box::new(req)))
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
}

pub enum ServerInfo {
    Sglang(Box<smg_grpc_client::sglang_proto::GetServerInfoResponse>),
    Vllm(smg_grpc_client::vllm_proto::GetServerInfoResponse),
    Trtllm(smg_grpc_client::trtllm_proto::GetServerInfoResponse),
}

impl ModelInfo {
    pub fn to_labels(&self) -> HashMap<String, String> {
        match self {
            ModelInfo::Sglang(info) => flat_labels(info),
            ModelInfo::Vllm(info) => flat_labels(info),
            ModelInfo::Trtllm(info) => flat_labels(info),
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

// ---------------------------------------------------------------------------
// Label helpers
// ---------------------------------------------------------------------------

/// Serialize to flat label map, skipping nulls/zeros/empty.
///
/// Booleans are emitted as `"true"` / `"false"` so downstream consumers
/// (e.g. `is_generation == "false"` for embedding detection) work correctly.
pub(crate) fn flat_labels<T: serde::Serialize>(value: &T) -> HashMap<String, String> {
    let mut labels = HashMap::new();
    if let Ok(serde_json::Value::Object(obj)) = serde_json::to_value(value) {
        for (key, val) in obj {
            match val {
                serde_json::Value::String(s) if !s.is_empty() && s != "null" => {
                    labels.insert(key, s);
                }
                serde_json::Value::Number(n) if n.as_f64().is_some_and(|v| v != 0.0) => {
                    // Format integers without decimal point
                    let formatted = n
                        .as_i64()
                        .map(|i| i.to_string())
                        .unwrap_or_else(|| n.to_string());
                    labels.insert(key, formatted);
                }
                serde_json::Value::Bool(b) => {
                    labels.insert(key, b.to_string());
                }
                serde_json::Value::Array(arr) if !arr.is_empty() => {
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
                        labels.insert(key.to_string(), s.clone());
                    }
                    prost_types::value::Kind::NumberValue(n) if *n != 0.0 => {
                        let formatted = if *n == (*n as i64) as f64 {
                            (*n as i64).to_string()
                        } else {
                            n.to_string()
                        };
                        labels.insert(key.to_string(), formatted);
                    }
                    prost_types::value::Kind::BoolValue(b) => {
                        labels.insert(key.to_string(), b.to_string());
                    }
                    _ => {}
                }
            }
        }
    }
}
