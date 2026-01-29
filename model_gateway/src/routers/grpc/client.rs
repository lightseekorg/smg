//! Unified gRPC client wrapper for SGLang, vLLM, and TensorRT-LLM backends

use openai_protocol::{chat::ChatCompletionRequest, generate::GenerateRequest};

use crate::{
    grpc_client::{
        sglang_proto::MultimodalInputs, SglangSchedulerClient, TrtllmServiceClient,
        VllmEngineClient,
    },
    routers::grpc::proto_wrapper::{
        ProtoEmbedRequest, ProtoEmbedResponse, ProtoGenerateRequest, ProtoStream,
    },
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
    /// Get reference to SGLang client (panics if not SGLang)
    pub fn as_sglang(&self) -> &SglangSchedulerClient {
        match self {
            Self::Sglang(client) => client,
            _ => panic!("Expected SGLang client"),
        }
    }

    /// Get mutable reference to SGLang client (panics if not SGLang)
    pub fn as_sglang_mut(&mut self) -> &mut SglangSchedulerClient {
        match self {
            Self::Sglang(client) => client,
            _ => panic!("Expected SGLang client"),
        }
    }

    /// Get reference to vLLM client (panics if not vLLM)
    pub fn as_vllm(&self) -> &VllmEngineClient {
        match self {
            Self::Vllm(client) => client,
            _ => panic!("Expected vLLM client"),
        }
    }

    /// Get mutable reference to vLLM client (panics if not vLLM)
    pub fn as_vllm_mut(&mut self) -> &mut VllmEngineClient {
        match self {
            Self::Vllm(client) => client,
            _ => panic!("Expected vLLM client"),
        }
    }

    /// Get reference to TensorRT-LLM client (panics if not TensorRT-LLM)
    pub fn as_trtllm(&self) -> &TrtllmServiceClient {
        match self {
            Self::Trtllm(client) => client,
            _ => panic!("Expected TensorRT-LLM client"),
        }
    }

    /// Get mutable reference to TensorRT-LLM client (panics if not TensorRT-LLM)
    pub fn as_trtllm_mut(&mut self) -> &mut TrtllmServiceClient {
        match self {
            Self::Trtllm(client) => client,
            _ => panic!("Expected TensorRT-LLM client"),
        }
    }

    /// Check if this is a SGLang client
    pub fn is_sglang(&self) -> bool {
        matches!(self, Self::Sglang(_))
    }

    /// Check if this is a vLLM client
    pub fn is_vllm(&self) -> bool {
        matches!(self, Self::Vllm(_))
    }

    /// Check if this is a TensorRT-LLM client
    pub fn is_trtllm(&self) -> bool {
        matches!(self, Self::Trtllm(_))
    }

    /// Connect to gRPC server (runtime-aware)
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

    /// Perform health check (dispatches to appropriate backend)
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
                // TensorRT-LLM returns status string, not separate healthy/message fields
                let healthy = resp.status.to_lowercase().contains("ok")
                    || resp.status.to_lowercase().contains("healthy");
                Ok(HealthCheckResponse {
                    healthy,
                    message: resp.status,
                })
            }
        }
    }

    /// Get model info (returns enum wrapping backend-specific response)
    pub async fn get_model_info(
        &self,
    ) -> Result<ModelInfo, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            Self::Sglang(client) => {
                let info = client.get_model_info().await?;
                Ok(ModelInfo::Sglang(Box::new(info)))
            }
            Self::Vllm(client) => {
                let info = client.get_model_info().await?;
                Ok(ModelInfo::Vllm(info))
            }
            Self::Trtllm(client) => {
                let info = client.get_model_info().await?;
                Ok(ModelInfo::Trtllm(info))
            }
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

    /// Submit an embedding request
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

    /// Build a generate request from a chat completion request
    ///
    /// Dispatches to the appropriate backend client and wraps the result.
    /// Note: `multimodal_inputs` is only used by SGLang, other backends ignore it.
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

    /// Build a plain generate request (non-chat)
    ///
    /// Dispatches to the appropriate backend client and wraps the result.
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

/// Unified ModelInfo wrapper
pub enum ModelInfo {
    Sglang(Box<crate::grpc_client::sglang_proto::GetModelInfoResponse>),
    Vllm(crate::grpc_client::vllm_proto::GetModelInfoResponse),
    Trtllm(crate::grpc_client::trtllm_proto::GetModelInfoResponse),
}

impl ModelInfo {
    /// Convert model info to label map for worker metadata
    pub fn to_labels(&self) -> std::collections::HashMap<String, String> {
        let mut labels = std::collections::HashMap::new();

        // Serialize to JSON Value (like pydantic's model_dump)
        let value = match self {
            ModelInfo::Sglang(info) => serde_json::to_value(info).ok(),
            ModelInfo::Vllm(info) => serde_json::to_value(info).ok(),
            ModelInfo::Trtllm(info) => serde_json::to_value(info).ok(),
        };

        // Convert JSON object to HashMap, filtering out empty/zero/false values
        if let Some(serde_json::Value::Object(obj)) = value {
            for (key, val) in obj {
                match val {
                    // Insert non-empty strings
                    serde_json::Value::String(s) if !s.is_empty() => {
                        labels.insert(key, s);
                    }
                    // Insert positive numbers
                    serde_json::Value::Number(n) if n.as_i64().unwrap_or(0) > 0 => {
                        labels.insert(key, n.to_string());
                    }
                    // Insert true booleans
                    serde_json::Value::Bool(true) => {
                        labels.insert(key, "true".to_string());
                    }
                    // Insert non-empty arrays as JSON strings (for architectures, etc.)
                    serde_json::Value::Array(arr) if !arr.is_empty() => {
                        if let Ok(json_str) = serde_json::to_string(&arr) {
                            labels.insert(key, json_str);
                        }
                    }
                    // Skip empty strings, zeros, false, nulls, empty arrays, objects
                    _ => {}
                }
            }
        }

        labels
    }
}
