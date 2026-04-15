//! Request building stage for embedding requests

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;
use uuid::Uuid;

use crate::routers::{
    error,
    grpc::{
        client::GrpcClient,
        common::stages::PipelineStage,
        context::{RequestContext, RequestType},
        proto_wrapper::{ProtoEmbedRequest, ProtoRequest},
    },
};

/// Request building stage for embedding requests
pub(crate) struct EmbeddingRequestBuildingStage;

impl EmbeddingRequestBuildingStage {
    pub fn new() -> Self {
        Self
    }
}

impl Default for EmbeddingRequestBuildingStage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PipelineStage for EmbeddingRequestBuildingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        // Preparation output should have tokenized input
        let prep_output = ctx.state.preparation.as_ref().ok_or_else(|| {
            error!(
                function = "EmbeddingRequestBuildingStage::execute",
                "Preparation output missing"
            );
            error::internal_error("preparation_missing", "Preparation output missing")
        })?;

        // Extract client
        let client = ctx
            .state
            .clients
            .as_ref()
            .and_then(|c| c.single())
            .ok_or_else(|| {
                error!(
                    function = "EmbeddingRequestBuildingStage::execute",
                    "Client not selected"
                );
                error::internal_error("client_missing", "Client not selected")
            })?;

        // Generate request ID with appropriate prefix based on request type
        let request_id = match &ctx.input.request_type {
            RequestType::Embedding(_) => format!("embed-{}", Uuid::now_v7()),
            RequestType::Classify(_) => format!("classify-{}", Uuid::now_v7()),
            _ => format!("embed-{}", Uuid::now_v7()), // fallback
        };

        // Build backend-specific embed request
        let original_text = prep_output.routing_text().map(String::from);
        let token_ids = prep_output.token_ids().to_vec();

        let proto_req = match client {
            GrpcClient::Sglang(c) => {
                let req = c.build_embed_request(request_id.clone(), original_text, token_ids);
                ProtoEmbedRequest::Sglang(Box::new(req))
            }
            GrpcClient::Vllm(c) => {
                let req = c.build_embed_request(request_id.clone(), original_text, token_ids);
                ProtoEmbedRequest::Vllm(Box::new(req))
            }
            GrpcClient::Trtllm(_) => {
                error!(
                    function = "EmbeddingRequestBuildingStage::execute",
                    "TensorRT-LLM embedding not yet supported"
                );
                return Err(error::not_implemented(
                    "unsupported_backend",
                    "TensorRT-LLM embedding is not yet supported via gRPC",
                ));
            }
            GrpcClient::Mlx(_) => {
                error!(
                    function = "EmbeddingRequestBuildingStage::execute",
                    "MLX embedding not supported"
                );
                return Err(error::not_implemented(
                    "unsupported_backend",
                    "MLX embedding is not supported via gRPC",
                ));
            }
        };

        ctx.state.proto_request = Some(ProtoRequest::Embed(proto_req));
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "EmbeddingRequestBuilding"
    }
}
