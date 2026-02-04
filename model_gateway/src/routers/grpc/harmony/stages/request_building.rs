//! Harmony Request Building Stage: Build gRPC request from Harmony-encoded tokens

use async_trait::async_trait;
use axum::response::Response;
use tracing::{debug, error};
use uuid::Uuid;

use crate::routers::{
    error,
    grpc::{
        client::GrpcClient,
        common::stages::{helpers, PipelineStage},
        context::{ClientSelection, RequestContext, RequestType},
        proto_wrapper::{ProtoGenerateRequest, ProtoRequest},
    },
};

/// Harmony Request Building stage: Convert Harmony tokens to gRPC request
///
/// Takes the Harmony-encoded input_ids from preparation and builds a proto::GenerateRequest.
/// Unlike regular request building, this uses token_ids directly (Harmony encoding handles messages).
pub(crate) struct HarmonyRequestBuildingStage {
    inject_pd_metadata: bool,
}

impl HarmonyRequestBuildingStage {
    /// Create a new Harmony request building stage
    pub fn new(inject_pd_metadata: bool) -> Self {
        Self { inject_pd_metadata }
    }
}

#[async_trait]
impl PipelineStage for HarmonyRequestBuildingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        // Get preparation output
        let prep = ctx.state.preparation.as_ref().ok_or_else(|| {
            error!(
                function = "HarmonyRequestBuildingStage::execute",
                "Preparation stage not completed"
            );
            error::internal_error("preparation_not_completed", "Preparation not completed")
        })?;

        // Get clients
        let clients = ctx.state.clients.as_ref().ok_or_else(|| {
            error!(
                function = "HarmonyRequestBuildingStage::execute",
                "Client acquisition stage not completed"
            );
            error::internal_error(
                "client_acquisition_not_completed",
                "Client acquisition not completed",
            )
        })?;
        let builder_client = match clients {
            ClientSelection::Single { client } => client,
            ClientSelection::Dual { prefill, .. } => prefill,
        };

        // Generate request_id based on request type
        let request_id = match &ctx.input.request_type {
            RequestType::Chat(_) => format!("chatcmpl-{}", Uuid::new_v4()),
            RequestType::Responses(_) => format!("responses-{}", Uuid::new_v4()),
            RequestType::Generate(_) => {
                error!(
                    function = "HarmonyRequestBuildingStage::execute",
                    "Generate request type not supported for Harmony models"
                );
                return Err(error::bad_request(
                    "harmony_generate_not_supported",
                    "Generate requests are not supported with Harmony models".to_string(),
                ));
            }
            RequestType::Embedding(_) => {
                error!(
                    function = "HarmonyRequestBuildingStage::execute",
                    "Embedding requests not supported for Harmony models"
                );
                return Err(error::bad_request(
                    "harmony_embedding_not_supported",
                    "Embedding requests are not supported with Harmony models".to_string(),
                ));
            }
            RequestType::Classify(_) => {
                error!(
                    function = "HarmonyRequestBuildingStage::execute",
                    "Classify requests not supported for Harmony models"
                );
                return Err(error::bad_request(
                    "harmony_classify_not_supported",
                    "Classify requests are not supported with Harmony models".to_string(),
                ));
            }
        };

        // Build gRPC request using token_ids directly (Harmony encoding already handled message rendering)
        let placeholder_processed_text = "[harmony]".to_string();

        // Build proto request based on backend type and request type
        let mut proto_request = match builder_client {
            GrpcClient::Sglang(sglang_client) => {
                let req = match &ctx.input.request_type {
                    RequestType::Chat(request) => {
                        let body = prep.filtered_request.as_ref().unwrap_or(request.as_ref());
                        sglang_client
                            .build_generate_request_from_chat(
                                request_id,
                                body,
                                placeholder_processed_text,
                                prep.token_ids.clone(),
                                None,
                                prep.tool_constraints.clone(),
                            )
                            .map_err(|e| {
                                error!(function = "HarmonyRequestBuildingStage::execute", error = %e, "Failed to build SGLang generate request");
                                error::bad_request("invalid_request_parameters", format!("Invalid request parameters: {}", e))
                            })?
                    }
                    RequestType::Responses(request) => sglang_client
                        .build_generate_request_from_responses(
                            request_id,
                            request.as_ref(),
                            placeholder_processed_text,
                            prep.token_ids.clone(),
                            prep.harmony_stop_ids.clone(),
                            prep.tool_constraints.clone(),
                        )
                        .map_err(|e| {
                            error!(function = "HarmonyRequestBuildingStage::execute", error = %e, "Failed to build SGLang generate request from responses");
                            error::bad_request("invalid_request_parameters", format!("Invalid request parameters: {}", e))
                        })?,
                    RequestType::Embedding(_) => {
                        return Err(error::bad_request(
                            "harmony_embedding_not_supported",
                            "Embedding requests are not supported with Harmony models".to_string(),
                        ));
                    }
                    _ => unreachable!(),
                };
                ProtoGenerateRequest::Sglang(Box::new(req))
            }
            GrpcClient::Vllm(vllm_client) => {
                let req = match &ctx.input.request_type {
                    RequestType::Chat(request) => {
                        let body = prep.filtered_request.as_ref().unwrap_or(request.as_ref());
                        vllm_client
                            .build_generate_request_from_chat(
                                request_id,
                                body,
                                placeholder_processed_text,
                                prep.token_ids.clone(),
                                prep.tool_constraints.clone(),
                            )
                            .map_err(|e| {
                                error!(function = "HarmonyRequestBuildingStage::execute", error = %e, "Failed to build vLLM generate request");
                                error::bad_request("invalid_request_parameters", format!("Invalid request parameters: {}", e))
                            })?
                    }
                    RequestType::Responses(request) => vllm_client
                        .build_generate_request_from_responses(
                            request_id,
                            request.as_ref(),
                            placeholder_processed_text,
                            prep.token_ids.clone(),
                            prep.harmony_stop_ids.clone(),
                            prep.tool_constraints.clone(),
                        )
                        .map_err(|e| {
                            error!(function = "HarmonyRequestBuildingStage::execute", error = %e, "Failed to build vLLM generate request from responses");
                            error::bad_request("invalid_request_parameters", format!("Invalid request parameters: {}", e))
                        })?,
                    RequestType::Embedding(_) => {
                        return Err(error::bad_request(
                            "harmony_embedding_not_supported",
                            "Embedding requests are not supported with Harmony models".to_string(),
                        ));
                    }
                    _ => unreachable!(),
                };
                ProtoGenerateRequest::Vllm(Box::new(req))
            }
            GrpcClient::Trtllm(trtllm_client) => {
                let req = match &ctx.input.request_type {
                    RequestType::Chat(request) => {
                        let body = prep.filtered_request.as_ref().unwrap_or(request.as_ref());
                        trtllm_client
                            .build_generate_request_from_chat(
                                request_id,
                                body,
                                placeholder_processed_text,
                                prep.token_ids.clone(),
                                prep.tool_constraints.clone(),
                            )
                            .map_err(|e| {
                                error!(function = "HarmonyRequestBuildingStage::execute", error = %e, "Failed to build TensorRT-LLM generate request");
                                error::bad_request("invalid_request_parameters", format!("Invalid request parameters: {}", e))
                            })?
                    }
                    RequestType::Responses(request) => trtllm_client
                        .build_generate_request_from_responses(
                            request_id,
                            request.as_ref(),
                            placeholder_processed_text,
                            prep.token_ids.clone(),
                            prep.harmony_stop_ids.clone(),
                            prep.tool_constraints.clone(),
                        )
                        .map_err(|e| {
                            error!(function = "HarmonyRequestBuildingStage::execute", error = %e, "Failed to build TensorRT-LLM generate request from responses");
                            error::bad_request("invalid_request_parameters", format!("Invalid request parameters: {}", e))
                        })?,
                    RequestType::Embedding(_) => {
                        return Err(error::bad_request(
                            "harmony_embedding_not_supported",
                            "Embedding requests are not supported with Harmony models".to_string(),
                        ));
                    }
                    _ => unreachable!(),
                };
                ProtoGenerateRequest::Trtllm(Box::new(req))
            }
        };

        // Inject Harmony stop token IDs into sampling params for ALL Harmony requests
        // These stop tokens (<|return|> and <|call|>) prevent the model from generating
        // malformed Harmony sequences
        if let Some(harmony_stops) = &prep.harmony_stop_ids {
            match &mut proto_request {
                ProtoGenerateRequest::Sglang(req) => {
                    if let Some(params) = req.sampling_params.as_mut() {
                        params.stop_token_ids.extend_from_slice(harmony_stops);
                        debug!(
                            stop_token_count = harmony_stops.len(),
                            "Injected Harmony stop tokens into SGLang sampling params"
                        );
                    }
                }
                ProtoGenerateRequest::Vllm(req) => {
                    if let Some(params) = req.sampling_params.as_mut() {
                        params.stop_token_ids.extend_from_slice(harmony_stops);
                        debug!(
                            stop_token_count = harmony_stops.len(),
                            "Injected Harmony stop tokens into vLLM sampling params"
                        );
                    }
                }
                ProtoGenerateRequest::Trtllm(req) => {
                    // TensorRT-LLM uses stop_words (repeated TokenSequence) instead of stop_token_ids
                    use crate::grpc_client::trtllm_proto::TokenSequence;
                    for &token_id in harmony_stops {
                        req.stop_words.push(TokenSequence {
                            token_ids: vec![token_id],
                        });
                    }
                    debug!(
                        stop_token_count = harmony_stops.len(),
                        "Injected Harmony stop tokens into TensorRT-LLM stop_words"
                    );
                }
            }
        }

        if self.inject_pd_metadata {
            if let Some(workers) = ctx.state.workers.as_ref() {
                helpers::maybe_inject_pd_metadata(&mut proto_request, workers);
            }
        }

        ctx.state.proto_request = Some(ProtoRequest::Generate(proto_request));
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "HarmonyRequestBuilding"
    }
}
