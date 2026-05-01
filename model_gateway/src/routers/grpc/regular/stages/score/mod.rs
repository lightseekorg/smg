//! Score pipeline stage for vLLM native gRPC `/v1/score` endpoint
//!
//! This stage handles native gRPC scoring by using the populated GrpcClient
//! to directly dispatch to the backend vLLM worker's `Score` RPC.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};
use tracing::{debug, error};
use uuid::Uuid;

use crate::routers::{
    error,
    grpc::{
        common::stages::PipelineStage,
        context::{ClientSelection, PreparationOutput, RequestContext},
        utils::tonic_ext::TonicStatusExt,
    },
};

/// Stage to prepare context for WorkerSelection
pub(crate) struct ScorePreparationStage;

#[async_trait::async_trait]
impl PipelineStage for ScorePreparationStage {
    fn name(&self) -> &'static str {
        "ScorePreparationStage"
    }

    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let req = ctx.score_request_arc();
        let original_text = req.text_1.clone();

        ctx.state.preparation = Some(PreparationOutput {
            original_text: Some(original_text),
            token_ids: Vec::new(), // Scoring worker routing doesn't strictly need accurate token lengths
            processed_messages: None,
            tool_constraints: None,
            filtered_request: None,
            harmony_mode: false,
            selection_text: None,
            harmony_messages: None,
            harmony_stop_ids: None,
        });

        Ok(None)
    }
}

/// Pipeline stage that executes `/v1/score` requests via gRPC
///
/// This stage leverages the `ClientAcquisitionStage` which populates the context
/// with the appropriate `GrpcClient`, and then directly uses native gRPC.
/// It short-circuits the pipeline by returning `Ok(Some(response))`
/// after forwarding, so no downstream stages are executed.
pub(crate) struct ScoreNativeStage;

impl ScoreNativeStage {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl PipelineStage for ScoreNativeStage {
    fn name(&self) -> &'static str {
        "ScoreNativeStage"
    }

    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let score_req = ctx.score_request_arc();

        let mut client = match ctx.state.clients.take() {
            Some(ClientSelection::Single { client: c }) => c,
            _ => {
                error!(stage = self.name(), "Missing client for ScoreNativeStage");
                return Err(error::internal_error(
                    "score_missing_client",
                    "Missing gRPC client",
                ));
            }
        };

        // Convert request into vLLM ScoreRequest proto
        let request_id = format!("score-{}", Uuid::now_v7());

        // `score_req.text_2` is a `StringOrVec`. `into_vec()` will give us the vector of contents.
        let text_2 = score_req.text_2.clone().into_vec();

        let proto_req =
            match client.build_score_request(request_id, score_req.text_1.clone(), text_2) {
                Ok(req) => req,
                Err(e) => {
                    error!(stage = self.name(), error = %e, "Backend unsupported for Score");
                    return Err(error::internal_error(
                        "score_backend_unsupported",
                        "Score is only supported on vLLM backend",
                    ));
                }
            };

        debug!(
            stage = self.name(),
            model = %score_req.model,
            "Forwarding Score via native gRPC"
        );

        let grpc_response = match client.score(proto_req).await {
            Ok(r) => r,
            Err(e) => {
                error!(stage = self.name(), error = %e, "native gRPC Score failed");
                return Err(e.to_http_error(
                    "native_grpc_score_failed",
                    format!("Native grpc score failed: {}", e.message()),
                ));
            }
        };

        // Put the client back into context so it drops with the right lifetime optionally
        ctx.state.clients = Some(ClientSelection::Single { client });

        // Assemble protocol response
        let mut results = Vec::with_capacity(grpc_response.data.len());
        for res in grpc_response.data {
            results.push(openai_protocol::rerank::ScoreData {
                index: res.index as usize,
                score: res.score as f64,
                object: "score".to_string(),
            });
        }

        let resp = openai_protocol::rerank::ScoreResponse {
            id: grpc_response.request_id,
            object: "list".to_string(),
            created: grpc_response.created,
            model: score_req.model.clone(),
            data: results,
            usage: Some(openai_protocol::common::UsageInfo {
                prompt_tokens: grpc_response.prompt_tokens,
                completion_tokens: 0,
                total_tokens: grpc_response.total_tokens,
                reasoning_tokens: None,
                prompt_tokens_details: None,
            }),
        };

        let axum_response = (StatusCode::OK, axum::Json(resp)).into_response();
        Ok(Some(axum_response))
    }
}
