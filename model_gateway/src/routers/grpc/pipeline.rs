//! Pipeline orchestrator for gRPC router request processing
//!
//! This module defines the RequestPipeline orchestrator that coordinates
//! the execution of pipeline stages from request preparation to response delivery.

use std::{sync::Arc, time::Instant};

use axum::response::{IntoResponse, Response};
use openai_protocol::{
    chat::{ChatCompletionRequest, ChatCompletionResponse},
    classify::ClassifyRequest,
    completion::CompletionRequest,
    embedding::EmbeddingRequest,
    generate::GenerateRequest,
    messages::CreateMessageRequest,
};
use reasoning_parser::ParserFactory as ReasoningParserFactory;
use tool_parser::ParserFactory as ToolParserFactory;
use tracing::{debug, error};

// Import embedding-specific, classify-specific, messages-specific, and completion-specific stages
use super::regular::stages::classify::ClassifyResponseProcessingStage;
use super::{
    common::{responses::ResponsesContext, stages::*},
    context::*,
    harmony,
    mode::Mode,
    regular::{
        processor,
        stages::{
            completion::{
                CompletionPreparationStage, CompletionRequestBuildingStage,
                CompletionResponseProcessingStage,
            },
            embedding::{
                preparation::EmbeddingPreparationStage,
                request_building::EmbeddingRequestBuildingStage,
                response_processing::EmbeddingResponseProcessingStage,
            },
            messages::{
                MessagePreparationStage, MessageRequestBuildingStage,
                MessageResponseProcessingStage,
            },
            ChatGeneratePreparationStage, ChatGenerateRequestBuildingStage,
            ChatGenerateResponseProcessingStage,
        },
        streaming,
    },
    utils::error_type_from_status,
};
use crate::{
    middleware::TenantRequestMeta,
    observability::metrics::{bool_to_static_str, metrics_labels, Metrics},
    policies::PolicyRegistry,
    routers::error,
    worker::WorkerRegistry,
};

/// Which endpoint a pipeline serves. Selects the endpoint-specific stage list
/// (preparation / request-building / response-processing); `Mode` then selects
/// the disaggregation params within that list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Endpoint {
    Chat,
    Messages,
    Completion,
    Harmony,
    Embeddings,
    Classify,
}

/// Construction dependencies shared by every endpoint pipeline.
///
/// Bundles the args the legacy `new_*` constructors took so `build` and the
/// constructors can compose the same stage lists from one place. The parser
/// factories/overrides are consumed only by the chat/messages/harmony processors;
/// completion builds its own default-factory processors and embeddings/classify
/// build none, matching the pre-refactor behavior.
#[derive(Clone)]
pub(crate) struct PipelineDeps {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    tool_parser_factory: ToolParserFactory,
    reasoning_parser_factory: ReasoningParserFactory,
    configured_tool_parser: Option<String>,
    configured_reasoning_parser: Option<String>,
}

impl PipelineDeps {
    /// Deps for the two-arg endpoints (embeddings/classify/completion) that carry
    /// no configured parsers. The parser fields are placeholders those endpoints
    /// never read.
    fn pair(worker_registry: Arc<WorkerRegistry>, policy_registry: Arc<PolicyRegistry>) -> Self {
        Self {
            worker_registry,
            policy_registry,
            tool_parser_factory: ToolParserFactory::default(),
            reasoning_parser_factory: ReasoningParserFactory::default(),
            configured_tool_parser: None,
            configured_reasoning_parser: None,
        }
    }

    /// Build the chat/messages response processor pair from the configured
    /// parser factories, labeled with `backend`.
    fn configured_processors(
        &self,
        backend: &'static str,
    ) -> (processor::ResponseProcessor, Arc<streaming::StreamingProcessor>) {
        let processor = processor::ResponseProcessor::new(
            self.tool_parser_factory.clone(),
            self.reasoning_parser_factory.clone(),
            self.configured_tool_parser.clone(),
            self.configured_reasoning_parser.clone(),
        );
        let streaming_processor = Arc::new(streaming::StreamingProcessor::new(
            self.tool_parser_factory.clone(),
            self.reasoning_parser_factory.clone(),
            self.configured_tool_parser.clone(),
            self.configured_reasoning_parser.clone(),
            backend,
        ));
        (processor, streaming_processor)
    }

    /// Build the completion response processor pair from default parser
    /// factories (completion does not use configured parsers), labeled `backend`.
    fn default_processors(
        backend: &'static str,
    ) -> (processor::ResponseProcessor, Arc<streaming::StreamingProcessor>) {
        let processor = processor::ResponseProcessor::new(
            ToolParserFactory::default(),
            ReasoningParserFactory::default(),
            None,
            None,
        );
        let streaming_processor = Arc::new(streaming::StreamingProcessor::new(
            ToolParserFactory::default(),
            ReasoningParserFactory::default(),
            None,
            None,
            backend,
        ));
        (processor, streaming_processor)
    }

    #[cfg(test)]
    fn test_default() -> Self {
        use crate::config::types::PolicyConfig;
        Self {
            worker_registry: Arc::new(WorkerRegistry::new()),
            policy_registry: Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
            tool_parser_factory: ToolParserFactory::default(),
            reasoning_parser_factory: ReasoningParserFactory::default(),
            configured_tool_parser: None,
            configured_reasoning_parser: None,
        }
    }

    /// The six positional args the full-arg legacy constructors expect.
    #[cfg(test)]
    fn clone_full_args(
        &self,
    ) -> (
        Arc<WorkerRegistry>,
        Arc<PolicyRegistry>,
        ToolParserFactory,
        ReasoningParserFactory,
        Option<String>,
        Option<String>,
    ) {
        (
            self.worker_registry.clone(),
            self.policy_registry.clone(),
            self.tool_parser_factory.clone(),
            self.reasoning_parser_factory.clone(),
            self.configured_tool_parser.clone(),
            self.configured_reasoning_parser.clone(),
        )
    }
}

/// Generic request pipeline for all request types
///
/// Orchestrates all stages from request preparation to response delivery.
/// Configured differently for regular vs PD mode.
#[derive(Clone)]
pub(crate) struct RequestPipeline {
    stages: Arc<Vec<Box<dyn PipelineStage>>>,
    /// Backend type for metrics labeling
    backend_type: &'static str,
}

impl RequestPipeline {
    fn wrong_response_type(
        &self,
        function: &'static str,
        expected: &'static str,
        response_type: &FinalResponse,
        model: &str,
        endpoint: &'static str,
    ) -> Response {
        error!(
            function = function,
            response_type = %response_type,
            "Wrong response type: expected {expected}, got {response_type}"
        );
        Metrics::record_router_error(
            metrics_labels::ROUTER_GRPC,
            self.backend_type,
            metrics_labels::CONNECTION_GRPC,
            model,
            endpoint,
            metrics_labels::ERROR_INTERNAL,
        );
        error::internal_error("wrong_response_type", "Internal error: wrong response type")
    }

    fn no_response_produced(
        &self,
        function: &'static str,
        model: &str,
        endpoint: &'static str,
    ) -> Response {
        error!(function = function, "No response produced by pipeline");
        Metrics::record_router_error(
            metrics_labels::ROUTER_GRPC,
            self.backend_type,
            metrics_labels::CONNECTION_GRPC,
            model,
            endpoint,
            metrics_labels::ERROR_INTERNAL,
        );
        error::internal_error("no_response_produced", "No response produced")
    }

    /// Build the pipeline for `endpoint` in the given disaggregation `mode`.
    ///
    /// Single entry point that maps `mode` to the per-stage worker-selection,
    /// execution-plan, and PD-injection params. Returns `None` for
    /// endpoint/mode combinations that have no pipeline: Harmony has no EPD
    /// variant, and embeddings/classify are single-worker only.
    ///
    /// This is the one place stage lists are assembled; every legacy `new_*`
    /// constructor delegates here, so the behavior-parity test guards them all.
    pub(crate) fn build(endpoint: Endpoint, mode: Mode, deps: &PipelineDeps) -> Option<Self> {
        // PD and EPD are both served by the "pd" backend metrics bucket; only
        // plain Regular reports as "regular".
        let backend = match mode {
            Mode::Regular => metrics_labels::BACKEND_REGULAR,
            Mode::PrefillDecode | Mode::EncodePrefillDecode => metrics_labels::BACKEND_PD,
        };
        let worker_selection = mode.worker_selection();
        let plan_kind = mode.plan_kind();
        let inject_pd_metadata = mode.inject_pd_metadata();

        let stages: Vec<Box<dyn PipelineStage>> = match endpoint {
            Endpoint::Chat => {
                let (processor, streaming_processor) = deps.configured_processors(backend);
                vec![
                    Box::new(ChatGeneratePreparationStage::new()),
                    Box::new(WorkerSelectionStage::new(
                        deps.worker_registry.clone(),
                        deps.policy_registry.clone(),
                        worker_selection,
                    )),
                    Box::new(ClientAcquisitionStage),
                    Box::new(ChatGenerateRequestBuildingStage::new(
                        inject_pd_metadata,
                        plan_kind,
                    )),
                    Box::new(DispatchMetadataStage),
                    Box::new(RequestExecutionStage::new()),
                    Box::new(ChatGenerateResponseProcessingStage::new(
                        processor,
                        streaming_processor,
                    )),
                ]
            }
            Endpoint::Messages => {
                let (processor, streaming_processor) = deps.configured_processors(backend);
                vec![
                    Box::new(MessagePreparationStage),
                    Box::new(WorkerSelectionStage::new(
                        deps.worker_registry.clone(),
                        deps.policy_registry.clone(),
                        worker_selection,
                    )),
                    Box::new(ClientAcquisitionStage),
                    Box::new(MessageRequestBuildingStage::new(
                        inject_pd_metadata,
                        plan_kind,
                    )),
                    Box::new(DispatchMetadataStage),
                    Box::new(RequestExecutionStage::new()),
                    Box::new(MessageResponseProcessingStage::new(
                        processor,
                        streaming_processor,
                    )),
                ]
            }
            Endpoint::Completion => {
                // Completion uses default parser factories, not the configured ones.
                let (processor, streaming_processor) = PipelineDeps::default_processors(backend);
                vec![
                    Box::new(CompletionPreparationStage),
                    Box::new(WorkerSelectionStage::new(
                        deps.worker_registry.clone(),
                        deps.policy_registry.clone(),
                        worker_selection,
                    )),
                    Box::new(ClientAcquisitionStage),
                    Box::new(CompletionRequestBuildingStage::new(
                        inject_pd_metadata,
                        plan_kind,
                    )),
                    Box::new(DispatchMetadataStage),
                    Box::new(RequestExecutionStage::new()),
                    Box::new(CompletionResponseProcessingStage::new(
                        processor,
                        streaming_processor,
                    )),
                ]
            }
            Endpoint::Harmony => {
                // Harmony has no EPD variant.
                if matches!(mode, Mode::EncodePrefillDecode) {
                    return None;
                }
                vec![
                    Box::new(harmony::stages::HarmonyPreparationStage::new()),
                    Box::new(WorkerSelectionStage::new(
                        deps.worker_registry.clone(),
                        deps.policy_registry.clone(),
                        worker_selection,
                    )),
                    Box::new(ClientAcquisitionStage),
                    Box::new(harmony::stages::HarmonyRequestBuildingStage::new(
                        inject_pd_metadata,
                        plan_kind,
                    )),
                    Box::new(DispatchMetadataStage),
                    Box::new(RequestExecutionStage::new()),
                    Box::new(harmony::stages::HarmonyResponseProcessingStage::new()),
                ]
            }
            Endpoint::Embeddings => {
                // Embeddings are single-worker only.
                if !matches!(mode, Mode::Regular) {
                    return None;
                }
                vec![
                    Box::new(EmbeddingPreparationStage::new()),
                    Box::new(WorkerSelectionStage::new(
                        deps.worker_registry.clone(),
                        deps.policy_registry.clone(),
                        worker_selection,
                    )),
                    Box::new(ClientAcquisitionStage),
                    Box::new(EmbeddingRequestBuildingStage::new()),
                    Box::new(DispatchMetadataStage),
                    Box::new(RequestExecutionStage::new()),
                    Box::new(EmbeddingResponseProcessingStage::new()),
                ]
            }
            Endpoint::Classify => {
                // Classify is single-worker only.
                if !matches!(mode, Mode::Regular) {
                    return None;
                }
                vec![
                    Box::new(EmbeddingPreparationStage::new()),
                    Box::new(WorkerSelectionStage::new(
                        deps.worker_registry.clone(),
                        deps.policy_registry.clone(),
                        worker_selection,
                    )),
                    Box::new(ClientAcquisitionStage),
                    Box::new(EmbeddingRequestBuildingStage::new()),
                    Box::new(DispatchMetadataStage),
                    Box::new(RequestExecutionStage::new()),
                    Box::new(ClassifyResponseProcessingStage::new()),
                ]
            }
        };

        Some(Self {
            stages: Arc::new(stages),
            backend_type: backend,
        })
    }

    /// `build` for the legacy constructors, which each pass a statically-valid
    /// endpoint/mode combo. Panics on an invalid combo (a programming error).
    #[expect(
        clippy::expect_used,
        reason = "legacy constructors pass only statically-valid endpoint/mode combos"
    )]
    fn build_or_panic(endpoint: Endpoint, mode: Mode, deps: PipelineDeps) -> Self {
        Self::build(endpoint, mode, &deps)
            .expect("legacy constructor must map to a valid pipeline")
    }

    /// Create a regular (single-worker) pipeline
    pub fn new_regular(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        tool_parser_factory: ToolParserFactory,
        reasoning_parser_factory: ReasoningParserFactory,
        configured_tool_parser: Option<String>,
        configured_reasoning_parser: Option<String>,
    ) -> Self {
        Self::build_or_panic(
            Endpoint::Chat,
            Mode::Regular,
            PipelineDeps {
                worker_registry,
                policy_registry,
                tool_parser_factory,
                reasoning_parser_factory,
                configured_tool_parser,
                configured_reasoning_parser,
            },
        )
    }

    /// Create a Harmony (single-worker) pipeline for Harmony-capable models
    pub fn new_harmony(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        tool_parser_factory: ToolParserFactory,
        reasoning_parser_factory: ReasoningParserFactory,
        configured_tool_parser: Option<String>,
        configured_reasoning_parser: Option<String>,
    ) -> Self {
        Self::build_or_panic(
            Endpoint::Harmony,
            Mode::Regular,
            PipelineDeps {
                worker_registry,
                policy_registry,
                tool_parser_factory,
                reasoning_parser_factory,
                configured_tool_parser,
                configured_reasoning_parser,
            },
        )
    }

    /// Create a Harmony PD (prefill-decode) pipeline
    // Dead outside tests (no production caller yet); the parity test does call it.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "Harmony PD pipeline is wired up by a later task")
    )]
    pub fn new_harmony_pd(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        tool_parser_factory: ToolParserFactory,
        reasoning_parser_factory: ReasoningParserFactory,
        configured_tool_parser: Option<String>,
        configured_reasoning_parser: Option<String>,
    ) -> Self {
        Self::build_or_panic(
            Endpoint::Harmony,
            Mode::PrefillDecode,
            PipelineDeps {
                worker_registry,
                policy_registry,
                tool_parser_factory,
                reasoning_parser_factory,
                configured_tool_parser,
                configured_reasoning_parser,
            },
        )
    }

    /// Create a PD (prefill-decode) pipeline
    pub fn new_pd(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        tool_parser_factory: ToolParserFactory,
        reasoning_parser_factory: ReasoningParserFactory,
        configured_tool_parser: Option<String>,
        configured_reasoning_parser: Option<String>,
    ) -> Self {
        Self::build_or_panic(
            Endpoint::Chat,
            Mode::PrefillDecode,
            PipelineDeps {
                worker_registry,
                policy_registry,
                tool_parser_factory,
                reasoning_parser_factory,
                configured_tool_parser,
                configured_reasoning_parser,
            },
        )
    }

    /// Create an EPD (encode-prefill-decode) pipeline.
    ///
    /// Mirrors `new_pd`; request building emits an
    /// `ExecutionPlan::EncodePrefillDecode` with
    /// encode bootstrap info/jobs alongside the prefill/decode request. Request
    /// building injects the encode bootstrap info and drops the prefill pixels when
    /// present;
    /// `inject_pd_metadata` stays false because TokenSpeed EPD uses the encode
    /// bootstrap info rather than SGLang bootstrap metadata.
    pub fn new_epd(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        tool_parser_factory: ToolParserFactory,
        reasoning_parser_factory: ReasoningParserFactory,
        configured_tool_parser: Option<String>,
        configured_reasoning_parser: Option<String>,
    ) -> Self {
        Self::build_or_panic(
            Endpoint::Chat,
            Mode::EncodePrefillDecode,
            PipelineDeps {
                worker_registry,
                policy_registry,
                tool_parser_factory,
                reasoning_parser_factory,
                configured_tool_parser,
                configured_reasoning_parser,
            },
        )
    }

    /// Create an embeddings pipeline
    pub fn new_embeddings(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
    ) -> Self {
        Self::build_or_panic(
            Endpoint::Embeddings,
            Mode::Regular,
            PipelineDeps::pair(worker_registry, policy_registry),
        )
    }

    /// Create a classify pipeline
    ///
    /// Classify reuses embedding stages for preparation and request building,
    /// but uses its own response processing for softmax + label mapping.
    pub fn new_classify(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
    ) -> Self {
        Self::build_or_panic(
            Endpoint::Classify,
            Mode::Regular,
            PipelineDeps::pair(worker_registry, policy_registry),
        )
    }

    /// Create a Messages API pipeline (single-worker)
    ///
    /// Uses Messages-specific stages for preparation, request building, and response
    /// processing. Shares worker selection, client acquisition, dispatch metadata,
    /// and request execution stages with other pipelines.
    pub fn new_messages(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        tool_parser_factory: ToolParserFactory,
        reasoning_parser_factory: ReasoningParserFactory,
        configured_tool_parser: Option<String>,
        configured_reasoning_parser: Option<String>,
    ) -> Self {
        Self::build_or_panic(
            Endpoint::Messages,
            Mode::Regular,
            PipelineDeps {
                worker_registry,
                policy_registry,
                tool_parser_factory,
                reasoning_parser_factory,
                configured_tool_parser,
                configured_reasoning_parser,
            },
        )
    }

    /// Create a Messages API PD (prefill-decode) pipeline
    pub fn new_messages_pd(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        tool_parser_factory: ToolParserFactory,
        reasoning_parser_factory: ReasoningParserFactory,
        configured_tool_parser: Option<String>,
        configured_reasoning_parser: Option<String>,
    ) -> Self {
        Self::build_or_panic(
            Endpoint::Messages,
            Mode::PrefillDecode,
            PipelineDeps {
                worker_registry,
                policy_registry,
                tool_parser_factory,
                reasoning_parser_factory,
                configured_tool_parser,
                configured_reasoning_parser,
            },
        )
    }

    /// Create a Messages API EPD (encode-prefill-decode) pipeline.
    ///
    /// Mirrors `new_messages_pd` with `ExecutionPlanKind::EncodePrefillDecode`,
    /// so request building plans encode jobs and request execution dispatches
    /// E/P/D together.
    pub fn new_messages_epd(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        tool_parser_factory: ToolParserFactory,
        reasoning_parser_factory: ReasoningParserFactory,
        configured_tool_parser: Option<String>,
        configured_reasoning_parser: Option<String>,
    ) -> Self {
        Self::build_or_panic(
            Endpoint::Messages,
            Mode::EncodePrefillDecode,
            PipelineDeps {
                worker_registry,
                policy_registry,
                tool_parser_factory,
                reasoning_parser_factory,
                configured_tool_parser,
                configured_reasoning_parser,
            },
        )
    }

    /// Create a Completion API pipeline (single-worker)
    ///
    /// Uses Completion-specific stages for preparation, request building, and response
    /// processing. Shares worker selection, client acquisition, dispatch metadata,
    /// and request execution stages with other pipelines.
    pub fn new_completion(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
    ) -> Self {
        Self::build_or_panic(
            Endpoint::Completion,
            Mode::Regular,
            PipelineDeps::pair(worker_registry, policy_registry),
        )
    }

    /// Create a Completion API PD (prefill-decode) pipeline
    pub fn new_completion_pd(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
    ) -> Self {
        Self::build_or_panic(
            Endpoint::Completion,
            Mode::PrefillDecode,
            PipelineDeps::pair(worker_registry, policy_registry),
        )
    }

    /// Create a Completion API EPD pipeline.
    ///
    /// Completion is text-only (no multimodal encode jobs), so this
    /// exists so a TokenSpeed EPD deployment can serve completion requests via
    /// `ExecutionPlan::EncodePrefillDecode` (which bypasses the runtime PD gate
    /// that rejects TokenSpeed) rather than the prefill/decode path.
    pub fn new_completion_epd(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
    ) -> Self {
        Self::build_or_panic(
            Endpoint::Completion,
            Mode::EncodePrefillDecode,
            PipelineDeps::pair(worker_registry, policy_registry),
        )
    }

    /// Execute the complete pipeline for a chat request
    pub async fn execute_chat(
        &self,
        request: Arc<ChatCompletionRequest>,
        headers: Option<http::HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
        tenant_request_meta: Option<TenantRequestMeta>,
    ) -> Response {
        let start = Instant::now();
        // Clone Arc for metrics (cheap atomic increment) to avoid borrow issues
        let request_for_metrics = Arc::clone(&request);
        let streaming = request.stream;

        // Record request start
        Metrics::record_router_request(
            metrics_labels::ROUTER_GRPC,
            self.backend_type,
            metrics_labels::CONNECTION_GRPC,
            &request_for_metrics.model,
            metrics_labels::ENDPOINT_CHAT,
            bool_to_static_str(streaming),
        );

        let mut ctx = RequestContext::for_chat(request, headers, model_id, components);
        ctx.input.tenant_request_meta = tenant_request_meta;

        for stage in self.stages.iter() {
            match stage.execute(&mut ctx).await {
                Ok(Some(response)) => {
                    // Stage completed with streaming response - record success and return
                    Metrics::record_router_duration(
                        metrics_labels::ROUTER_GRPC,
                        self.backend_type,
                        metrics_labels::CONNECTION_GRPC,
                        &request_for_metrics.model,
                        metrics_labels::ENDPOINT_CHAT,
                        start.elapsed(),
                    );
                    return response;
                }
                Ok(None) => continue,
                Err(response) => {
                    Metrics::record_router_error(
                        metrics_labels::ROUTER_GRPC,
                        self.backend_type,
                        metrics_labels::CONNECTION_GRPC,
                        &request_for_metrics.model,
                        metrics_labels::ENDPOINT_CHAT,
                        error_type_from_status(response.status()),
                    );
                    error!(
                        "Stage {} failed with status {}",
                        stage.name(),
                        response.status()
                    );
                    return response;
                }
            }
        }

        match ctx.state.response.final_response {
            Some(FinalResponse::Chat(response)) => {
                Metrics::record_router_duration(
                    metrics_labels::ROUTER_GRPC,
                    self.backend_type,
                    metrics_labels::CONNECTION_GRPC,
                    &request_for_metrics.model,
                    metrics_labels::ENDPOINT_CHAT,
                    start.elapsed(),
                );
                axum::Json(response).into_response()
            }
            Some(
                response_type @ (FinalResponse::Generate(_)
                | FinalResponse::Completion(_)
                | FinalResponse::Embedding(_)
                | FinalResponse::Classify(_)
                | FinalResponse::Messages(_)),
            ) => self.wrong_response_type(
                "execute_chat",
                "Chat",
                &response_type,
                &request_for_metrics.model,
                metrics_labels::ENDPOINT_CHAT,
            ),
            None => self.no_response_produced(
                "execute_chat",
                &request_for_metrics.model,
                metrics_labels::ENDPOINT_CHAT,
            ),
        }
    }

    /// Execute the complete pipeline for a generate request
    pub async fn execute_generate(
        &self,
        request: Arc<GenerateRequest>,
        headers: Option<http::HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
        tenant_request_meta: Option<TenantRequestMeta>,
    ) -> Response {
        let start = Instant::now();
        let streaming = request.stream;

        // Record request start
        Metrics::record_router_request(
            metrics_labels::ROUTER_GRPC,
            self.backend_type,
            metrics_labels::CONNECTION_GRPC,
            &model_id,
            metrics_labels::ENDPOINT_GENERATE,
            bool_to_static_str(streaming),
        );

        let mut ctx = RequestContext::for_generate(request, headers, model_id.clone(), components);
        ctx.input.tenant_request_meta = tenant_request_meta;

        for stage in self.stages.iter() {
            match stage.execute(&mut ctx).await {
                Ok(Some(response)) => {
                    Metrics::record_router_duration(
                        metrics_labels::ROUTER_GRPC,
                        self.backend_type,
                        metrics_labels::CONNECTION_GRPC,
                        &model_id,
                        metrics_labels::ENDPOINT_GENERATE,
                        start.elapsed(),
                    );
                    return response;
                }
                Ok(None) => continue,
                Err(response) => {
                    Metrics::record_router_error(
                        metrics_labels::ROUTER_GRPC,
                        self.backend_type,
                        metrics_labels::CONNECTION_GRPC,
                        &model_id,
                        metrics_labels::ENDPOINT_GENERATE,
                        error_type_from_status(response.status()),
                    );
                    error!(
                        "Stage {} failed with status {}",
                        stage.name(),
                        response.status()
                    );
                    return response;
                }
            }
        }

        match ctx.state.response.final_response {
            Some(FinalResponse::Generate(response)) => {
                Metrics::record_router_duration(
                    metrics_labels::ROUTER_GRPC,
                    self.backend_type,
                    metrics_labels::CONNECTION_GRPC,
                    &model_id,
                    metrics_labels::ENDPOINT_GENERATE,
                    start.elapsed(),
                );
                axum::Json(response).into_response()
            }
            Some(
                response_type @ (FinalResponse::Chat(_)
                | FinalResponse::Completion(_)
                | FinalResponse::Embedding(_)
                | FinalResponse::Classify(_)
                | FinalResponse::Messages(_)),
            ) => self.wrong_response_type(
                "execute_generate",
                "Generate",
                &response_type,
                &model_id,
                metrics_labels::ENDPOINT_GENERATE,
            ),
            None => self.no_response_produced(
                "execute_generate",
                &model_id,
                metrics_labels::ENDPOINT_GENERATE,
            ),
        }
    }

    /// Execute the complete pipeline for a completion request
    pub async fn execute_completion(
        &self,
        request: Arc<CompletionRequest>,
        headers: Option<http::HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
        tenant_request_meta: Option<TenantRequestMeta>,
    ) -> Response {
        let start = Instant::now();
        let model = request.model.clone();
        let streaming = request.stream;

        Metrics::record_router_request(
            metrics_labels::ROUTER_GRPC,
            self.backend_type,
            metrics_labels::CONNECTION_GRPC,
            &model,
            metrics_labels::ENDPOINT_COMPLETIONS,
            bool_to_static_str(streaming),
        );

        let mut ctx = RequestContext::for_completion(request, headers, model_id, components);
        ctx.input.tenant_request_meta = tenant_request_meta;

        for stage in self.stages.iter() {
            match stage.execute(&mut ctx).await {
                Ok(Some(response)) => {
                    Metrics::record_router_duration(
                        metrics_labels::ROUTER_GRPC,
                        self.backend_type,
                        metrics_labels::CONNECTION_GRPC,
                        &model,
                        metrics_labels::ENDPOINT_COMPLETIONS,
                        start.elapsed(),
                    );
                    return response;
                }
                Ok(None) => continue,
                Err(response) => {
                    Metrics::record_router_error(
                        metrics_labels::ROUTER_GRPC,
                        self.backend_type,
                        metrics_labels::CONNECTION_GRPC,
                        &model,
                        metrics_labels::ENDPOINT_COMPLETIONS,
                        error_type_from_status(response.status()),
                    );
                    error!(
                        "Stage {} failed with status {}",
                        stage.name(),
                        response.status()
                    );
                    return response;
                }
            }
        }

        match ctx.state.response.final_response {
            Some(FinalResponse::Completion(response)) => {
                Metrics::record_router_duration(
                    metrics_labels::ROUTER_GRPC,
                    self.backend_type,
                    metrics_labels::CONNECTION_GRPC,
                    &model,
                    metrics_labels::ENDPOINT_COMPLETIONS,
                    start.elapsed(),
                );
                axum::Json(response).into_response()
            }
            Some(
                response_type @ (FinalResponse::Chat(_)
                | FinalResponse::Generate(_)
                | FinalResponse::Embedding(_)
                | FinalResponse::Classify(_)
                | FinalResponse::Messages(_)),
            ) => self.wrong_response_type(
                "execute_completion",
                "Completion",
                &response_type,
                &model,
                metrics_labels::ENDPOINT_COMPLETIONS,
            ),
            None => self.no_response_produced(
                "execute_completion",
                &model,
                metrics_labels::ENDPOINT_COMPLETIONS,
            ),
        }
    }

    /// Execute the complete pipeline for an embedding request
    pub async fn execute_embeddings(
        &self,
        request: Arc<EmbeddingRequest>,
        headers: Option<http::HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
        tenant_request_meta: Option<TenantRequestMeta>,
    ) -> Response {
        debug!(
            "execute_embeddings: Starting execution for model: {}",
            &model_id
        );
        let start = Instant::now();

        // Record request start
        Metrics::record_router_request(
            metrics_labels::ROUTER_GRPC,
            self.backend_type,
            metrics_labels::CONNECTION_GRPC,
            &model_id,
            metrics_labels::ENDPOINT_EMBEDDINGS,
            bool_to_static_str(false),
        );

        let mut ctx = RequestContext::for_embedding(request, headers, model_id.clone(), components);
        ctx.input.tenant_request_meta = tenant_request_meta;

        for stage in self.stages.iter() {
            debug!("execute_embeddings: Executing stage: {}", stage.name());
            match stage.execute(&mut ctx).await {
                Ok(Some(response)) => {
                    debug!(
                        "execute_embeddings: Stage {} returned final response.",
                        stage.name()
                    );
                    Metrics::record_router_duration(
                        metrics_labels::ROUTER_GRPC,
                        self.backend_type,
                        metrics_labels::CONNECTION_GRPC,
                        &model_id,
                        metrics_labels::ENDPOINT_EMBEDDINGS,
                        start.elapsed(),
                    );
                    return response;
                }
                Ok(None) => {
                    debug!(
                        "execute_embeddings: Stage {} completed, continuing to next stage.",
                        stage.name()
                    );
                    continue;
                }
                Err(response) => {
                    error!(
                        "execute_embeddings: Stage {} failed with status {:?}, returning error response.",
                        stage.name(),
                        response.status()
                    );
                    Metrics::record_router_error(
                        metrics_labels::ROUTER_GRPC,
                        self.backend_type,
                        metrics_labels::CONNECTION_GRPC,
                        &model_id,
                        metrics_labels::ENDPOINT_EMBEDDINGS,
                        error_type_from_status(response.status()),
                    );
                    return response;
                }
            }
        }

        debug!(
            "execute_embeddings: Pipeline finished, processing final_response. Current state: {:?}",
            ctx.state.response.final_response
        );
        match ctx.state.response.final_response {
            Some(FinalResponse::Embedding(response)) => {
                Metrics::record_router_duration(
                    metrics_labels::ROUTER_GRPC,
                    self.backend_type,
                    metrics_labels::CONNECTION_GRPC,
                    &model_id,
                    metrics_labels::ENDPOINT_EMBEDDINGS,
                    start.elapsed(),
                );
                axum::Json(response).into_response()
            }
            Some(_) => {
                error!(function = "execute_embeddings", "Wrong response type");
                error::internal_error("wrong_response_type", "Internal error: wrong response type")
            }
            None => {
                error!(
                    function = "execute_embeddings",
                    "No final response produced by pipeline."
                );
                error::internal_error("no_response_produced", "No response produced")
            }
        }
    }

    /// Execute the complete pipeline for a classify request
    pub async fn execute_classify(
        &self,
        request: Arc<ClassifyRequest>,
        headers: Option<http::HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
        tenant_request_meta: Option<TenantRequestMeta>,
    ) -> Response {
        debug!(
            "execute_classify: Starting execution for model: {}",
            &model_id
        );
        let start = Instant::now();

        // Record request start
        Metrics::record_router_request(
            metrics_labels::ROUTER_GRPC,
            self.backend_type,
            metrics_labels::CONNECTION_GRPC,
            &model_id,
            metrics_labels::ENDPOINT_CLASSIFY,
            bool_to_static_str(false), // Classify is never streaming
        );

        let mut ctx = RequestContext::for_classify(request, headers, model_id.clone(), components);
        ctx.input.tenant_request_meta = tenant_request_meta;

        for stage in self.stages.iter() {
            debug!("execute_classify: Executing stage: {}", stage.name());
            match stage.execute(&mut ctx).await {
                Ok(Some(response)) => {
                    debug!(
                        "execute_classify: Stage {} returned final response.",
                        stage.name()
                    );
                    Metrics::record_router_duration(
                        metrics_labels::ROUTER_GRPC,
                        self.backend_type,
                        metrics_labels::CONNECTION_GRPC,
                        &model_id,
                        metrics_labels::ENDPOINT_CLASSIFY,
                        start.elapsed(),
                    );
                    return response;
                }
                Ok(None) => {
                    debug!(
                        "execute_classify: Stage {} completed, continuing to next stage.",
                        stage.name()
                    );
                    continue;
                }
                Err(response) => {
                    error!(
                        "execute_classify: Stage {} failed with status {:?}, returning error response.",
                        stage.name(),
                        response.status()
                    );
                    Metrics::record_router_error(
                        metrics_labels::ROUTER_GRPC,
                        self.backend_type,
                        metrics_labels::CONNECTION_GRPC,
                        &model_id,
                        metrics_labels::ENDPOINT_CLASSIFY,
                        error_type_from_status(response.status()),
                    );
                    return response;
                }
            }
        }

        debug!(
            "execute_classify: Pipeline finished, processing final_response. Current state: {:?}",
            ctx.state.response.final_response
        );
        match ctx.state.response.final_response {
            Some(FinalResponse::Classify(response)) => {
                Metrics::record_router_duration(
                    metrics_labels::ROUTER_GRPC,
                    self.backend_type,
                    metrics_labels::CONNECTION_GRPC,
                    &model_id,
                    metrics_labels::ENDPOINT_CLASSIFY,
                    start.elapsed(),
                );
                axum::Json(response).into_response()
            }
            Some(_) => {
                error!(function = "execute_classify", "Wrong response type");
                error::internal_error("wrong_response_type", "Internal error: wrong response type")
            }
            None => {
                error!(
                    function = "execute_classify",
                    "No final response produced by pipeline."
                );
                error::internal_error("no_response_produced", "No response produced")
            }
        }
    }

    /// Execute the complete pipeline for a Messages API request
    pub async fn execute_messages(
        &self,
        request: Arc<CreateMessageRequest>,
        headers: Option<http::HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
        tenant_request_meta: Option<TenantRequestMeta>,
    ) -> Response {
        let start = Instant::now();
        let streaming = request.stream.unwrap_or(false);

        // Record request start
        Metrics::record_router_request(
            metrics_labels::ROUTER_GRPC,
            self.backend_type,
            metrics_labels::CONNECTION_GRPC,
            &request.model,
            metrics_labels::ENDPOINT_MESSAGES,
            bool_to_static_str(streaming),
        );

        let mut ctx = RequestContext::for_messages(request.clone(), headers, model_id, components);
        ctx.input.tenant_request_meta = tenant_request_meta;

        for stage in self.stages.iter() {
            match stage.execute(&mut ctx).await {
                Ok(Some(response)) => {
                    // Stage completed with streaming response
                    Metrics::record_router_duration(
                        metrics_labels::ROUTER_GRPC,
                        self.backend_type,
                        metrics_labels::CONNECTION_GRPC,
                        &request.model,
                        metrics_labels::ENDPOINT_MESSAGES,
                        start.elapsed(),
                    );
                    return response;
                }
                Ok(None) => continue,
                Err(response) => {
                    Metrics::record_router_error(
                        metrics_labels::ROUTER_GRPC,
                        self.backend_type,
                        metrics_labels::CONNECTION_GRPC,
                        &request.model,
                        metrics_labels::ENDPOINT_MESSAGES,
                        error_type_from_status(response.status()),
                    );
                    error!(
                        "Stage {} failed with status {}",
                        stage.name(),
                        response.status()
                    );
                    return response;
                }
            }
        }

        match ctx.state.response.final_response {
            Some(FinalResponse::Messages(response)) => {
                Metrics::record_router_duration(
                    metrics_labels::ROUTER_GRPC,
                    self.backend_type,
                    metrics_labels::CONNECTION_GRPC,
                    &request.model,
                    metrics_labels::ENDPOINT_MESSAGES,
                    start.elapsed(),
                );
                axum::Json(response).into_response()
            }
            Some(
                response_type @ (FinalResponse::Chat(_)
                | FinalResponse::Generate(_)
                | FinalResponse::Completion(_)
                | FinalResponse::Embedding(_)
                | FinalResponse::Classify(_)),
            ) => self.wrong_response_type(
                "execute_messages",
                "Messages",
                &response_type,
                &request.model,
                metrics_labels::ENDPOINT_MESSAGES,
            ),
            None => self.no_response_produced(
                "execute_messages",
                &request.model,
                metrics_labels::ENDPOINT_MESSAGES,
            ),
        }
    }

    /// Execute chat pipeline for responses endpoint
    ///
    /// Used by ALL non-streaming /v1/responses requests.
    /// Uses the same 7 pipeline stages as execute_chat(), with two differences:
    /// 1. Returns Result<ChatCompletionResponse, Response> for tool_loop composition
    /// 2. Disallows streaming (responses endpoint uses different SSE format)
    pub async fn execute_chat_for_responses(
        &self,
        request: Arc<ChatCompletionRequest>,
        headers: Option<http::HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
        tenant_request_meta: Option<TenantRequestMeta>,
    ) -> Result<ChatCompletionResponse, Response> {
        let mut ctx = RequestContext::for_chat(request, headers, model_id, components);
        ctx.input.tenant_request_meta = tenant_request_meta;

        for (idx, stage) in self.stages.iter().enumerate() {
            match stage.execute(&mut ctx).await {
                Ok(Some(_response)) => {
                    // Streaming not supported for responses sync mode
                    error!(
                        function = "execute_chat_for_responses",
                        "Streaming attempted in responses context"
                    );
                    return Err(error::bad_request(
                        "streaming_not_supported",
                        "Streaming is not supported in this context".to_string(),
                    ));
                }
                Ok(None) => {
                    continue;
                }
                Err(response) => {
                    // Error occurred - return the response as-is to preserve HTTP status codes
                    error!(
                        "Stage {} ({}) failed with status {}",
                        idx + 1,
                        stage.name(),
                        response.status()
                    );
                    return Err(response);
                }
            }
        }

        match ctx.state.response.final_response {
            Some(FinalResponse::Chat(response)) => Ok(response),
            Some(FinalResponse::Generate(_))
            | Some(FinalResponse::Completion(_))
            | Some(FinalResponse::Embedding(_))
            | Some(FinalResponse::Classify(_))
            | Some(FinalResponse::Messages(_)) => {
                error!(
                    function = "execute_chat_for_responses",
                    "Wrong response type: expected Chat, got Generate/Embedding/Classify/Messages"
                );
                Err(error::internal_error(
                    "wrong_response_type",
                    "Internal error: wrong response type",
                ))
            }
            None => {
                error!(
                    function = "execute_chat_for_responses",
                    "No response produced by pipeline"
                );
                Err(error::internal_error(
                    "no_response_produced",
                    "No response produced",
                ))
            }
        }
    }

    /// Execute Harmony Responses API request through all pipeline stages
    ///
    /// This method runs a single iteration of the Responses API request,
    /// returning either ToolCallsFound (continue serving) or Completed (final response).
    ///
    /// Called by harmony::responses::serve_harmony_responses() for each iteration.
    ///
    /// # Arguments
    ///
    /// * `request` - Responses API request
    /// * `ctx` - Harmony Responses context with MCP manager and components
    ///
    /// # Returns
    ///
    /// ResponsesIterationResult indicating whether to continue iteration or return
    pub async fn execute_harmony_responses(
        &self,
        request: &openai_protocol::responses::ResponsesRequest,
        harmony_ctx: &ResponsesContext,
        tenant_request_meta: Option<TenantRequestMeta>,
    ) -> Result<harmony::ResponsesIterationResult, Response> {
        // Create RequestContext for this Responses request
        let mut ctx = RequestContext::for_responses(
            Arc::new(request.clone()),
            None,                  // No headers needed for internal pipeline execution
            request.model.clone(), // Model ID from request
            harmony_ctx.components.clone(),
        );
        ctx.input.tenant_request_meta = tenant_request_meta;

        for (idx, stage) in self.stages.iter().enumerate() {
            match stage.execute(&mut ctx).await {
                Ok(Some(response)) => {
                    // Stage returned early response (e.g., streaming) - not expected for Responses iteration
                    error!(
                        "Stage {} ({}) returned unexpected response during Responses iteration",
                        idx + 1,
                        stage.name()
                    );
                    return Err(response);
                }
                Ok(None) => {
                    continue;
                }
                Err(response) => {
                    // Stage failed
                    error!(
                        "Stage {} ({}) failed with status {}",
                        idx + 1,
                        stage.name(),
                        response.status()
                    );
                    return Err(response);
                }
            }
        }

        // Extract ResponsesIterationResult from context
        // This should have been set by HarmonyResponseProcessingStage
        ctx.state
            .response
            .responses_iteration_result
            .take()
            .ok_or_else(|| {
                error!(
                    function = "execute_harmony_responses",
                    "No ResponsesIterationResult produced by pipeline"
                );
                error::internal_error(
                    "no_responses_iteration_result",
                    "No ResponsesIterationResult produced by pipeline",
                )
            })
    }

    /// Execute Harmony Responses pipeline iteration with streaming support
    ///
    /// This version executes the pipeline up to the dispatch stage and returns
    /// the raw ExecutionResult (with stream) and LoadGuards for token-level streaming processing.
    /// The caller is responsible for keeping load_guards alive until stream processing completes.
    pub async fn execute_harmony_responses_streaming(
        &self,
        request: &openai_protocol::responses::ResponsesRequest,
        harmony_ctx: &ResponsesContext,
        tenant_request_meta: Option<TenantRequestMeta>,
    ) -> Result<(ExecutionResult, Option<LoadGuards>), Response> {
        // Create RequestContext for this Responses request
        let mut ctx = RequestContext::for_responses(
            Arc::new(request.clone()),
            None,
            request.model.clone(),
            harmony_ctx.components.clone(),
        );
        ctx.input.tenant_request_meta = tenant_request_meta;

        for (idx, stage) in self.stages.iter().enumerate() {
            match stage.execute(&mut ctx).await {
                Ok(Some(response)) => {
                    error!(
                        "Stage {} ({}) returned unexpected response during streaming Responses",
                        idx + 1,
                        stage.name()
                    );
                    return Err(response);
                }
                Ok(None) => continue,
                Err(response) => {
                    error!(
                        "Stage {} ({}) failed with status {}",
                        idx + 1,
                        stage.name(),
                        response.status()
                    );
                    return Err(response);
                }
            }
        }

        // Extract execution_result (the raw stream from workers) and load_guards
        let execution_result = ctx.state.response.execution_result.take().ok_or_else(|| {
            error!(
                function = "execute_harmony_responses_streaming",
                "No ExecutionResult produced by pipeline"
            );
            error::internal_error(
                "no_execution_result_produced",
                "No ExecutionResult produced by pipeline",
            )
        })?;

        let load_guards = ctx.state.load_guards.take();

        Ok((execution_result, load_guards))
    }
}

#[cfg(test)]
mod build_parity_tests {
    use super::*;
    use crate::routers::grpc::mode::Mode;

    fn sigs(p: &RequestPipeline) -> Vec<String> {
        p.stages.iter().map(|s| s.signature()).collect()
    }

    /// Assert `build(endpoint, mode)` reproduces `legacy` (same stage sequence +
    /// mode-bearing args, same metrics backend label).
    fn assert_parity(endpoint: Endpoint, mode: Mode, deps: &PipelineDeps, legacy: RequestPipeline) {
        let built = RequestPipeline::build(endpoint, mode, deps)
            .unwrap_or_else(|| panic!("build({endpoint:?}, {mode:?}) should be valid"));
        assert_eq!(
            sigs(&built),
            sigs(&legacy),
            "stage parity for {endpoint:?}/{mode:?}"
        );
        assert_eq!(
            built.backend_type, legacy.backend_type,
            "backend_type parity for {endpoint:?}/{mode:?}"
        );
    }

    #[test]
    fn build_reproduces_legacy_constructors() {
        let deps = PipelineDeps::test_default();
        let (wr, pr, tpf, rpf, ctp, crp) = deps.clone_full_args();

        // Chat/Messages/Completion support every mode.
        assert_parity(
            Endpoint::Chat,
            Mode::Regular,
            &deps,
            RequestPipeline::new_regular(
                wr.clone(),
                pr.clone(),
                tpf.clone(),
                rpf.clone(),
                ctp.clone(),
                crp.clone(),
            ),
        );
        assert_parity(
            Endpoint::Chat,
            Mode::PrefillDecode,
            &deps,
            RequestPipeline::new_pd(
                wr.clone(),
                pr.clone(),
                tpf.clone(),
                rpf.clone(),
                ctp.clone(),
                crp.clone(),
            ),
        );
        assert_parity(
            Endpoint::Chat,
            Mode::EncodePrefillDecode,
            &deps,
            RequestPipeline::new_epd(
                wr.clone(),
                pr.clone(),
                tpf.clone(),
                rpf.clone(),
                ctp.clone(),
                crp.clone(),
            ),
        );
        assert_parity(
            Endpoint::Messages,
            Mode::Regular,
            &deps,
            RequestPipeline::new_messages(
                wr.clone(),
                pr.clone(),
                tpf.clone(),
                rpf.clone(),
                ctp.clone(),
                crp.clone(),
            ),
        );
        assert_parity(
            Endpoint::Messages,
            Mode::PrefillDecode,
            &deps,
            RequestPipeline::new_messages_pd(
                wr.clone(),
                pr.clone(),
                tpf.clone(),
                rpf.clone(),
                ctp.clone(),
                crp.clone(),
            ),
        );
        assert_parity(
            Endpoint::Messages,
            Mode::EncodePrefillDecode,
            &deps,
            RequestPipeline::new_messages_epd(
                wr.clone(),
                pr.clone(),
                tpf.clone(),
                rpf.clone(),
                ctp.clone(),
                crp.clone(),
            ),
        );
        assert_parity(
            Endpoint::Completion,
            Mode::Regular,
            &deps,
            RequestPipeline::new_completion(wr.clone(), pr.clone()),
        );
        assert_parity(
            Endpoint::Completion,
            Mode::PrefillDecode,
            &deps,
            RequestPipeline::new_completion_pd(wr.clone(), pr.clone()),
        );
        assert_parity(
            Endpoint::Completion,
            Mode::EncodePrefillDecode,
            &deps,
            RequestPipeline::new_completion_epd(wr.clone(), pr.clone()),
        );

        // Harmony: Regular + PD only; EPD is invalid.
        assert!(
            RequestPipeline::build(Endpoint::Harmony, Mode::EncodePrefillDecode, &deps).is_none(),
            "Harmony EPD must be invalid"
        );
        assert_parity(
            Endpoint::Harmony,
            Mode::Regular,
            &deps,
            RequestPipeline::new_harmony(
                wr.clone(),
                pr.clone(),
                tpf.clone(),
                rpf.clone(),
                ctp.clone(),
                crp.clone(),
            ),
        );
        assert_parity(
            Endpoint::Harmony,
            Mode::PrefillDecode,
            &deps,
            RequestPipeline::new_harmony_pd(wr.clone(), pr.clone(), tpf, rpf, ctp, crp),
        );

        // Embeddings/Classify: Regular only; PD/EPD are invalid.
        for endpoint in [Endpoint::Embeddings, Endpoint::Classify] {
            assert!(
                RequestPipeline::build(endpoint, Mode::PrefillDecode, &deps).is_none(),
                "{endpoint:?} PD must be invalid"
            );
            assert!(
                RequestPipeline::build(endpoint, Mode::EncodePrefillDecode, &deps).is_none(),
                "{endpoint:?} EPD must be invalid"
            );
        }
        assert_parity(
            Endpoint::Embeddings,
            Mode::Regular,
            &deps,
            RequestPipeline::new_embeddings(wr.clone(), pr.clone()),
        );
        assert_parity(
            Endpoint::Classify,
            Mode::Regular,
            &deps,
            RequestPipeline::new_classify(wr, pr),
        );
    }
}
