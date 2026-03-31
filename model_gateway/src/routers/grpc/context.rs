//! Request context types for gRPC router pipeline
//!
//! This module provides the core context types that flow through the router pipeline,
//! eliminating deep parameter passing chains and providing a single source of truth
//! for request state.

use std::sync::Arc;

use axum::http::HeaderMap;
use llm_tokenizer::{stop::StopSequenceDecoder, traits::Tokenizer, TokenizerRegistry};
use openai_protocol::{
    chat::{ChatCompletionRequest, ChatCompletionResponse},
    classify::{ClassifyRequest, ClassifyResponse},
    completion::{CompletionRequest, CompletionResponse},
    embedding::{EmbeddingRequest, EmbeddingResponse},
    generate::{GenerateRequest, GenerateResponse},
    messages::{CreateMessageRequest, Message},
    responses::ResponsesRequest,
};
use reasoning_parser::ParserFactory as ReasoningParserFactory;
use tool_parser::ParserFactory as ToolParserFactory;
use tracing::debug;

use super::{
    client::GrpcClient,
    multimodal::MultimodalComponents,
    proto_wrapper::{ProtoEmbedComplete, ProtoRequest, ProtoStream},
};
use crate::core::{RuntimeType, Worker, WorkerLoadGuard};

/// Main request processing context
///
/// This is the single source of truth for all request state as it flows
/// through the pipeline stages. Uses Rust's type system to enforce proper
/// stage ordering at compile time.
pub(crate) struct RequestContext {
    pub input: RequestInput,
    pub components: Arc<SharedComponents>,
    pub state: ProcessingState,
}

/// Immutable request input
pub(crate) struct RequestInput {
    pub request_type: RequestType,
    pub headers: Option<HeaderMap>,
    pub model_id: String,
}

/// Request type variants
/// Using Arc instead of Box to enable cheap cloning for background tasks
pub(crate) enum RequestType {
    Chat(Arc<ChatCompletionRequest>),
    Generate(Arc<GenerateRequest>),
    Completion(Arc<CompletionRequest>),
    Responses(Arc<ResponsesRequest>),
    Embedding(Arc<EmbeddingRequest>),
    Classify(Arc<ClassifyRequest>),
    Messages(Arc<CreateMessageRequest>),
}

impl std::fmt::Display for RequestType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Chat(_) => write!(f, "Chat"),
            Self::Generate(_) => write!(f, "Generate"),
            Self::Completion(_) => write!(f, "Completion"),
            Self::Responses(_) => write!(f, "Responses"),
            Self::Embedding(_) => write!(f, "Embedding"),
            Self::Classify(_) => write!(f, "Classify"),
            Self::Messages(_) => write!(f, "Messages"),
        }
    }
}

impl std::fmt::Display for FinalResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Chat(_) => write!(f, "Chat"),
            Self::Generate(_) => write!(f, "Generate"),
            Self::Completion(_) => write!(f, "Completion"),
            Self::Embedding(_) => write!(f, "Embedding"),
            Self::Classify(_) => write!(f, "Classify"),
            Self::Messages(_) => write!(f, "Messages"),
        }
    }
}

/// Shared components (injected once at creation)
pub(crate) struct SharedComponents {
    pub tokenizer_registry: Arc<TokenizerRegistry>,
    #[expect(dead_code)]
    pub tool_parser_factory: ToolParserFactory,
    #[expect(dead_code)]
    pub reasoning_parser_factory: ReasoningParserFactory,
    /// Multimodal processing components (initialized at router creation)
    pub multimodal: Option<Arc<MultimodalComponents>>,
}

/// Mutable processing state (evolves through pipeline stages)
#[derive(Default)]
pub(crate) struct ProcessingState {
    // Stage 1: Preparation outputs
    pub preparation: Option<PreparationOutput>,

    /// Resolved tokenizer (set once in preparation, reused in response processing)
    /// This avoids redundant registry lookups across pipeline stages.
    pub tokenizer: Option<Arc<dyn Tokenizer>>,

    // Stage 2: Worker selection outputs
    pub workers: Option<WorkerSelection>,

    // Stage 3: Client acquisition outputs
    pub clients: Option<ClientSelection>,

    // Stage 4: Request building outputs
    pub proto_request: Option<ProtoRequest>,

    // Stage 5: Dispatch metadata
    pub dispatch: Option<DispatchMetadata>,

    // Load guard for worker load tracking (created at execution stage)
    pub load_guards: Option<LoadGuards>,

    // Stage 6: Response processing state
    pub response: ResponseState,
}

/// Output from preparation stage (Step 1)
pub(crate) struct PreparationOutput {
    /// Original text (for chat) or resolved text (for generate)
    pub original_text: Option<String>,

    /// Tokenized input
    pub token_ids: Vec<u32>,

    /// Processed messages (chat only)
    pub processed_messages: Option<super::ProcessedMessages>,

    /// Tool call constraints (if applicable)
    pub tool_constraints: Option<(String, String)>,

    /// Filtered request (if tools were filtered)
    pub filtered_request: Option<ChatCompletionRequest>,

    // Harmony-specific fields
    /// Whether this is a Harmony request (default: false)
    pub harmony_mode: bool,

    /// Selection text for worker routing (Harmony only)
    pub selection_text: Option<String>,

    /// Harmony messages for history tracking (Harmony only)
    #[expect(dead_code)]
    pub harmony_messages: Option<Vec<super::harmony::HarmonyMessage>>,

    /// Stop token IDs for Harmony models
    pub harmony_stop_ids: Option<Vec<u32>>,
}

/// Worker selection (Step 2)
pub(crate) enum WorkerSelection {
    Single {
        worker: Arc<dyn Worker>,
    },
    Dual {
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
        runtime_type: RuntimeType,
    },
}

/// Client selection (Step 3)
pub(crate) enum ClientSelection {
    Single {
        client: GrpcClient,
    },
    Dual {
        prefill: GrpcClient,
        decode: GrpcClient,
    },
}

/// Dispatch metadata (Step 5)
#[derive(Clone)]
pub(crate) struct DispatchMetadata {
    pub request_id: String,
    pub model: String,
    pub created: u64,
    pub weight_version: Option<String>,
}

/// Load guards for worker load tracking
/// Automatically decrements load when dropped
pub(crate) enum LoadGuards {
    Single {
        _guard: WorkerLoadGuard,
    },
    Dual {
        _prefill: WorkerLoadGuard,
        _decode: WorkerLoadGuard,
    },
}

impl LoadGuards {
    pub fn new(selection: &WorkerSelection, headers: Option<&HeaderMap>) -> Self {
        match selection {
            WorkerSelection::Single { worker } => LoadGuards::Single {
                _guard: WorkerLoadGuard::new(worker.clone(), headers),
            },
            WorkerSelection::Dual {
                prefill, decode, ..
            } => LoadGuards::Dual {
                _prefill: WorkerLoadGuard::new(prefill.clone(), headers),
                _decode: WorkerLoadGuard::new(decode.clone(), headers),
            },
        }
    }
}

/// Response processing state (Step 6)
#[derive(Default)]
pub(crate) struct ResponseState {
    /// Stop sequence decoder
    pub stop_decoder: Option<StopSequenceDecoder>,

    /// Execution result (streams from workers)
    pub execution_result: Option<ExecutionResult>,

    /// Final processed response
    pub final_response: Option<FinalResponse>,

    /// Responses API iteration result (Harmony only, for tool loop orchestration)
    pub responses_iteration_result: Option<super::harmony::ResponsesIterationResult>,
}

impl RequestContext {
    /// Create context for chat completion request
    pub fn for_chat(
        request: Arc<ChatCompletionRequest>,
        headers: Option<HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
    ) -> Self {
        Self {
            input: RequestInput {
                request_type: RequestType::Chat(request),
                headers,
                model_id,
            },
            components,
            state: ProcessingState::default(),
        }
    }

    /// Create context for generate request
    pub fn for_generate(
        request: Arc<GenerateRequest>,
        headers: Option<HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
    ) -> Self {
        Self {
            input: RequestInput {
                request_type: RequestType::Generate(request),
                headers,
                model_id,
            },
            components,
            state: ProcessingState::default(),
        }
    }

    /// Create context for completion request
    pub fn for_completion(
        request: Arc<CompletionRequest>,
        headers: Option<HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
    ) -> Self {
        Self {
            input: RequestInput {
                request_type: RequestType::Completion(request),
                headers,
                model_id,
            },
            components,
            state: ProcessingState::default(),
        }
    }

    /// Create context for Responses API request
    pub fn for_responses(
        request: Arc<ResponsesRequest>,
        headers: Option<HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
    ) -> Self {
        Self {
            input: RequestInput {
                request_type: RequestType::Responses(request),
                headers,
                model_id,
            },
            components,
            state: ProcessingState::default(),
        }
    }

    /// Create context for embedding request
    pub fn for_embedding(
        request: Arc<EmbeddingRequest>,
        headers: Option<HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
    ) -> Self {
        Self {
            input: RequestInput {
                request_type: RequestType::Embedding(request),
                headers,
                model_id,
            },
            components,
            state: ProcessingState::default(),
        }
    }

    /// Create context for classify request
    pub fn for_classify(
        request: Arc<ClassifyRequest>,
        headers: Option<HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
    ) -> Self {
        Self {
            input: RequestInput {
                request_type: RequestType::Classify(request),
                headers,
                model_id,
            },
            components,
            state: ProcessingState::default(),
        }
    }

    /// Create context for messages request
    pub fn for_messages(
        request: Arc<CreateMessageRequest>,
        headers: Option<HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
    ) -> Self {
        Self {
            input: RequestInput {
                request_type: RequestType::Messages(request),
                headers,
                model_id,
            },
            components,
            state: ProcessingState::default(),
        }
    }

    /// Get chat request (panics if not chat)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via RequestType construction"
    )]
    pub fn chat_request(&self) -> &ChatCompletionRequest {
        match &self.input.request_type {
            RequestType::Chat(req) => req.as_ref(),
            _ => panic!("Expected chat request"),
        }
    }

    /// Get Arc clone of chat request (panics if not chat)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via RequestType construction"
    )]
    pub fn chat_request_arc(&self) -> Arc<ChatCompletionRequest> {
        match &self.input.request_type {
            RequestType::Chat(req) => Arc::clone(req),
            _ => panic!("Expected chat request"),
        }
    }

    /// Get generate request (panics if not generate)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via RequestType construction"
    )]
    pub fn generate_request(&self) -> &GenerateRequest {
        match &self.input.request_type {
            RequestType::Generate(req) => req.as_ref(),
            _ => panic!("Expected generate request"),
        }
    }

    /// Get Arc clone of generate request (panics if not generate)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via RequestType construction"
    )]
    pub fn generate_request_arc(&self) -> Arc<GenerateRequest> {
        match &self.input.request_type {
            RequestType::Generate(req) => Arc::clone(req),
            _ => panic!("Expected generate request"),
        }
    }

    /// Get completion request (panics if not completion)
    #[expect(
        dead_code,
        reason = "ref accessor provided for API completeness alongside Arc accessor"
    )]
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via RequestType construction"
    )]
    pub fn completion_request(&self) -> &CompletionRequest {
        match &self.input.request_type {
            RequestType::Completion(req) => req.as_ref(),
            _ => panic!("Expected completion request"),
        }
    }

    /// Get Arc clone of completion request (panics if not completion)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via RequestType construction"
    )]
    pub fn completion_request_arc(&self) -> Arc<CompletionRequest> {
        match &self.input.request_type {
            RequestType::Completion(req) => Arc::clone(req),
            _ => panic!("Expected completion request"),
        }
    }

    /// Get Arc clone of responses request (panics if not responses)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via RequestType construction"
    )]
    pub fn responses_request_arc(&self) -> Arc<ResponsesRequest> {
        match &self.input.request_type {
            RequestType::Responses(req) => Arc::clone(req),
            _ => panic!("Expected responses request"),
        }
    }

    /// Get messages request (panics if not messages)
    #[expect(
        dead_code,
        reason = "scaffolding for Messages API pipeline, wired in follow-up PR"
    )]
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via RequestType construction"
    )]
    pub fn messages_request(&self) -> &CreateMessageRequest {
        match &self.input.request_type {
            RequestType::Messages(req) => req.as_ref(),
            _ => panic!("Expected messages request"),
        }
    }

    /// Get Arc clone of messages request (panics if not messages)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via RequestType construction"
    )]
    pub fn messages_request_arc(&self) -> Arc<CreateMessageRequest> {
        match &self.input.request_type {
            RequestType::Messages(req) => Arc::clone(req),
            _ => panic!("Expected messages request"),
        }
    }

    /// Check if request is streaming
    pub fn is_streaming(&self) -> bool {
        match &self.input.request_type {
            RequestType::Chat(req) => req.stream,
            RequestType::Generate(req) => req.stream,
            RequestType::Completion(req) => req.stream,
            RequestType::Responses(req) => req.stream.unwrap_or(false),
            RequestType::Messages(req) => req.stream.unwrap_or(false),
            RequestType::Embedding(_) => false, // Embeddings are never streaming
            RequestType::Classify(_) => false,  // Classification is never streaming
        }
    }

    /// Get the cached tokenizer, cloning the Arc (cheap 8-byte clone)
    ///
    /// Returns None if tokenizer hasn't been resolved yet.
    /// The tokenizer is resolved once in the preparation stage and cached for reuse.
    pub fn tokenizer_arc(&self) -> Option<Arc<dyn Tokenizer>> {
        self.state.tokenizer.clone()
    }
}

/// Some methods are kept for API completeness even if currently unused.
#[expect(dead_code)]
impl WorkerSelection {
    pub fn is_dual(&self) -> bool {
        matches!(self, Self::Dual { .. })
    }

    pub fn single(&self) -> Option<&Arc<dyn Worker>> {
        match self {
            Self::Single { worker } => Some(worker),
            Self::Dual { .. } => None,
        }
    }

    /// Record circuit breaker outcome for all workers based on HTTP status code.
    pub fn record_outcome(&self, status_code: u16) {
        match self {
            Self::Single { worker } => worker.record_outcome(status_code),
            Self::Dual {
                prefill, decode, ..
            } => {
                prefill.record_outcome(status_code);
                decode.record_outcome(status_code);
            }
        }
    }

    /// Record circuit breaker outcomes for dual dispatch (individual tracking)
    pub fn record_dual_outcomes(&self, prefill_status: u16, decode_status: u16) {
        if let Self::Dual {
            prefill, decode, ..
        } = self
        {
            prefill.record_outcome(prefill_status);
            decode.record_outcome(decode_status);
        }
    }

    /// Record circuit breaker outcome for prefill worker only (sequential PD)
    pub fn record_outcome_prefill(&self, status_code: u16) {
        match self {
            Self::Dual { prefill, .. } => prefill.record_outcome(status_code),
            Self::Single { .. } => {
                debug!("record_outcome_prefill called on Single worker selection, ignoring");
            }
        }
    }

    /// Record circuit breaker outcome for decode worker only (sequential PD)
    pub fn record_outcome_decode(&self, status_code: u16) {
        match self {
            Self::Dual { decode, .. } => decode.record_outcome(status_code),
            Self::Single { .. } => {
                debug!("record_outcome_decode called on Single worker selection, ignoring");
            }
        }
    }

    #[expect(clippy::type_complexity)]
    pub fn dual(&self) -> Option<(&Arc<dyn Worker>, &Arc<dyn Worker>)> {
        match self {
            Self::Dual {
                prefill, decode, ..
            } => Some((prefill, decode)),
            Self::Single { .. } => None,
        }
    }

    pub fn prefill_worker(&self) -> Option<&Arc<dyn Worker>> {
        match self {
            Self::Dual { prefill, .. } => Some(prefill),
            Self::Single { .. } => None,
        }
    }

    pub fn decode_worker(&self) -> Option<&Arc<dyn Worker>> {
        match self {
            Self::Dual { decode, .. } => Some(decode),
            Self::Single { .. } => None,
        }
    }

    /// Get the runtime type for PD mode (from dual workers)
    pub fn pd_runtime_type(&self) -> Option<&RuntimeType> {
        match self {
            Self::Dual { runtime_type, .. } => Some(runtime_type),
            Self::Single { .. } => None,
        }
    }
}

/// Some methods are kept for API completeness even if currently unused.
#[expect(dead_code)]
impl ClientSelection {
    pub fn single(&self) -> Option<&GrpcClient> {
        match self {
            Self::Single { client } => Some(client),
            Self::Dual { .. } => None,
        }
    }

    pub fn single_mut(&mut self) -> Option<&mut GrpcClient> {
        match self {
            Self::Single { client } => Some(client),
            Self::Dual { .. } => None,
        }
    }

    pub fn dual_mut(&mut self) -> Option<(&mut GrpcClient, &mut GrpcClient)> {
        match self {
            Self::Dual { prefill, decode } => Some((prefill, decode)),
            Self::Single { .. } => None,
        }
    }

    pub fn prefill_client(&self) -> Option<&GrpcClient> {
        match self {
            Self::Dual { prefill, .. } => Some(prefill),
            Self::Single { .. } => None,
        }
    }

    pub fn prefill_client_mut(&mut self) -> Option<&mut GrpcClient> {
        match self {
            Self::Dual { prefill, .. } => Some(prefill),
            Self::Single { .. } => None,
        }
    }

    pub fn decode_client(&self) -> Option<&GrpcClient> {
        match self {
            Self::Dual { decode, .. } => Some(decode),
            Self::Single { .. } => None,
        }
    }

    pub fn decode_client_mut(&mut self) -> Option<&mut GrpcClient> {
        match self {
            Self::Dual { decode, .. } => Some(decode),
            Self::Single { .. } => None,
        }
    }
}

/// Result of request execution (streams from workers)
/// Uses ProtoStream to automatically abort on cancellation
pub(crate) enum ExecutionResult {
    Single {
        stream: ProtoStream,
    },
    Dual {
        prefill: ProtoStream,
        decode: Box<ProtoStream>,
    },
    /// Embedding requests return a single response, not a stream
    Embedding {
        response: ProtoEmbedComplete,
    },
}

/// Final processed response
#[derive(Debug)]
pub(crate) enum FinalResponse {
    Chat(ChatCompletionResponse),
    /// Generate response is a Vec of GenerateResponse (n=1 returns single item, n>1 returns multiple)
    Generate(Vec<GenerateResponse>),
    /// Completion response (OpenAI /v1/completions format)
    Completion(CompletionResponse),
    /// Embedding response
    Embedding(EmbeddingResponse),
    /// Classification response
    Classify(ClassifyResponse),
    /// Messages API response
    Messages(Message),
}
