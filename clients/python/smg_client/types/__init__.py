"""SMG client type definitions.

All types are auto-generated from the Rust `protocols` crate via the OpenAPI spec.
Run `cargo run -p openapi-gen` then `datamodel-codegen` to regenerate.
"""

# ruff: noqa: F401 — re-exports
from smg_client.types._generated import (
    ChatChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ChatMessageDelta,
    ChatStreamChoice,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    CompletionStreamChoice,
    CompletionStreamResponse,
    CreateMessageRequest,
    EmbeddingObject,
    EmbeddingRequest,
    EmbeddingResponse,
    ErrorDetail,
    ErrorResponse,
    Function,
    ImageUrl,
    InputMessage,
    JsonSchemaFormat,
    Message,
    MessageDelta,
    MessageDeltaUsage,
    MessageStreamEvent,
    RerankRequest,
    RerankResponse,
    RerankResult,
    StreamOptions,
    Tool,
    ToolCall,
    Usage,
    UsageInfo,
    VideoUrl,
)
