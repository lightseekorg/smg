"""SMG client type definitions.

All types are auto-generated from the Rust `protocols` crate via the OpenAPI spec.
Run ``make generate-clients`` to regenerate.
"""

# ruff: noqa: F401 — re-exports
try:
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
except ModuleNotFoundError as _e:
    raise ModuleNotFoundError(
        "smg_client.types._generated not found. "
        "Run 'make generate-clients' to generate the types module."
    ) from _e
