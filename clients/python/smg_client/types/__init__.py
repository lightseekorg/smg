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
        ClassifyData,
        ClassifyRequest,
        ClassifyResponse,
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
        ParseFunctionCallRequest,
        ParseFunctionCallResponse,
        RerankRequest,
        RerankResponse,
        RerankResult,
        ResponsesRequest,
        ResponsesResponse,
        SeparateReasoningRequest,
        SeparateReasoningResponse,
        StreamOptions,
        Tool,
        ToolCall,
        Usage,
        UsageInfo,
        VideoUrl,
        WorkerInfo,
        WorkerSpec,
        WorkerUpdateRequest,
    )
except ModuleNotFoundError as _e:
    raise ModuleNotFoundError(
        "smg_client.types._generated not found. "
        "Run 'make generate-clients' to generate the types module."
    ) from _e


# ---------------------------------------------------------------------------
# Extend auto-generated types with SDK-compatible properties
# ---------------------------------------------------------------------------


@property  # type: ignore[misc]
def _responses_output_text(self: ResponsesResponse) -> str:
    """Concatenate text from all output_text content parts (OpenAI SDK compat).

    Iterates over output items of type "message" and collects text from
    content parts whose type is "output_text".
    """
    texts: list[str] = []
    for item in self.output or []:
        if (
            getattr(item, "type", None)
            and str(item.type.value if hasattr(item.type, "value") else item.type) == "message"
        ):
            for part in getattr(item, "content", []):
                part_type = getattr(part, "type", None)
                type_str = (
                    str(part_type.value if hasattr(part_type, "value") else part_type)
                    if part_type
                    else ""
                )
                if type_str == "output_text":
                    text = getattr(part, "text", None)
                    if text is not None:
                        texts.append(text)
    return "".join(texts)


ResponsesResponse.output_text = _responses_output_text
