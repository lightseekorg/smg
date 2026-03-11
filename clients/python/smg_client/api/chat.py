"""Chat completions API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload

from smg_client._helpers import prepare_body
from smg_client._streaming import AsyncStream, SyncStream
from smg_client.types import ChatCompletionResponse, ChatCompletionStreamResponse

if TYPE_CHECKING:
    from smg_client._transport import AsyncTransport, SyncTransport


class SyncCompletions:
    """Synchronous chat completions API.

    Matches the OpenAI SDK interface::

        # Non-streaming
        resp = client.chat.completions.create(model="...", messages=[...])

        # Streaming
        stream = client.chat.completions.create(model="...", messages=[...], stream=True)
        for chunk in stream:
            print(chunk.choices[0].delta.content)
    """

    def __init__(self, transport: SyncTransport) -> None:
        self._transport = transport

    @overload
    def create(self, *, stream: Literal[False] = ..., **kwargs: Any) -> ChatCompletionResponse: ...
    @overload
    def create(
        self, *, stream: Literal[True], **kwargs: Any
    ) -> SyncStream[ChatCompletionStreamResponse]: ...

    def create(
        self, **kwargs: Any
    ) -> ChatCompletionResponse | SyncStream[ChatCompletionStreamResponse]:
        """Create a chat completion.

        Args:
            **kwargs: ChatCompletionRequest fields (model, messages, temperature, etc.)
                stream: If True, returns a SyncStream of chunks. Default False.
                extra_body: Dict merged into the request body (OpenAI SDK compat).
                extra_headers: Dict merged into request headers.

        Returns:
            ChatCompletionResponse for non-streaming, SyncStream for streaming.
        """
        is_stream = kwargs.pop("stream", False)
        body, extra_headers = prepare_body(kwargs)
        body["stream"] = is_stream

        if is_stream:
            resp = self._transport.request(
                "POST",
                "/v1/chat/completions",
                json=body,
                stream=True,
                headers=extra_headers,
            )
            return SyncStream(resp, ChatCompletionStreamResponse)

        resp = self._transport.request(
            "POST",
            "/v1/chat/completions",
            json=body,
            headers=extra_headers,
        )
        return ChatCompletionResponse.model_validate_json(resp.content)

    def create_stream(self, **kwargs: Any) -> SyncStream[ChatCompletionStreamResponse]:
        """Create a streaming chat completion (backward-compat alias).

        Prefer ``create(stream=True, ...)`` for OpenAI SDK compatibility.
        """
        kwargs["stream"] = True
        return self.create(**kwargs)  # type: ignore[return-value]


class AsyncCompletions:
    """Async chat completions API."""

    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    @overload
    async def create(
        self, *, stream: Literal[False] = ..., **kwargs: Any
    ) -> ChatCompletionResponse: ...
    @overload
    async def create(
        self, *, stream: Literal[True], **kwargs: Any
    ) -> AsyncStream[ChatCompletionStreamResponse]: ...

    async def create(
        self, **kwargs: Any
    ) -> ChatCompletionResponse | AsyncStream[ChatCompletionStreamResponse]:
        """Create a chat completion."""
        is_stream = kwargs.pop("stream", False)
        body, extra_headers = prepare_body(kwargs)
        body["stream"] = is_stream

        if is_stream:
            resp = await self._transport.request(
                "POST",
                "/v1/chat/completions",
                json=body,
                stream=True,
                headers=extra_headers,
            )
            return AsyncStream(resp, ChatCompletionStreamResponse)

        resp = await self._transport.request(
            "POST",
            "/v1/chat/completions",
            json=body,
            headers=extra_headers,
        )
        return ChatCompletionResponse.model_validate_json(resp.content)

    async def create_stream(self, **kwargs: Any) -> AsyncStream[ChatCompletionStreamResponse]:
        """Create a streaming chat completion (backward-compat alias)."""
        kwargs["stream"] = True
        return await self.create(**kwargs)  # type: ignore[return-value]


class SyncChat:
    """Synchronous chat namespace (chat.completions)."""

    def __init__(self, transport: SyncTransport) -> None:
        self.completions = SyncCompletions(transport)


class AsyncChat:
    """Async chat namespace (chat.completions)."""

    def __init__(self, transport: AsyncTransport) -> None:
        self.completions = AsyncCompletions(transport)
