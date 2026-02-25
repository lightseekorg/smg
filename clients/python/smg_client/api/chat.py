"""Chat completions API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from smg_client._streaming import AsyncStream, SyncStream
from smg_client.types import ChatCompletionResponse, ChatCompletionStreamResponse

if TYPE_CHECKING:
    from smg_client._transport import AsyncTransport, SyncTransport


class SyncCompletions:
    """Synchronous chat completions API."""

    def __init__(self, transport: SyncTransport) -> None:
        self._transport = transport

    def create(self, **kwargs: Any) -> ChatCompletionResponse:
        """Create a chat completion.

        Args:
            **kwargs: ChatCompletionRequest fields (model, messages, temperature, etc.)

        Returns:
            ChatCompletionResponse for non-streaming requests.
        """
        kwargs.setdefault("stream", False)
        resp = self._transport.request("POST", "/v1/chat/completions", json=kwargs)
        return ChatCompletionResponse.model_validate_json(resp.content)

    def create_stream(self, **kwargs: Any) -> SyncStream[ChatCompletionStreamResponse]:
        """Create a streaming chat completion.

        Returns a context manager that yields ChatCompletionStreamResponse chunks.

        Usage::

            with client.chat.completions.create_stream(
                model="llama-3.1-8b",
                messages=[{"role": "user", "content": "Hello"}],
            ) as stream:
                for chunk in stream:
                    print(chunk.choices[0].delta.content, end="")
        """
        kwargs["stream"] = True
        resp = self._transport.request("POST", "/v1/chat/completions", json=kwargs, stream=True)
        return SyncStream(resp, ChatCompletionStreamResponse)


class AsyncCompletions:
    """Async chat completions API."""

    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    async def create(self, **kwargs: Any) -> ChatCompletionResponse:
        """Create a chat completion."""
        kwargs.setdefault("stream", False)
        resp = await self._transport.request("POST", "/v1/chat/completions", json=kwargs)
        return ChatCompletionResponse.model_validate_json(resp.content)

    async def create_stream(self, **kwargs: Any) -> AsyncStream[ChatCompletionStreamResponse]:
        """Create a streaming chat completion."""
        kwargs["stream"] = True
        resp = await self._transport.request(
            "POST", "/v1/chat/completions", json=kwargs, stream=True
        )
        return AsyncStream(resp, ChatCompletionStreamResponse)


class SyncChat:
    """Synchronous chat namespace (chat.completions)."""

    def __init__(self, transport: SyncTransport) -> None:
        self.completions = SyncCompletions(transport)


class AsyncChat:
    """Async chat namespace (chat.completions)."""

    def __init__(self, transport: AsyncTransport) -> None:
        self.completions = AsyncCompletions(transport)
