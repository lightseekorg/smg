"""Legacy completions API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from smg_client._streaming import AsyncStream, SyncStream
from smg_client.types import CompletionResponse, CompletionStreamResponse

if TYPE_CHECKING:
    from smg_client._transport import AsyncTransport, SyncTransport


class SyncCompletions:
    """Synchronous completions API."""

    def __init__(self, transport: SyncTransport) -> None:
        self._transport = transport

    def create(self, **kwargs: Any) -> CompletionResponse:
        kwargs.setdefault("stream", False)
        resp = self._transport.request("POST", "/v1/completions", json=kwargs)
        return CompletionResponse.model_validate_json(resp.content)

    def create_stream(self, **kwargs: Any) -> SyncStream[CompletionStreamResponse]:
        kwargs["stream"] = True
        resp = self._transport.request("POST", "/v1/completions", json=kwargs, stream=True)
        return SyncStream(resp, CompletionStreamResponse)


class AsyncCompletions:
    """Async completions API."""

    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    async def create(self, **kwargs: Any) -> CompletionResponse:
        kwargs.setdefault("stream", False)
        resp = await self._transport.request("POST", "/v1/completions", json=kwargs)
        return CompletionResponse.model_validate_json(resp.content)

    async def create_stream(self, **kwargs: Any) -> AsyncStream[CompletionStreamResponse]:
        kwargs["stream"] = True
        resp = await self._transport.request("POST", "/v1/completions", json=kwargs, stream=True)
        return AsyncStream(resp, CompletionStreamResponse)
