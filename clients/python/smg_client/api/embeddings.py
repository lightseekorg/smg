"""Embeddings API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from smg_client.types import EmbeddingResponse

if TYPE_CHECKING:
    from smg_client._transport import AsyncTransport, SyncTransport


class SyncEmbeddings:
    """Synchronous embeddings API."""

    def __init__(self, transport: SyncTransport) -> None:
        self._transport = transport

    def create(self, **kwargs: Any) -> EmbeddingResponse:
        resp = self._transport.request("POST", "/v1/embeddings", json=kwargs)
        return EmbeddingResponse.model_validate_json(resp.content)


class AsyncEmbeddings:
    """Async embeddings API."""

    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    async def create(self, **kwargs: Any) -> EmbeddingResponse:
        resp = await self._transport.request("POST", "/v1/embeddings", json=kwargs)
        return EmbeddingResponse.model_validate_json(resp.content)
