"""Rerank API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from smg_client.types import RerankResponse

if TYPE_CHECKING:
    from smg_client._transport import AsyncTransport, SyncTransport


class SyncRerank:
    """Synchronous rerank API."""

    def __init__(self, transport: SyncTransport) -> None:
        self._transport = transport

    def create(self, **kwargs: Any) -> RerankResponse:
        resp = self._transport.request("POST", "/v1/rerank", json=kwargs)
        return RerankResponse.model_validate_json(resp.content)


class AsyncRerank:
    """Async rerank API."""

    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    async def create(self, **kwargs: Any) -> RerankResponse:
        resp = await self._transport.request("POST", "/v1/rerank", json=kwargs)
        return RerankResponse.model_validate_json(resp.content)
