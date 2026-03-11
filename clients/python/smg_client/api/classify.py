"""Classify API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from smg_client.types import ClassifyResponse

if TYPE_CHECKING:
    from smg_client._transport import AsyncTransport, SyncTransport


class SyncClassify:
    """Synchronous classify API."""

    def __init__(self, transport: SyncTransport) -> None:
        self._transport = transport

    def create(self, **kwargs: Any) -> ClassifyResponse:
        resp = self._transport.request("POST", "/v1/classify", json=kwargs)
        return ClassifyResponse.model_validate_json(resp.content)


class AsyncClassify:
    """Async classify API."""

    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    async def create(self, **kwargs: Any) -> ClassifyResponse:
        resp = await self._transport.request("POST", "/v1/classify", json=kwargs)
        return ClassifyResponse.model_validate_json(resp.content)
