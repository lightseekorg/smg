"""Models API."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from smg_client._transport import AsyncTransport, SyncTransport


class ModelObject(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    object: str = "model"
    created: int | None = None
    owned_by: str | None = None
    root: str | None = None


class ModelList(BaseModel):
    model_config = ConfigDict(extra="allow")

    object: str = "list"
    data: list[ModelObject]


class SyncModels:
    """Synchronous models API."""

    def __init__(self, transport: SyncTransport) -> None:
        self._transport = transport

    def list(self) -> ModelList:
        resp = self._transport.request("GET", "/v1/models")
        return ModelList.model_validate_json(resp.content)


class AsyncModels:
    """Async models API."""

    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    async def list(self) -> ModelList:
        resp = await self._transport.request("GET", "/v1/models")
        return ModelList.model_validate_json(resp.content)
