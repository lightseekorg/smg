"""Models API."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from smg_client._transport import AsyncTransport, SyncTransport


class ModelObject(BaseModel):
    """A model object returned by ``/v1/models``.

    Accepts both OpenAI format (``object``, ``created``, ``owned_by``) and
    Anthropic format (``display_name``, ``created_at``).
    """

    model_config = ConfigDict(extra="allow")

    id: str
    object: str = "model"
    created: int | None = None
    owned_by: str | None = None
    root: str | None = None
    display_name: str | None = None
    created_at: str | None = None


class ModelList(BaseModel):
    """Response from ``GET /v1/models``.

    Accepts both OpenAI format (``object: "list"``) and Anthropic format
    (``has_more``, ``first_id``, ``last_id`` pagination fields).
    """

    model_config = ConfigDict(extra="allow")

    object: str = "list"
    data: list[ModelObject]
    has_more: bool | None = None
    first_id: str | None = None
    last_id: str | None = None


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
