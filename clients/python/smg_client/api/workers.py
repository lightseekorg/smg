"""Workers API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from smg_client.types import WorkerApiResponse, WorkerInfo, WorkerListResponse

if TYPE_CHECKING:
    from smg_client._transport import AsyncTransport, SyncTransport


class SyncWorkers:
    """Synchronous workers API."""

    def __init__(self, transport: SyncTransport) -> None:
        self._transport = transport

    def create(self, **kwargs: Any) -> WorkerApiResponse:
        resp = self._transport.request("POST", "/workers", json=kwargs)
        return WorkerApiResponse.model_validate_json(resp.content)

    def list(self) -> WorkerListResponse:
        resp = self._transport.request("GET", "/workers")
        return WorkerListResponse.model_validate_json(resp.content)

    def get(self, worker_id: str) -> WorkerInfo:
        resp = self._transport.request("GET", f"/workers/{worker_id}")
        return WorkerInfo.model_validate_json(resp.content)

    def update(self, worker_id: str, **kwargs: Any) -> WorkerApiResponse:
        resp = self._transport.request("PUT", f"/workers/{worker_id}", json=kwargs)
        return WorkerApiResponse.model_validate_json(resp.content)

    def delete(self, worker_id: str) -> WorkerApiResponse:
        resp = self._transport.request("DELETE", f"/workers/{worker_id}")
        return WorkerApiResponse.model_validate_json(resp.content)


class AsyncWorkers:
    """Async workers API."""

    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    async def create(self, **kwargs: Any) -> WorkerApiResponse:
        resp = await self._transport.request("POST", "/workers", json=kwargs)
        return WorkerApiResponse.model_validate_json(resp.content)

    async def list(self) -> WorkerListResponse:
        resp = await self._transport.request("GET", "/workers")
        return WorkerListResponse.model_validate_json(resp.content)

    async def get(self, worker_id: str) -> WorkerInfo:
        resp = await self._transport.request("GET", f"/workers/{worker_id}")
        return WorkerInfo.model_validate_json(resp.content)

    async def update(self, worker_id: str, **kwargs: Any) -> WorkerApiResponse:
        resp = await self._transport.request(
            "PUT", f"/workers/{worker_id}", json=kwargs
        )
        return WorkerApiResponse.model_validate_json(resp.content)

    async def delete(self, worker_id: str) -> WorkerApiResponse:
        resp = await self._transport.request("DELETE", f"/workers/{worker_id}")
        return WorkerApiResponse.model_validate_json(resp.content)
