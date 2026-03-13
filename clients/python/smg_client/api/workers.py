"""Workers API.

Note: worker mutation endpoints (create/update/delete) and list return raw
dicts because the server response shapes differ from the protocol types.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from smg_client.types import WorkerInfo

if TYPE_CHECKING:
    from smg_client._transport import AsyncTransport, SyncTransport


class SyncWorkers:
    """Synchronous workers API."""

    def __init__(self, transport: SyncTransport) -> None:
        self._transport = transport

    def create(self, **kwargs: Any) -> Any:
        """Register a new worker. Returns 202 with {status, worker_id, url, location}."""
        resp = self._transport.request("POST", "/workers", json=kwargs)
        return resp.json()

    def list(self) -> Any:
        """List all workers. Returns {workers, total, stats}."""
        resp = self._transport.request("GET", "/workers")
        return resp.json()

    def get(self, worker_id: str) -> WorkerInfo:
        resp = self._transport.request("GET", f"/workers/{worker_id}")
        return WorkerInfo.model_validate_json(resp.content)

    def update(self, worker_id: str, **kwargs: Any) -> Any:
        """Update a worker. Returns 202 with {status, worker_id, message}."""
        resp = self._transport.request("PUT", f"/workers/{worker_id}", json=kwargs)
        return resp.json()

    def delete(self, worker_id: str) -> Any:
        """Remove a worker. Returns 202 with {status, worker_id, message}."""
        resp = self._transport.request("DELETE", f"/workers/{worker_id}")
        return resp.json()


class AsyncWorkers:
    """Async workers API."""

    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    async def create(self, **kwargs: Any) -> Any:
        """Register a new worker. Returns 202 with {status, worker_id, url, location}."""
        resp = await self._transport.request("POST", "/workers", json=kwargs)
        return resp.json()

    async def list(self) -> Any:
        """List all workers. Returns {workers, total, stats}."""
        resp = await self._transport.request("GET", "/workers")
        return resp.json()

    async def get(self, worker_id: str) -> WorkerInfo:
        resp = await self._transport.request("GET", f"/workers/{worker_id}")
        return WorkerInfo.model_validate_json(resp.content)

    async def update(self, worker_id: str, **kwargs: Any) -> Any:
        """Update a worker. Returns 202 with {status, worker_id, message}."""
        resp = await self._transport.request("PUT", f"/workers/{worker_id}", json=kwargs)
        return resp.json()

    async def delete(self, worker_id: str) -> Any:
        """Remove a worker. Returns 202 with {status, worker_id, message}."""
        resp = await self._transport.request("DELETE", f"/workers/{worker_id}")
        return resp.json()
