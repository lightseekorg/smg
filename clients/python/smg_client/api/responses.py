"""Responses API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from smg_client.types import ResponsesResponse

if TYPE_CHECKING:
    from smg_client._transport import AsyncTransport, SyncTransport


class SyncResponses:
    """Synchronous responses API."""

    def __init__(self, transport: SyncTransport) -> None:
        self._transport = transport

    def create(self, **kwargs: Any) -> ResponsesResponse:
        resp = self._transport.request("POST", "/v1/responses", json=kwargs)
        return ResponsesResponse.model_validate_json(resp.content)

    def get(self, response_id: str) -> ResponsesResponse:
        resp = self._transport.request("GET", f"/v1/responses/{response_id}")
        return ResponsesResponse.model_validate_json(resp.content)

    def delete(self, response_id: str) -> None:
        """Delete a response. Note: not yet implemented on the server (returns 501)."""
        self._transport.request("DELETE", f"/v1/responses/{response_id}")

    def cancel(self, response_id: str) -> ResponsesResponse:
        resp = self._transport.request("POST", f"/v1/responses/{response_id}/cancel", json={})
        return ResponsesResponse.model_validate_json(resp.content)

    def list_input_items(self, response_id: str) -> Any:
        resp = self._transport.request("GET", f"/v1/responses/{response_id}/input_items")
        return resp.json()


class AsyncResponses:
    """Async responses API."""

    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    async def create(self, **kwargs: Any) -> ResponsesResponse:
        resp = await self._transport.request("POST", "/v1/responses", json=kwargs)
        return ResponsesResponse.model_validate_json(resp.content)

    async def get(self, response_id: str) -> ResponsesResponse:
        resp = await self._transport.request("GET", f"/v1/responses/{response_id}")
        return ResponsesResponse.model_validate_json(resp.content)

    async def delete(self, response_id: str) -> None:
        """Delete a response. Note: not yet implemented on the server (returns 501)."""
        await self._transport.request("DELETE", f"/v1/responses/{response_id}")

    async def cancel(self, response_id: str) -> ResponsesResponse:
        resp = await self._transport.request("POST", f"/v1/responses/{response_id}/cancel", json={})
        return ResponsesResponse.model_validate_json(resp.content)

    async def list_input_items(self, response_id: str) -> Any:
        resp = await self._transport.request("GET", f"/v1/responses/{response_id}/input_items")
        return resp.json()
