"""Parser API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from smg_client.types import ParseFunctionCallResponse, SeparateReasoningResponse

if TYPE_CHECKING:
    from smg_client._transport import AsyncTransport, SyncTransport


class SyncParser:
    """Synchronous parser API."""

    def __init__(self, transport: SyncTransport) -> None:
        self._transport = transport

    def parse_function_call(self, **kwargs: Any) -> ParseFunctionCallResponse:
        resp = self._transport.request("POST", "/parse/function_call", json=kwargs)
        return ParseFunctionCallResponse.model_validate_json(resp.content)

    def separate_reasoning(self, **kwargs: Any) -> SeparateReasoningResponse:
        resp = self._transport.request("POST", "/parse/reasoning", json=kwargs)
        return SeparateReasoningResponse.model_validate_json(resp.content)


class AsyncParser:
    """Async parser API."""

    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    async def parse_function_call(self, **kwargs: Any) -> ParseFunctionCallResponse:
        resp = await self._transport.request("POST", "/parse/function_call", json=kwargs)
        return ParseFunctionCallResponse.model_validate_json(resp.content)

    async def separate_reasoning(self, **kwargs: Any) -> SeparateReasoningResponse:
        resp = await self._transport.request("POST", "/parse/reasoning", json=kwargs)
        return SeparateReasoningResponse.model_validate_json(resp.content)
