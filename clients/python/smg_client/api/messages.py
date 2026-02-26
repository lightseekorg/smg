"""Anthropic Messages API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from smg_client._streaming import AnthropicAsyncStream, AnthropicSyncStream
from smg_client.types import Message

if TYPE_CHECKING:
    from smg_client._transport import AsyncTransport, SyncTransport


class SyncMessages:
    """Synchronous Anthropic Messages API."""

    def __init__(self, transport: SyncTransport) -> None:
        self._transport = transport

    def create(self, **kwargs: Any) -> Message:
        """Create a message.

        Args:
            **kwargs: CreateMessageRequest fields (model, messages, max_tokens, etc.)
        """
        kwargs["stream"] = False
        resp = self._transport.request("POST", "/v1/messages", json=kwargs)
        return Message.model_validate_json(resp.content)

    def create_stream(self, **kwargs: Any) -> AnthropicSyncStream:
        """Create a streaming message.

        Returns an AnthropicSyncStream that yields event dicts with a ``type`` field.

        Usage::

            with client.messages.create_stream(
                model="claude-3-5-sonnet",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hello"}],
            ) as stream:
                for event in stream:
                    if event["type"] == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            print(delta["text"], end="")
        """
        kwargs["stream"] = True
        resp = self._transport.request("POST", "/v1/messages", json=kwargs, stream=True)
        return AnthropicSyncStream(resp)


class AsyncMessages:
    """Async Anthropic Messages API."""

    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    async def create(self, **kwargs: Any) -> Message:
        kwargs["stream"] = False
        resp = await self._transport.request("POST", "/v1/messages", json=kwargs)
        return Message.model_validate_json(resp.content)

    async def create_stream(self, **kwargs: Any) -> AnthropicAsyncStream:
        kwargs["stream"] = True
        resp = await self._transport.request("POST", "/v1/messages", json=kwargs, stream=True)
        return AnthropicAsyncStream(resp)
