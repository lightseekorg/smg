"""Anthropic Messages API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from smg_client._helpers import prepare_body
from smg_client._streaming import AnthropicAsyncStream, AnthropicSyncStream
from smg_client.types import Message

if TYPE_CHECKING:
    from smg_client._transport import AsyncTransport, SyncTransport


class SyncMessages:
    """Synchronous Anthropic Messages API.

    Matches the Anthropic SDK interface::

        # Non-streaming
        msg = client.messages.create(model="claude-3-5-sonnet", ...)

        # Streaming
        with client.messages.stream(model="claude-3-5-sonnet", ...) as stream:
            for event in stream:
                if event.type == "content_block_delta":
                    print(event.delta.text)
            text = stream.get_final_text()
    """

    def __init__(self, transport: SyncTransport) -> None:
        self._transport = transport

    def create(self, **kwargs: Any) -> Message:
        """Create a message (non-streaming).

        Args:
            **kwargs: CreateMessageRequest fields (model, messages, max_tokens, etc.)
                extra_body: Dict merged into the request body.
                extra_headers: Dict merged into request headers.
        """
        kwargs.pop("stream", None)
        body, extra_headers = prepare_body(kwargs)
        body["stream"] = False
        resp = self._transport.request("POST", "/v1/messages", json=body, headers=extra_headers)
        return Message.model_validate_json(resp.content)

    def stream(self, **kwargs: Any) -> AnthropicSyncStream:
        """Create a streaming message.

        Returns an AnthropicSyncStream context manager matching the Anthropic SDK::

            with client.messages.stream(
                model="claude-3-5-sonnet",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hello"}],
            ) as stream:
                for event in stream:
                    if event.type == "content_block_delta":
                        print(event.delta.text, end="")
                text = stream.get_final_text()

        Args:
            **kwargs: CreateMessageRequest fields.
                extra_body: Dict merged into the request body.
                extra_headers: Dict merged into request headers.
        """
        kwargs.pop("stream", None)
        body, extra_headers = prepare_body(kwargs)
        body["stream"] = True
        resp = self._transport.request(
            "POST", "/v1/messages", json=body, stream=True, headers=extra_headers
        )
        return AnthropicSyncStream(resp)

    def create_stream(self, **kwargs: Any) -> AnthropicSyncStream:
        """Create a streaming message (backward-compat alias).

        Prefer ``stream(...)`` for Anthropic SDK compatibility.
        """
        return self.stream(**kwargs)


class AsyncMessages:
    """Async Anthropic Messages API."""

    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    async def create(self, **kwargs: Any) -> Message:
        """Create a message (non-streaming)."""
        kwargs.pop("stream", None)
        body, extra_headers = prepare_body(kwargs)
        body["stream"] = False
        resp = await self._transport.request(
            "POST", "/v1/messages", json=body, headers=extra_headers
        )
        return Message.model_validate_json(resp.content)

    async def stream(self, **kwargs: Any) -> AnthropicAsyncStream:
        """Create a streaming message (Anthropic SDK compat)."""
        kwargs.pop("stream", None)
        body, extra_headers = prepare_body(kwargs)
        body["stream"] = True
        resp = await self._transport.request(
            "POST", "/v1/messages", json=body, stream=True, headers=extra_headers
        )
        return AnthropicAsyncStream(resp)

    async def create_stream(self, **kwargs: Any) -> AnthropicAsyncStream:
        """Create a streaming message (backward-compat alias)."""
        return await self.stream(**kwargs)
