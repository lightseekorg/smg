"""Typed stream wrappers for different API streaming protocols."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypeVar

from smg_client._errors import SmgError
from smg_client._sse import iter_sse_async, iter_sse_sync

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    import httpx

T = TypeVar("T")


class SyncStream(Generic[T]):
    """Synchronous typed stream over SSE events.

    Works as both an iterator and a context manager, matching the OpenAI SDK::

        # As iterator (OpenAI SDK style):
        stream = client.chat.completions.create(stream=True, ...)
        for chunk in stream:
            print(chunk.choices[0].delta.content)

        # As context manager:
        with client.chat.completions.create(stream=True, ...) as stream:
            for chunk in stream:
                print(chunk.choices[0].delta.content)
    """

    def __init__(self, response: httpx.Response, model_cls: type[T]) -> None:
        self._response = response
        self._model_cls = model_cls
        self._iterator = iter_sse_sync(response)

    def __iter__(self) -> Iterator[T]:
        return self

    def __next__(self) -> T:
        try:
            event = next(self._iterator)
        except StopIteration:
            self.close()
            raise
        return self._model_cls.model_validate_json(event.data)

    def __enter__(self) -> SyncStream[T]:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self._response.close()


class AsyncStream(Generic[T]):
    """Async typed stream over SSE events.

    Usage::

        stream = await client.chat.completions.create(stream=True, ...)
        async with stream:
            async for chunk in stream:
                print(chunk.choices[0].delta.content)
    """

    def __init__(self, response: httpx.Response, model_cls: type[T]) -> None:
        self._response = response
        self._model_cls = model_cls
        self._iterator = iter_sse_async(response)

    def __aiter__(self) -> AsyncIterator[T]:
        return self

    async def __anext__(self) -> T:
        try:
            event = await self._iterator.__anext__()
        except StopAsyncIteration:
            await self.aclose()
            raise
        return self._model_cls.model_validate_json(event.data)

    async def __aenter__(self) -> AsyncStream[T]:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._response.aclose()


# ---------------------------------------------------------------------------
# Anthropic Messages streaming
# ---------------------------------------------------------------------------


class EventObject:
    """Recursively converts a dict into an attribute-accessible object.

    Matches the Anthropic SDK event interface where events have dotted
    attribute access (``event.type``, ``event.delta.text``, etc.)
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, EventObject(value))
            elif isinstance(value, list):
                setattr(
                    self,
                    key,
                    [EventObject(item) if isinstance(item, dict) else item for item in value],
                )
            else:
                setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"EventObject({self._data!r})"


def _parse_anthropic_event(event: Any) -> EventObject:
    """Parse an SSE event into an EventObject for the Anthropic stream."""
    try:
        data = event.json()
    except ValueError as e:
        raise SmgError(f"Failed to parse SSE event data as JSON: {e}") from e
    if not isinstance(data, dict):
        raise SmgError(f"Expected JSON object in SSE event, got {type(data).__name__}")
    # Anthropic events carry the type in the SSE event: field.
    # This is authoritative — always override the JSON body's type.
    if event.event:
        data["type"] = event.event
    return EventObject(data)


class ResponsesSyncStream:
    """Synchronous stream for Responses API events.

    Matches the OpenAI SDK ``client.responses.create(stream=True)`` interface::

        resp = client.responses.create(model="...", input="Hello", stream=True)
        for event in resp:
            if event.type == "response.output_item.added":
                print(event.item)

    Works as both an iterator and a context manager.
    """

    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self._iterator = iter_sse_sync(response)

    def __iter__(self) -> Iterator[EventObject]:
        return self

    def __next__(self) -> EventObject:
        try:
            event = next(self._iterator)
        except StopIteration:
            self.close()
            raise
        try:
            data = event.json()
        except ValueError as e:
            raise SmgError(f"Failed to parse SSE event data as JSON: {e}") from e
        if not isinstance(data, dict):
            raise SmgError(f"Expected JSON object in SSE event, got {type(data).__name__}")
        if event.event:
            data["type"] = event.event
        return EventObject(data)

    def __enter__(self) -> ResponsesSyncStream:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self._response.close()


class ResponsesAsyncStream:
    """Async stream for Responses API events.

    Matches the OpenAI SDK async responses streaming interface.
    """

    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self._iterator = iter_sse_async(response)

    def __aiter__(self) -> AsyncIterator[EventObject]:
        return self

    async def __anext__(self) -> EventObject:
        try:
            event = await self._iterator.__anext__()
        except StopAsyncIteration:
            await self.aclose()
            raise
        try:
            data = event.json()
        except ValueError as e:
            raise SmgError(f"Failed to parse SSE event data as JSON: {e}") from e
        if not isinstance(data, dict):
            raise SmgError(f"Expected JSON object in SSE event, got {type(data).__name__}")
        if event.event:
            data["type"] = event.event
        return EventObject(data)

    async def __aenter__(self) -> ResponsesAsyncStream:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._response.aclose()


class AnthropicSyncStream:
    """Synchronous stream for Anthropic Messages API events.

    Matches the Anthropic SDK ``MessageStream`` interface::

        with client.messages.stream(
            model="claude-3-5-sonnet",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        ) as stream:
            for event in stream:
                if event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        print(event.delta.text, end="")

            text = stream.get_final_text()
            message = stream.get_final_message()
    """

    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self._iterator = iter_sse_sync(response)
        self._text_parts: list[str] = []
        self._content_blocks: dict[int, dict[str, Any]] = {}
        self._final_message_data: dict[str, Any] | None = None
        self._consumed = False

    def __iter__(self) -> Iterator[EventObject]:
        return self

    def __next__(self) -> EventObject:
        try:
            event = next(self._iterator)
        except StopIteration:
            self.close()
            raise
        obj = _parse_anthropic_event(event)
        self._accumulate(obj)
        return obj

    def _accumulate(self, obj: EventObject) -> None:
        """Track text deltas and final message data for get_final_text/get_final_message."""
        event_type = getattr(obj, "type", None)
        if event_type == "content_block_start":
            # Track all content blocks (text, tool_use, etc.) for get_final_message
            block = getattr(obj, "content_block", None)
            if block and self._final_message_data is not None:
                self._content_blocks[getattr(obj, "index", len(self._content_blocks))] = (
                    block._data.copy()
                )
        elif event_type == "content_block_delta":
            delta = getattr(obj, "delta", None)
            if delta and getattr(delta, "type", None) == "text_delta":
                text = getattr(delta, "text", "")
                if text:
                    self._text_parts.append(text)
                    # Update the corresponding content block
                    idx = getattr(obj, "index", None)
                    if idx is not None and idx in self._content_blocks:
                        block = self._content_blocks[idx]
                        block["text"] = block.get("text", "") + text
            elif delta and getattr(delta, "type", None) == "input_json_delta":
                # Accumulate tool input JSON for tool_use blocks
                idx = getattr(obj, "index", None)
                if idx is not None and idx in self._content_blocks:
                    block = self._content_blocks[idx]
                    partial = getattr(delta, "partial_json", "")
                    block.setdefault("_input_json_parts", []).append(partial)
        elif event_type == "message_start":
            msg = getattr(obj, "message", None)
            if msg:
                self._final_message_data = msg._data.copy()
                self._final_message_data["content"] = []
        elif event_type == "message_delta":
            delta = getattr(obj, "delta", None)
            if delta and self._final_message_data is not None:
                # Merge delta fields (stop_reason, etc.)
                self._final_message_data.update(delta._data)
            usage = getattr(obj, "usage", None)
            if usage and self._final_message_data is not None:
                self._final_message_data["usage"] = usage._data

    def _drain(self) -> None:
        """Consume remaining events if not fully iterated."""
        if not self._consumed:
            try:
                while True:
                    next(self)
            except StopIteration:
                pass
            self._consumed = True

    def get_final_text(self) -> str:
        """Return all accumulated text from text_delta events.

        If the stream has not been fully consumed, this will drain it first.
        """
        self._drain()
        return "".join(self._text_parts)

    def _finalize_content(self) -> None:
        """Build final message content from accumulated content blocks."""
        if self._final_message_data is None:
            return
        if self._content_blocks:
            import json as _json

            content = []
            for idx in sorted(self._content_blocks):
                block = self._content_blocks[idx]
                # Assemble tool_use input from accumulated JSON fragments
                parts = block.pop("_input_json_parts", None)
                if parts:
                    try:
                        block["input"] = _json.loads("".join(parts))
                    except (ValueError, TypeError):
                        block["input"] = {}
                content.append(block)
            self._final_message_data["content"] = content
        elif self._text_parts:
            # Fallback: if no content_block_start events were received
            self._final_message_data["content"] = [
                {"type": "text", "text": "".join(self._text_parts)}
            ]

    def get_final_message(self) -> Any:
        """Return the final message assembled from stream events.

        Returns a Message object if the types module is available,
        otherwise returns an EventObject.
        """
        self._drain()
        if self._final_message_data is None:
            raise SmgError("No message_start event received in stream")

        # Build content from accumulated content blocks
        self._finalize_content()

        try:
            from smg_client.types import Message

            return Message.model_validate(self._final_message_data)
        except (ImportError, ValueError):
            return EventObject(self._final_message_data)

    def __enter__(self) -> AnthropicSyncStream:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self._response.close()


class AnthropicAsyncStream:
    """Async stream for Anthropic Messages API events.

    Matches the Anthropic SDK async ``MessageStream`` interface.
    """

    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self._iterator = iter_sse_async(response)
        self._text_parts: list[str] = []
        self._content_blocks: dict[int, dict[str, Any]] = {}
        self._final_message_data: dict[str, Any] | None = None
        self._consumed = False

    def __aiter__(self) -> AsyncIterator[EventObject]:
        return self

    async def __anext__(self) -> EventObject:
        try:
            event = await self._iterator.__anext__()
        except StopAsyncIteration:
            await self.aclose()
            raise
        obj = _parse_anthropic_event(event)
        self._accumulate(obj)
        return obj

    def _accumulate(self, obj: EventObject) -> None:
        """Track text deltas and final message data."""
        event_type = getattr(obj, "type", None)
        if event_type == "content_block_start":
            block = getattr(obj, "content_block", None)
            if block and self._final_message_data is not None:
                self._content_blocks[getattr(obj, "index", len(self._content_blocks))] = (
                    block._data.copy()
                )
        elif event_type == "content_block_delta":
            delta = getattr(obj, "delta", None)
            if delta and getattr(delta, "type", None) == "text_delta":
                text = getattr(delta, "text", "")
                if text:
                    self._text_parts.append(text)
                    idx = getattr(obj, "index", None)
                    if idx is not None and idx in self._content_blocks:
                        block = self._content_blocks[idx]
                        block["text"] = block.get("text", "") + text
            elif delta and getattr(delta, "type", None) == "input_json_delta":
                idx = getattr(obj, "index", None)
                if idx is not None and idx in self._content_blocks:
                    block = self._content_blocks[idx]
                    partial = getattr(delta, "partial_json", "")
                    block.setdefault("_input_json_parts", []).append(partial)
        elif event_type == "message_start":
            msg = getattr(obj, "message", None)
            if msg:
                self._final_message_data = msg._data.copy()
                self._final_message_data["content"] = []
        elif event_type == "message_delta":
            delta = getattr(obj, "delta", None)
            if delta and self._final_message_data is not None:
                self._final_message_data.update(delta._data)
            usage = getattr(obj, "usage", None)
            if usage and self._final_message_data is not None:
                self._final_message_data["usage"] = usage._data

    async def _drain(self) -> None:
        """Consume remaining events if not fully iterated."""
        if not self._consumed:
            try:
                while True:
                    await self.__anext__()
            except StopAsyncIteration:
                pass
            self._consumed = True

    async def get_final_text(self) -> str:
        """Return all accumulated text from text_delta events."""
        await self._drain()
        return "".join(self._text_parts)

    def _finalize_content(self) -> None:
        """Build final message content from accumulated content blocks."""
        if self._final_message_data is None:
            return
        if self._content_blocks:
            import json as _json

            content = []
            for idx in sorted(self._content_blocks):
                block = self._content_blocks[idx]
                parts = block.pop("_input_json_parts", None)
                if parts:
                    try:
                        block["input"] = _json.loads("".join(parts))
                    except (ValueError, TypeError):
                        block["input"] = {}
                content.append(block)
            self._final_message_data["content"] = content
        elif self._text_parts:
            self._final_message_data["content"] = [
                {"type": "text", "text": "".join(self._text_parts)}
            ]

    async def get_final_message(self) -> Any:
        """Return the final message assembled from stream events."""
        await self._drain()
        if self._final_message_data is None:
            raise SmgError("No message_start event received in stream")

        self._finalize_content()

        try:
            from smg_client.types import Message

            return Message.model_validate(self._final_message_data)
        except (ImportError, ValueError):
            return EventObject(self._final_message_data)

    async def __aenter__(self) -> AnthropicAsyncStream:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._response.aclose()
