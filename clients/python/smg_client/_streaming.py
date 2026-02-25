"""Typed stream wrappers for different API streaming protocols."""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

from smg_client._sse import iter_sse_async, iter_sse_sync

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    import httpx

T = TypeVar("T")


class SyncStream(Generic[T]):
    """Synchronous typed stream over SSE events.

    Usage::

        with client.chat.completions.create_stream(req) as stream:
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
        event = next(self._iterator)
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

        async with client.chat.completions.create_stream(req) as stream:
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
        event = await self._iterator.__anext__()
        return self._model_cls.model_validate_json(event.data)

    async def __aenter__(self) -> AsyncStream[T]:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._response.aclose()


class AnthropicSyncStream:
    """Synchronous stream for Anthropic Messages API events.

    Anthropic uses `event: type\\ndata: {...}` format where the event type
    is stored in the SSE event field, not in the JSON data.

    Usage::

        with client.messages.create_stream(req) as stream:
            for event in stream:
                if event.type == "content_block_delta":
                    print(event.delta.text)
    """

    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self._iterator = iter_sse_sync(response)

    def __iter__(self) -> Iterator[dict]:
        return self

    def __next__(self) -> dict:
        event = next(self._iterator)
        data = event.json()
        # Anthropic events have `type` in both the SSE event field and JSON body.
        # The SSE event field is authoritative; ensure it's set in the parsed dict.
        if event.event and "type" not in data:
            data["type"] = event.event
        return data

    def __enter__(self) -> AnthropicSyncStream:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self._response.close()


class AnthropicAsyncStream:
    """Async stream for Anthropic Messages API events."""

    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self._iterator = iter_sse_async(response)

    def __aiter__(self) -> AsyncIterator[dict]:
        return self

    async def __anext__(self) -> dict:
        event = await self._iterator.__anext__()
        data = event.json()
        if event.event and "type" not in data:
            data["type"] = event.event
        return data

    async def __aenter__(self) -> AnthropicAsyncStream:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._response.aclose()
