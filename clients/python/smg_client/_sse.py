"""Server-Sent Events (SSE) parser.

Handles three SSE protocol variants:
1. OpenAI: `data: {...}\\n\\n` lines, terminated by `data: [DONE]\\n\\n`
2. Anthropic: `event: type\\ndata: {...}\\n\\n` pairs
3. Responses API: `event: type\\ndata: {...}\\n\\n` plus `data: [DONE]`
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    import httpx


@dataclass
class SseEvent:
    """A parsed SSE event."""

    data: str
    event: str | None = None

    def json(self) -> dict:
        """Parse the data field as JSON."""
        return json.loads(self.data)


def iter_sse_sync(response: httpx.Response) -> Iterator[SseEvent]:
    """Parse SSE events from a synchronous streaming response."""
    event_type: str | None = None

    for line in response.iter_lines():
        if not line:
            # Empty line = event boundary, but we yield on data lines
            event_type = None
            continue

        if line.startswith("event:"):
            event_type = line[len("event:") :].strip()
            continue

        if line.startswith("data:"):
            data = line[len("data:") :].strip()

            if data == "[DONE]":
                return

            yield SseEvent(data=data, event=event_type)
            event_type = None
            continue

        # Lines starting with ":" are comments (keep-alive pings)


async def iter_sse_async(response: httpx.Response) -> AsyncIterator[SseEvent]:
    """Parse SSE events from an async streaming response."""
    event_type: str | None = None

    async for line in response.aiter_lines():
        if not line:
            event_type = None
            continue

        if line.startswith("event:"):
            event_type = line[len("event:") :].strip()
            continue

        if line.startswith("data:"):
            data = line[len("data:") :].strip()

            if data == "[DONE]":
                return

            yield SseEvent(data=data, event=event_type)
            event_type = None
            continue
