"""Main client classes."""

from __future__ import annotations

from smg_client._config import ClientConfig
from smg_client._transport import AsyncTransport, SyncTransport
from smg_client.api.chat import AsyncChat, SyncChat
from smg_client.api.classify import AsyncClassify, SyncClassify
from smg_client.api.completions import (
    AsyncCompletions as AsyncLegacyCompletions,
)
from smg_client.api.completions import (
    SyncCompletions as SyncLegacyCompletions,
)
from smg_client.api.embeddings import AsyncEmbeddings, SyncEmbeddings
from smg_client.api.messages import AsyncMessages, SyncMessages
from smg_client.api.models import AsyncModels, SyncModels
from smg_client.api.parser import AsyncParser, SyncParser
from smg_client.api.rerank import AsyncRerank, SyncRerank
from smg_client.api.responses import AsyncResponses, SyncResponses
from smg_client.api.workers import AsyncWorkers, SyncWorkers


class SmgClient:
    """Synchronous SMG client.

    Usage::

        client = SmgClient(base_url="http://localhost:30000")

        # Chat completions
        resp = client.chat.completions.create(
            model="llama-3.1-8b",
            messages=[{"role": "user", "content": "Hello"}],
        )
        print(resp.choices[0].message.content)

        # Streaming
        with client.chat.completions.create_stream(
            model="llama-3.1-8b",
            messages=[{"role": "user", "content": "Hello"}],
        ) as stream:
            for chunk in stream:
                print(chunk.choices[0].delta.content, end="")

        # Anthropic Messages
        msg = client.messages.create(
            model="claude-3-5-sonnet",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )
        print(msg.content[0].text)

        # Embeddings
        emb = client.embeddings.create(model="bge-large", input="Hello world")

        # Models
        models = client.models.list()

        client.close()
    """

    def __init__(
        self,
        base_url: str = "http://localhost:30000",
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        config = ClientConfig(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers or {},
        )
        self._transport = SyncTransport(config)

        self.chat = SyncChat(self._transport)
        self.classify = SyncClassify(self._transport)
        self.completions = SyncLegacyCompletions(self._transport)
        self.embeddings = SyncEmbeddings(self._transport)
        self.messages = SyncMessages(self._transport)
        self.models = SyncModels(self._transport)
        self.parser = SyncParser(self._transport)
        self.rerank = SyncRerank(self._transport)
        self.responses = SyncResponses(self._transport)
        self.workers = SyncWorkers(self._transport)

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._transport.close()

    def __enter__(self) -> SmgClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class AsyncSmgClient:
    """Async SMG client.

    Usage::

        async with AsyncSmgClient(base_url="http://localhost:30000") as client:
            resp = await client.chat.completions.create(
                model="llama-3.1-8b",
                messages=[{"role": "user", "content": "Hello"}],
            )

            stream = await client.chat.completions.create_stream(
                model="llama-3.1-8b",
                messages=[{"role": "user", "content": "Hello"}],
            )
            async with stream:
                async for chunk in stream:
                    print(chunk.choices[0].delta.content, end="")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:30000",
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        config = ClientConfig(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers or {},
        )
        self._transport = AsyncTransport(config)

        self.chat = AsyncChat(self._transport)
        self.classify = AsyncClassify(self._transport)
        self.completions = AsyncLegacyCompletions(self._transport)
        self.embeddings = AsyncEmbeddings(self._transport)
        self.messages = AsyncMessages(self._transport)
        self.models = AsyncModels(self._transport)
        self.parser = AsyncParser(self._transport)
        self.rerank = AsyncRerank(self._transport)
        self.responses = AsyncResponses(self._transport)
        self.workers = AsyncWorkers(self._transport)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._transport.close()

    async def __aenter__(self) -> AsyncSmgClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()
