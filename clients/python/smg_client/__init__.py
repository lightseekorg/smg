"""SMG Python client SDK.

Usage::

    from smg_client import SmgClient, AsyncSmgClient

    # Sync
    client = SmgClient(base_url="http://localhost:30000")
    resp = client.chat.completions.create(
        model="llama-3.1-8b",
        messages=[{"role": "user", "content": "Hello"}],
    )

    # Async
    async with AsyncSmgClient(base_url="http://localhost:30000") as client:
        resp = await client.chat.completions.create(
            model="llama-3.1-8b",
            messages=[{"role": "user", "content": "Hello"}],
        )
"""

from smg_client._client import AsyncSmgClient, SmgClient
from smg_client._config import ClientConfig

# ConnectionError and TimeoutError intentionally shadow Python builtins
# (same pattern as the OpenAI Python client SDK).
from smg_client._errors import (
    ApiError,
    AuthenticationError,
    BadRequestError,
    ConnectionError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    ServiceUnavailableError,
    SmgError,
    TimeoutError,
)

__version__ = "0.1.0"

__all__ = [
    "AsyncSmgClient",
    "SmgClient",
    "ClientConfig",
    # Errors
    "ApiError",
    "AuthenticationError",
    "BadRequestError",
    "ConnectionError",
    "InternalServerError",
    "NotFoundError",
    "PermissionDeniedError",
    "RateLimitError",
    "ServiceUnavailableError",
    "SmgError",
    "TimeoutError",
]
