"""Pytest fixture helpers for the in-process mock MCP server.

Provides ``mock_mcp_server`` as a session-scoped fixture plus thin re-exports so
tests can ``from infra import mock_mcp_server`` without reaching into internals.

Session scope
-------------
A single ``MockMcpServer`` instance is shared across every test module that
requests the fixture. Tools are stateless (no per-test cleanup required), and
spinning up a Starlette/uvicorn app per module costs ~100 ms we don't need to
pay. If a test ever needs isolation it can request a fresh
``MockMcpServer`` directly:

    from infra import MockMcpServer

    with MockMcpServer() as srv:
        ...

``last_call_args`` reset
------------------------
Because the server is session-scoped, ``srv.call_log`` accumulates across tests.
Tests that assert on "the last call" should read ``srv.last_call_args`` (which
already reflects only the most recent call) rather than walking the full log.
Tests that want a clean slate can assign ``srv._call_log.clear()`` in a local
fixture — this is intentional backdoor access; if it turns into a common need
we'll add a public ``clear_call_log()`` method.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from .mock_mcp_server import IMAGE_GENERATION_PNG_BASE64, MockMcpServer


@pytest.fixture(scope="session")
def mock_mcp_server() -> Iterator[MockMcpServer]:
    """Session-scoped ``MockMcpServer`` bound to an auto-allocated local port."""
    server = MockMcpServer()
    server.start()
    try:
        yield server
    finally:
        server.stop()


__all__ = [
    "IMAGE_GENERATION_PNG_BASE64",
    "MockMcpServer",
    "mock_mcp_server",
]
