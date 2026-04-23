"""Shared fixtures for the Responses API test suite.

Hosts the ``gateway_with_mock_mcp`` fixture used by the image_generation tests
(and, once the TODO stubs in ``infra/mock_mcp_server.py`` are uncommented, by
the upcoming web_search / file_search / code_interpreter tests). The fixture
writes a minimal MCP config file pointing at the in-process mock server and
launches an OpenAI cloud gateway configured to route ``image_generation``
built-in tool requests through it.

Design notes
------------
* The gateway CLI exposes ``--mcp-config-path`` (see
  ``bindings/python/src/smg/router_args.py``) which takes a path to a YAML
  config deserialized into ``crates/mcp/src/core/config.rs::McpConfig``. That
  config contains a list of servers, each with an optional ``builtin_type`` —
  setting ``builtin_type: image_generation`` on a server makes the gateway
  route every ``{"type": "image_generation"}`` tool request through it
  instead of passing the tool to the upstream model.
* We connect with ``protocol: streamable`` which matches the ``FastMCP`` app
  produced by ``streamable_http_app()``. ``sse`` would also work but would
  require an extra event-stream hop per call.
* We use the OpenAI cloud backend (``setup_backend == "openai"``) because
  local-backend routing for ``image_generation`` requires real GPU workers
  and a model that actually emits ``image_generation`` tool calls.
  ``skip_for_runtime`` on ``sglang`` and ``vllm`` keeps the gRPC lanes off
  until R6.3/R6.4 stabilise in CI.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Iterator

import openai
import pytest
import yaml

# Fixture re-export: pytest discovers fixtures by walking the names defined
# in the conftest module. Importing ``mock_mcp_server`` here makes it
# visible to every test collected from this conftest without the
# PytestAssertRewriteWarning we'd get from registering ``infra.mock_mcp``
# as a ``pytest_plugins`` entry after it's already been imported via
# ``infra``. Ruff flags this as F401 (unused import) and then flags the
# fixture argument sites as F811 (redefinition); both are ignored
# intentionally — pytest needs the name to be bound at module scope.
from infra import MockMcpServer, launch_cloud_gateway
from infra.mock_mcp import mock_mcp_server as mock_mcp_server  # noqa: F401

logger = logging.getLogger(__name__)


# =============================================================================
# Mock MCP config
# =============================================================================


def _image_generation_mcp_config(mock_mcp_url: str) -> dict:
    """Build an MCP config that routes image_generation through the mock server.

    The shape mirrors ``crates/mcp/src/core/config.rs::McpConfig`` (YAML
    deserialization). ``builtin_type: image_generation`` is the knob that
    tells the gateway to route ``{"type": "image_generation"}`` tool requests
    through this server; ``builtin_tool_name`` names the tool on our server
    that implements the semantics (``image_generation`` for the mock).
    ``response_format: image_generation_call`` asks the transformer to shape
    the MCP tool result into a Responses API ``image_generation_call`` output
    item.
    """
    return {
        "servers": [
            {
                "name": "mock-image-gen",
                "protocol": "streamable",
                "url": mock_mcp_url,
                "builtin_type": "image_generation",
                "builtin_tool_name": "image_generation",
                "tools": {
                    "image_generation": {
                        "response_format": "image_generation_call",
                    }
                },
            }
        ]
    }


@pytest.fixture(scope="session")
def mock_mcp_config_file(mock_mcp_server: MockMcpServer) -> Iterator[str]:  # noqa: F811
    """Write the MCP config YAML to a tempfile and yield its path.

    Session-scoped so all tests in the module share a single config; the
    gateway re-reads the file at startup so sharing is safe.

    Note: ``# noqa: F811`` silences ruff's "redefinition of unused name"
    warning. The ``mock_mcp_server`` import above exists solely so pytest
    discovers the fixture in this module; using the name as a parameter
    here is exactly the pattern that triggers F811.
    """
    config = _image_generation_mcp_config(mock_mcp_server.url)
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="mock_mcp_image_gen_")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.dump(config, f)
        logger.info("MCP config for mock image_generation at %s", path)
        yield path
    finally:
        if os.path.exists(path):
            os.unlink(path)


# =============================================================================
# Gateway fixtures
# =============================================================================


@pytest.fixture(scope="class")
def gateway_with_mock_mcp(
    mock_mcp_server: MockMcpServer,  # noqa: F811
    mock_mcp_config_file: str,
) -> Iterator[tuple]:
    """Launch an OpenAI cloud gateway wired to the mock MCP server.

    Returns ``(gateway, client, mock_mcp_server)``. The gateway is the
    fully-booted ``Gateway`` object, the client is a vanilla ``openai.OpenAI``
    pointed at it with the real ``OPENAI_API_KEY`` (so that function-calling
    still works upstream), and the mock is passed through so tests can assert
    on ``last_call_args`` / ``call_log`` without re-requesting the fixture.

    The gateway is class-scoped to mirror ``setup_backend``; image_generation
    tests run quickly but bootstrapping the cloud gateway (~1-2 s) is the
    dominant cost, so a class-scoped reuse keeps the suite snappy.
    """
    api_key_env = "OPENAI_API_KEY"
    if not os.environ.get(api_key_env):
        pytest.skip(f"{api_key_env} not set — image_generation e2e needs OpenAI")

    logger.info(
        "Launching OpenAI cloud gateway with mock MCP config (url=%s, config=%s)",
        mock_mcp_server.url,
        mock_mcp_config_file,
    )
    gateway = launch_cloud_gateway(
        "openai",
        history_backend="memory",
        extra_args=["--mcp-config-path", mock_mcp_config_file],
    )

    # Construct the client inside a try/except so a failure here does not
    # leak the already-launched gateway. The outer ``finally`` handles the
    # happy-path teardown after the test yields.
    try:
        client = openai.OpenAI(
            base_url=f"{gateway.base_url}/v1",
            api_key=os.environ[api_key_env],
        )
    except Exception:
        gateway.shutdown()
        raise

    try:
        yield gateway, client, mock_mcp_server
    finally:
        gateway.shutdown()


# =============================================================================
# Tool argument helpers
# =============================================================================


@pytest.fixture
def image_gen_tool_args() -> dict:
    """Canonical ``image_generation`` tool payload used across the suite.

    Centralised here so tests that want to override (e.g., size) can start
    from a shared baseline. The keys mirror
    ``crates/protocols/src/responses.rs::ImageGenerationTool``.
    """
    return {
        "type": "image_generation",
        "size": "1024x1024",
        "quality": "standard",
    }
