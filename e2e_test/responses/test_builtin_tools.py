"""Built-in tool tests for Response API.

Tests for built-in tool routing (web_search_preview, code_interpreter, file_search)
that are routed through MCP servers with response format transformation.

Prerequisites:
- Brave MCP Server running on port 8001 (set up in CI via pr-test-rust.yml)
- OPENAI_API_KEY environment variable set for cloud backend tests
"""

from __future__ import annotations

import logging
import os
import socket
import tempfile
import time

import pytest
import yaml

logger = logging.getLogger(__name__)

BRAVE_MCP_PORT = 8001
BRAVE_MCP_URL = f"http://localhost:{BRAVE_MCP_PORT}/sse"

WEB_SEARCH_PREVIEW_TOOL = {"type": "web_search_preview"}

BRAVE_MCP_TOOL = {
    "type": "mcp",
    "server_label": "brave",
    "server_url": BRAVE_MCP_URL,
    "require_approval": "never",
}

WEB_SEARCH_PROMPT = (
    "Search the web for information about the Rust programming language. "
    "Use your web search tool and provide a brief summary."
)


def is_brave_server_available() -> bool:
    """Check if Brave MCP server is running on expected port."""
    try:
        with socket.create_connection(("localhost", BRAVE_MCP_PORT), timeout=1):
            return True
    except (OSError, socket.timeout):
        return False


def create_mcp_config() -> dict:
    """Create MCP config that routes web_search_preview to Brave MCP server."""
    return {
        "servers": [
            {
                "name": "brave-builtin",
                "protocol": "sse",
                "url": BRAVE_MCP_URL,
                "builtin_type": "web_search_preview",
                "builtin_tool_name": "brave_web_search",
            }
        ]
    }


@pytest.fixture(scope="module")
def mcp_config_file():
    """Create temporary MCP config file for tests."""
    config = create_mcp_config()
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="mcp_builtin_")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.dump(config, f)
        logger.info("Created MCP config at %s", path)
        yield path
    finally:
        if os.path.exists(path):
            os.unlink(path)


@pytest.fixture(scope="module")
def require_brave_server():
    """Skip tests if Brave MCP server is not available."""
    if not is_brave_server_available():
        pytest.skip(
            f"Brave MCP server not available on port {BRAVE_MCP_PORT}. "
            "Run: docker run -d -p 8001:8080 -e BRAVE_API_KEY=<key> "
            "shoofio/brave-search-mcp-sse:1.0.10"
        )


# Note: These tests require manual gateway configuration with MCP config.
# In CI, the gateway is started with --mcp-config-path pointing to a config
# that has builtin_type: web_search_preview configured.
#
# The marker approach doesn't work well for dynamic config paths, so these
# tests serve as documentation and can be run manually with proper setup.


@pytest.mark.parametrize("setup_backend", ["openai"], indirect=True)
class TestBuiltinVsMcpComparison:
    """Compare built-in tool behavior vs direct MCP tool behavior.

    These tests verify baseline MCP behavior without requiring builtin routing config.
    """

    def test_mcp_tool_produces_mcp_call(self, setup_backend):
        """Verify that direct MCP tool produces mcp_call output."""
        _, model, client, gateway = setup_backend

        time.sleep(2)

        resp = client.responses.create(
            model=model,
            input="Search the web for Python programming language.",
            tools=[BRAVE_MCP_TOOL],
            stream=False,
        )

        assert resp.error is None, f"Response error: {resp.error}"
        assert resp.output is not None

        output_types = [item.type for item in resp.output]
        logger.info("MCP tool output types: %s", output_types)

        # Direct MCP should produce mcp_call
        assert "mcp_call" in output_types, (
            f"Direct MCP tool should produce mcp_call, got: {output_types}"
        )


@pytest.mark.parametrize("setup_backend", ["openai"], indirect=True)
class TestBuiltinToolsCloudBackend:
    """Built-in tool tests against cloud backend (OpenAI).

    These tests verify that built-in tool types are accepted by the API.
    Full routing tests require MCP config with builtin_type configured.
    """

    def test_web_search_preview_accepted(self, setup_backend):
        """Test that web_search_preview tool type is accepted."""
        _, model, client, gateway = setup_backend

        time.sleep(2)

        resp = client.responses.create(
            model=model,
            input=WEB_SEARCH_PROMPT,
            tools=[WEB_SEARCH_PREVIEW_TOOL],
            stream=False,
        )

        assert resp.id is not None
        assert resp.status in ("completed", "incomplete")

    def test_mixed_builtin_and_function_tools(self, setup_backend):
        """Test mixing web_search_preview with function tools."""
        _, model, client, gateway = setup_backend

        time.sleep(2)

        get_weather_function = {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"],
            },
        }

        resp = client.responses.create(
            model=model,
            input="What's the weather in Seattle?",
            tools=[WEB_SEARCH_PREVIEW_TOOL, get_weather_function],
            stream=False,
        )

        assert resp.error is None
        assert resp.id is not None


@pytest.mark.e2e
@pytest.mark.model("gpt-oss")
@pytest.mark.gateway(
    extra_args=["--reasoning-parser=gpt-oss", "--history-backend", "memory"]
)
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestBuiltinToolsLocalBackend:
    """Built-in tool tests against local gRPC backend.

    These tests verify built-in tool handling with local models.
    """

    def test_web_search_preview_accepted(self, setup_backend):
        """Test that web_search_preview tool type is accepted by local backend."""
        _, model, client, gateway = setup_backend

        time.sleep(1)

        resp = client.responses.create(
            model=model,
            input=WEB_SEARCH_PROMPT,
            tools=[WEB_SEARCH_PREVIEW_TOOL],
            stream=False,
        )

        assert resp.id is not None

    def test_mixed_builtin_and_function_tools(self, setup_backend):
        """Test mixing web_search_preview with function tools on local backend."""
        _, model, client, gateway = setup_backend

        time.sleep(1)

        get_weather_function = {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"],
            },
        }

        resp = client.responses.create(
            model=model,
            input="What's the weather in Seattle?",
            tools=[WEB_SEARCH_PREVIEW_TOOL, get_weather_function],
            stream=False,
        )

        assert resp.id is not None


# =============================================================================
# Full Integration Tests (require Brave MCP server + proper gateway config)
# =============================================================================
# These tests are designed for CI where:
# 1. Brave MCP server runs on port 8001
# 2. Gateway is configured with builtin_type: web_search_preview
#
# To run locally:
# 1. Start Brave MCP: docker run -d -p 8001:8080 -e BRAVE_API_KEY=<key> shoofio/brave-search-mcp-sse:1.0.10
# 2. Create mcp.yaml with builtin_type config
# 3. Start gateway with --mcp-config-path mcp.yaml
# 4. Run tests with proper backend


class TestBuiltinToolRouting:
    """Full integration tests for built-in tool routing.

    These tests verify the complete flow:
    1. Client sends {type: "web_search_preview"}
    2. Gateway routes to configured MCP server
    3. Response contains web_search_call (not mcp_call)

    Requires:
    - Brave MCP server on port 8001
    - Gateway with builtin_type config
    """

    @pytest.fixture(autouse=True)
    def check_prerequisites(self, require_brave_server):
        """Ensure Brave server is available for these tests."""
        pass

    @pytest.mark.skip(reason="Requires gateway with builtin_type MCP config")
    def test_web_search_preview_produces_web_search_call(self):
        """Test that web_search_preview produces web_search_call output.

        This is the core test for Phase 2 built-in tool support.
        Skip by default - enable when running with proper gateway config.
        """
        # This test would verify:
        # 1. Send request with {type: "web_search_preview"}
        # 2. Verify response has web_search_call in output types
        # 3. Verify response does NOT have mcp_call
        pass

    @pytest.mark.skip(reason="Requires gateway with builtin_type MCP config")
    def test_response_tools_shows_original_type(self):
        """Test that response tools field shows web_search_preview, not mcp."""
        # This test would verify the tools field in response
        # mirrors the original request tool type
        pass
