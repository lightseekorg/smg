"""Built-in tool tests for Response API.

Tests for built-in tool routing (web_search_preview, code_interpreter, file_search)
that are routed through MCP servers with response format transformation.

This tests the Phase 2 built-in tool support feature where OpenAI-style built-in tools
like `{type: "web_search_preview"}` are routed to MCP servers and transformed to
produce output items like `web_search_call` instead of `mcp_call`.

Prerequisites:
- Brave MCP Server running on port 8001 (set up in CI via pr-test-rust.yml)
- OPENAI_API_KEY environment variable set for cloud backend tests
"""

from __future__ import annotations

import atexit
import logging
import os
import tempfile
import time

import pytest
import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# Test Configuration
# =============================================================================

# Built-in tool definition (what client sends)
WEB_SEARCH_PREVIEW_TOOL = {
    "type": "web_search_preview",
}

# For comparison: MCP tool definition (what client would send for direct MCP)
BRAVE_MCP_TOOL = {
    "type": "mcp",
    "server_label": "brave",
    "server_description": "A Tool to do web search",
    "server_url": "http://localhost:8001/sse",
    "require_approval": "never",
}

# Test prompt that should trigger a web search
WEB_SEARCH_PROMPT = (
    "Search the web for information about the Rust programming language. "
    "Use your web search tool and provide a brief one-sentence summary."
)


def create_mcp_config_with_builtin(brave_url: str = "http://localhost:8001/sse") -> dict:
    """Create MCP config that routes web_search_preview to Brave MCP server."""
    return {
        "servers": [
            {
                "name": "brave-builtin",
                "protocol": "sse",
                "url": brave_url,
                "builtin_type": "web_search_preview",
                "builtin_tool_name": "brave_web_search",
            }
        ]
    }


# Create module-level MCP config file that persists for all tests
_MCP_CONFIG_PATH = None


def _get_mcp_config_path() -> str:
    """Get or create the MCP config file path."""
    global _MCP_CONFIG_PATH
    if _MCP_CONFIG_PATH is None:
        config = create_mcp_config_with_builtin()
        fd, path = tempfile.mkstemp(suffix=".yaml", prefix="mcp_builtin_config_")
        with os.fdopen(fd, "w") as f:
            yaml.dump(config, f)
        _MCP_CONFIG_PATH = path
        # Register cleanup
        atexit.register(lambda: os.unlink(path) if os.path.exists(path) else None)
        logger.info("Created MCP config at %s", path)
    return _MCP_CONFIG_PATH


# =============================================================================
# Cloud Backend Tests (OpenAI) - Built-in Tool Routing
# =============================================================================


# Gateway extra_args including MCP config path
def _get_builtin_gateway_args() -> list[str]:
    """Get gateway args with MCP config for built-in tool routing."""
    return ["--mcp-config-path", _get_mcp_config_path()]


@pytest.mark.gateway(extra_args=_get_builtin_gateway_args())
@pytest.mark.parametrize("setup_backend", ["openai"], indirect=True)
class TestBuiltinToolsCloud:
    """Built-in tool tests against cloud APIs with MCP routing.

    These tests verify that when a client sends `{type: "web_search_preview"}`,
    the router correctly routes it to the configured MCP server and transforms
    the response to produce `web_search_call` output items.
    """

    def test_web_search_preview_produces_web_search_call(self, setup_backend):
        """Test that web_search_preview tool produces web_search_call output.

        This is the core test for Phase 2 built-in tool support:
        1. Client sends {type: "web_search_preview"} tool
        2. Router routes to Brave MCP server
        3. Response contains web_search_call output items (NOT mcp_call)
        """
        _, model, client, gateway = setup_backend

        time.sleep(2)  # Avoid rate limiting

        resp = client.responses.create(
            model=model,
            input=WEB_SEARCH_PROMPT,
            tools=[WEB_SEARCH_PREVIEW_TOOL],
            stream=False,
        )

        assert resp.error is None, f"Response error: {resp.error}"
        assert resp.id is not None
        assert resp.status == "completed"
        assert resp.output is not None

        output_types = [item.type for item in resp.output]
        logger.info("Output types: %s", output_types)

        # CRITICAL: Should have web_search_call, NOT mcp_call
        assert "web_search_call" in output_types, (
            f"Expected web_search_call in output types, got: {output_types}. "
            "Built-in tool routing may not be working correctly."
        )
        assert "mcp_call" not in output_types, (
            f"Should not have mcp_call when using web_search_preview, got: {output_types}. "
            "Response format transformation may not be working."
        )

        # Verify web_search_call structure
        web_search_calls = [item for item in resp.output if item.type == "web_search_call"]
        assert len(web_search_calls) > 0

        for call in web_search_calls:
            assert call.id is not None
            assert call.status == "completed"

    def test_web_search_preview_streaming(self, setup_backend):
        """Test web_search_preview tool with streaming produces correct events."""
        _, model, client, gateway = setup_backend

        time.sleep(2)  # Avoid rate limiting

        resp = client.responses.create(
            model=model,
            input=WEB_SEARCH_PROMPT,
            tools=[WEB_SEARCH_PREVIEW_TOOL],
            stream=True,
        )

        events = list(resp)
        assert len(events) > 0

        event_types = [event.type for event in events]
        logger.info("Event types: %s", event_types)

        # Check for correct streaming events
        assert "response.created" in event_types
        assert "response.completed" in event_types

        # CRITICAL: Should have web_search_call events, NOT mcp_call events
        has_web_search_events = any("web_search_call" in et for et in event_types)
        has_mcp_events = any("mcp_call" in et for et in event_types)

        assert has_web_search_events, (
            f"Expected web_search_call events in streaming, got: {event_types}"
        )
        assert not has_mcp_events, (
            f"Should not have mcp_call events when using web_search_preview: {event_types}"
        )

        # Verify final response structure
        completed_events = [e for e in events if e.type == "response.completed"]
        assert len(completed_events) == 1

        final_response = completed_events[0].response
        final_output_types = [item.type for item in final_response.output]

        assert "web_search_call" in final_output_types
        assert "mcp_call" not in final_output_types

    def test_mixed_builtin_and_function_tools(self, setup_backend):
        """Test mixing web_search_preview with regular function tools."""
        _, model, client, gateway = setup_backend

        # Add a regular function tool alongside web_search_preview
        get_weather_function = {
            "type": "function",
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name, e.g., San Francisco",
                    }
                },
                "required": ["location"],
            },
        }

        resp = client.responses.create(
            model=model,
            input="Search the web for Rust programming and also tell me the weather in Seattle.",
            tools=[WEB_SEARCH_PREVIEW_TOOL, get_weather_function],
            stream=False,
        )

        assert resp.error is None
        assert resp.id is not None
        assert resp.output is not None

        output_types = [item.type for item in resp.output]
        logger.info("Mixed tools output types: %s", output_types)

        # Should have either web_search_call or function_call (or both)
        has_tool_call = "web_search_call" in output_types or "function_call" in output_types
        assert has_tool_call, f"Expected tool calls in output, got: {output_types}"

        # Should NOT have mcp_call
        assert "mcp_call" not in output_types

    def test_response_tools_field_shows_original_type(self, setup_backend):
        """Test that the response tools field shows web_search_preview, not mcp."""
        _, model, client, gateway = setup_backend

        time.sleep(2)

        resp = client.responses.create(
            model=model,
            input=WEB_SEARCH_PROMPT,
            tools=[WEB_SEARCH_PREVIEW_TOOL],
            stream=False,
        )

        assert resp.error is None

        # Check that tools in response show the original type
        if hasattr(resp, "tools") and resp.tools:
            for tool in resp.tools:
                tool_dict = tool if isinstance(tool, dict) else tool.model_dump()
                # Should be web_search_preview, not mcp
                assert tool_dict.get("type") == "web_search_preview", (
                    f"Response tools should show original type, got: {tool_dict}"
                )


# =============================================================================
# Local Backend Tests (gRPC with Harmony model) - Built-in Tool Routing
# =============================================================================


@pytest.mark.e2e
@pytest.mark.model("gpt-oss")
@pytest.mark.gateway(
    extra_args=[
        "--reasoning-parser=gpt-oss",
        "--history-backend",
        "memory",
    ]
)
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestBuiltinToolsLocal:
    """Built-in tool tests against local gRPC backend.

    Note: These tests require the gateway to be configured with the MCP config
    that routes web_search_preview to the Brave MCP server. The extra_args
    marker doesn't support dynamic values, so these tests use the static
    config defined in the marker.

    For full built-in tool testing, run with the cloud backend tests.
    """

    def test_web_search_preview_basic(self, setup_backend):
        """Test that web_search_preview tool is accepted by local backend."""
        _, model, client, gateway = setup_backend

        time.sleep(2)

        # This test verifies the tool is accepted; full routing requires MCP config
        resp = client.responses.create(
            model=model,
            input=WEB_SEARCH_PROMPT,
            tools=[WEB_SEARCH_PREVIEW_TOOL],
            stream=False,
        )

        # Should not error - tool should be recognized
        assert resp.error is None or "tool" not in str(resp.error).lower()
        assert resp.id is not None


# =============================================================================
# Comparison Tests - Verify Difference Between MCP and Built-in
# =============================================================================


@pytest.mark.parametrize("setup_backend", ["openai"], indirect=True)
class TestBuiltinVsMcpComparison:
    """Tests comparing built-in tool behavior vs direct MCP tool behavior.

    These tests verify that:
    1. Direct MCP tools produce mcp_call output items
    2. Built-in tools (when properly configured) produce their specific output types
    """

    def test_mcp_tool_produces_mcp_call(self, setup_backend):
        """Verify that direct MCP tool produces mcp_call (baseline)."""
        _, model, client, gateway = setup_backend

        time.sleep(2)

        resp = client.responses.create(
            model=model,
            input="Search the web for Python programming language information.",
            tools=[BRAVE_MCP_TOOL],
            stream=False,
            reasoning={"effort": "low"},
        )

        assert resp.error is None

        output_types = [item.type for item in resp.output]
        logger.info("MCP tool output types: %s", output_types)

        # Direct MCP should produce mcp_call
        assert "mcp_call" in output_types, (
            f"Direct MCP tool should produce mcp_call, got: {output_types}"
        )
        # And should NOT produce web_search_call
        assert "web_search_call" not in output_types
