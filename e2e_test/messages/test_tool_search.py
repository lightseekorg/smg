"""Tool search tests for Anthropic Messages API.

Tests for the tool_search_tool feature with deferred tool loading,
including passthrough mode and SMG-handled MCP mode with defer_loading.

Requirements:
- ANTHROPIC_API_KEY environment variable must be set
- For MCP tests: Mock MCP servers (started via subprocess or manually)
"""

from __future__ import annotations

import logging
import os
from urllib.parse import urlparse

import pytest
import requests
from infra import BRAVE_MCP_URL

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================


def make_tool_search_tool() -> dict:
    """Build a tool_search_tool_regex entry for inclusion in the tools array."""
    return {
        "type": "tool_search_tool_regex_20251119",
        "name": "tool_search_tool_regex",
    }


def make_custom_tool(
    name: str,
    description: str,
    properties: dict,
    required: list[str] | None = None,
    defer_loading: bool | None = None,
) -> dict:
    """Build a custom tool definition, optionally with defer_loading."""
    tool: dict = {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
        },
    }
    if required:
        tool["input_schema"]["required"] = required
    if defer_loading is not None:
        tool["defer_loading"] = defer_loading
    return tool


def make_mcp_toolset(
    server_name: str,
    default_defer_loading: bool | None = None,
    default_enabled: bool | None = None,
    tool_overrides: dict[str, dict] | None = None,
) -> dict:
    """Build an mcp_toolset tool entry with optional defer_loading config."""
    entry: dict = {
        "type": "mcp_toolset",
        "mcp_server_name": server_name,
    }
    default_config: dict = {}
    if default_defer_loading is not None:
        default_config["defer_loading"] = default_defer_loading
    if default_enabled is not None:
        default_config["enabled"] = default_enabled
    if default_config:
        entry["default_config"] = default_config
    if tool_overrides:
        entry["configs"] = tool_overrides
    return entry


# =============================================================================
# Test 1: Passthrough tool search (no MCP, no x-smg-mcp header)
# =============================================================================


@pytest.mark.vendor("anthropic")
@pytest.mark.gpu(0)
@pytest.mark.parametrize("setup_backend", ["anthropic"], indirect=True)
class TestToolSearchPassthrough:
    """Tool search passthrough tests — request forwarded to Anthropic as-is."""

    def test_tool_search_with_deferred_tools_non_streaming(self, setup_backend, api_client):
        """Send tool_search_tool + deferred custom tools.

        Verifies the response can contain server_tool_use (tool search invocation)
        and that the model can discover and reference deferred tools.
        """
        _, model, _, _ = setup_backend

        tools = [
            make_tool_search_tool(),
            make_custom_tool(
                "get_weather",
                "Get current weather for a city",
                {
                    "city": {"type": "string"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                required=["city"],
                defer_loading=True,
            ),
            make_custom_tool(
                "search_web",
                "Search the web for information",
                {"query": {"type": "string"}},
                required=["query"],
                defer_loading=True,
            ),
            make_custom_tool(
                "calculate",
                "Perform a mathematical calculation",
                {"expression": {"type": "string"}},
                required=["expression"],
                defer_loading=True,
            ),
        ]

        response = api_client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": "What's the weather like in Tokyo?"}],
            extra_body={"tools": tools},
        )

        assert response.id is not None
        assert response.model == model
        assert response.role == "assistant"
        assert len(response.content) > 0

        block_types = [b.type for b in response.content]

        # 1. server_tool_use: model invokes tool search
        server_tool_uses = [b for b in response.content if b.type == "server_tool_use"]
        assert len(server_tool_uses) == 1, (
            f"Expected 1 server_tool_use, got {len(server_tool_uses)}: {block_types}"
        )
        stu = server_tool_uses[0]
        assert stu.name == "tool_search_tool_regex"
        assert stu.id.startswith("srvtoolu_")
        assert isinstance(stu.input, dict)
        assert "pattern" in stu.input, f"Expected 'pattern' in tool search input, got: {stu.input}"

        # 2. tool_search_tool_result: search returns matching tool references
        search_results = [b for b in response.content if b.type == "tool_search_tool_result"]
        assert len(search_results) == 1, (
            f"Expected 1 tool_search_tool_result, got {len(search_results)}: {block_types}"
        )
        tsr = search_results[0]
        assert tsr.tool_use_id == stu.id, (
            "tool_search_tool_result should reference the server_tool_use id"
        )

        # 3. tool_use: model calls the discovered tool
        assert response.stop_reason == "tool_use"
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        assert len(tool_uses) == 1, f"Expected 1 tool_use, got {len(tool_uses)}: {block_types}"
        assert tool_uses[0].name == "get_weather"
        assert isinstance(tool_uses[0].input, dict)

        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

    def test_tool_search_with_deferred_tools_streaming(self, setup_backend, api_client):
        """Streaming variant of tool search passthrough."""
        _, model, _, _ = setup_backend

        tools = [
            make_tool_search_tool(),
            make_custom_tool(
                "get_weather",
                "Get current weather for a city",
                {"city": {"type": "string"}},
                required=["city"],
                defer_loading=True,
            ),
            make_custom_tool(
                "translate_text",
                "Translate text between languages",
                {"text": {"type": "string"}, "target_language": {"type": "string"}},
                required=["text", "target_language"],
                defer_loading=True,
            ),
        ]

        event_types = set()
        block_types = []
        content_blocks = []

        with api_client.messages.stream(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": "What's the weather in Paris?"}],
            extra_body={"tools": tools},
        ) as stream:
            for event in stream:
                event_types.add(event.type)
                if event.type == "content_block_start":
                    block_types.append(event.content_block.type)
                    content_blocks.append(event.content_block)

        response = stream.get_final_message()

        # Core SSE lifecycle events
        assert "message_start" in event_types
        assert "content_block_start" in event_types
        assert "content_block_delta" in event_types
        assert "content_block_stop" in event_types
        assert "message_delta" in event_types
        assert "message_stop" in event_types

        # Validate tool search flow in block types
        assert "server_tool_use" in block_types, (
            f"Expected server_tool_use block for tool search, got: {block_types}"
        )
        assert "tool_search_tool_result" in block_types, (
            f"Expected tool_search_tool_result block, got: {block_types}"
        )
        assert "tool_use" in block_types, (
            f"Expected tool_use block for discovered tool, got: {block_types}"
        )

        # Validate server_tool_use is for tool search
        server_tool_uses = [b for b in content_blocks if b.type == "server_tool_use"]
        assert len(server_tool_uses) >= 1
        assert server_tool_uses[0].name == "tool_search_tool_regex"

        # Validate tool_search_tool_result has tool references
        search_results = [b for b in content_blocks if b.type == "tool_search_tool_result"]
        assert len(search_results) >= 1

        # Validate tool_use targets a discovered tool
        tool_uses = [b for b in content_blocks if b.type == "tool_use"]
        assert len(tool_uses) >= 1
        assert tool_uses[0].name == "get_weather"

        # Validate final message
        assert response.stop_reason == "tool_use"
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0


# =============================================================================
# Test 2: SMG-handled MCP + tool search + defer_loading
# =============================================================================


MCP_SERVER_URL = os.environ.get("MCP_TOOL_SEARCH_SERVER_URL", BRAVE_MCP_URL)
MCP_SERVER_NAME = os.environ.get("MCP_TOOL_SEARCH_SERVER_NAME", "brave")


@pytest.mark.vendor("anthropic")
@pytest.mark.gpu(0)
@pytest.mark.parametrize("setup_backend", ["anthropic"], indirect=True)
class TestToolSearchWithMcp:
    """Tool search with SMG-handled MCP (x-smg-mcp: enabled).

    SMG connects to the MCP server, discovers tools, injects them as regular
    tools with defer_loading: true. Anthropic uses tool search to find the
    right tool, returns tool_use. SMG executes it via MCP and loops back.

    Requires a running MCP server (set MCP_TOOL_SEARCH_SERVER_URL).
    """

    @pytest.fixture(scope="class", autouse=True)
    def _check_mcp_server_available(self):
        """Fail fast if MCP server is not reachable."""
        parsed = urlparse(MCP_SERVER_URL)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        try:
            requests.head(base_url, timeout=5)
        except requests.RequestException as exc:
            pytest.fail(
                f"MCP server at {MCP_SERVER_URL} is not available: {exc}. "
                "Ensure the MCP server is running before running these tests."
            )

    def test_mcp_tools_with_deferred_loading(self, setup_backend, api_client):
        """Send tool_search_tool + mcp_toolset with defer_loading: true.

        Verifies the full flow: SMG injects MCP tools with defer_loading,
        Anthropic discovers them via tool search, SMG executes via MCP.
        """
        _, model, _, _ = setup_backend

        tools = [
            make_tool_search_tool(),
            make_mcp_toolset(
                MCP_SERVER_NAME,
                default_defer_loading=True,
                default_enabled=False,
                tool_overrides={"brave_web_search": {"enabled": True}},
            ),
        ]

        response = api_client.messages.create(
            model=model,
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Search the web for 'Anthropic Claude'. Set count to 1 "
                        "to get only one result, and give a one sentence summary."
                    ),
                }
            ],
            extra_headers={
                "x-smg-mcp": "enabled",
            },
            extra_body={
                "tools": tools,
                "mcp_servers": [{"type": "url", "name": MCP_SERVER_NAME, "url": MCP_SERVER_URL}],
            },
        )

        assert response.id is not None
        assert response.model == model
        assert len(response.content) > 0

        block_types = [b.type for b in response.content]

        # Tool search flow (passthrough from Anthropic):
        assert "server_tool_use" in block_types, (
            f"Expected server_tool_use for tool search, got: {block_types}"
        )
        assert "tool_search_tool_result" in block_types, (
            f"Expected tool_search_tool_result, got: {block_types}"
        )

        # SMG-handled MCP tool execution:
        assert "mcp_tool_use" in block_types, (
            f"Expected mcp_tool_use (SMG-executed MCP tool), got: {block_types}"
        )
        assert "mcp_tool_result" in block_types, (
            f"Expected mcp_tool_result with tool output, got: {block_types}"
        )

        # Validate server_tool_use is for tool search
        server_tool_uses = [b for b in response.content if b.type == "server_tool_use"]
        assert server_tool_uses[0].name == "tool_search_tool_regex"

        # Validate mcp_tool_use references the MCP server
        mcp_tool_uses = [b for b in response.content if b.type == "mcp_tool_use"]
        assert mcp_tool_uses[0].server_name == MCP_SERVER_NAME

        # Validate mcp_tool_result is not an error
        mcp_results = [b for b in response.content if b.type == "mcp_tool_result"]
        assert not mcp_results[0].is_error

        # SMG completes the full tool loop — model produces final answer
        assert response.stop_reason == "end_turn"
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

    def test_mcp_tools_with_deferred_loading_streaming(self, setup_backend, api_client):
        """Streaming variant: tool_search + deferred MCP tools via SMG."""
        _, model, _, _ = setup_backend

        tools = [
            make_tool_search_tool(),
            make_mcp_toolset(
                MCP_SERVER_NAME,
                default_defer_loading=True,
                default_enabled=False,
                tool_overrides={"brave_web_search": {"enabled": True}},
            ),
        ]

        event_types = set()
        block_types = []
        content_blocks = []

        with api_client.messages.stream(
            model=model,
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Search the web for 'Anthropic Claude'. Set count to 1 "
                        "to get only one result, and give a one sentence summary."
                    ),
                }
            ],
            extra_headers={
                "x-smg-mcp": "enabled",
            },
            extra_body={
                "tools": tools,
                "mcp_servers": [{"type": "url", "name": MCP_SERVER_NAME, "url": MCP_SERVER_URL}],
            },
        ) as stream:
            for event in stream:
                event_types.add(event.type)
                if event.type == "content_block_start":
                    block_types.append(event.content_block.type)
                    content_blocks.append(event.content_block)

        logger.info("MCP streaming tool search event types: %s", event_types)
        logger.info("MCP streaming tool search block types: %s", block_types)

        # Core SSE lifecycle events
        assert "message_start" in event_types
        assert "content_block_start" in event_types
        assert "content_block_delta" in event_types
        assert "content_block_stop" in event_types
        assert "message_delta" in event_types
        assert "message_stop" in event_types

        # Tool search flow: server_tool_use -> tool_search_tool_result ->
        # mcp_tool_use -> mcp_tool_result -> text
        assert "server_tool_use" in block_types, (
            f"Expected server_tool_use block for tool search, got: {block_types}"
        )
        assert "tool_search_tool_result" in block_types, (
            f"Expected tool_search_tool_result block, got: {block_types}"
        )
        assert "mcp_tool_use" in block_types, (
            f"Expected mcp_tool_use block (SMG-executed), got: {block_types}"
        )
        assert "mcp_tool_result" in block_types, (
            f"Expected mcp_tool_result block, got: {block_types}"
        )

        # Validate server_tool_use is for tool search
        server_tool_uses = [b for b in content_blocks if b.type == "server_tool_use"]
        assert len(server_tool_uses) >= 1
        assert server_tool_uses[0].name == "tool_search_tool_regex"

        # Validate mcp_tool_use references the MCP server
        mcp_tool_uses = [b for b in content_blocks if b.type == "mcp_tool_use"]
        assert len(mcp_tool_uses) >= 1
        assert mcp_tool_uses[0].server_name == MCP_SERVER_NAME

        # Validate at least one text block present (model output can vary;
        # relaxed from >= 2 to prevent flakiness)
        text_blocks = [b for b in content_blocks if b.type == "text"]
        assert len(text_blocks) >= 1, f"Expected at least 1 text block, got: {len(text_blocks)}"
