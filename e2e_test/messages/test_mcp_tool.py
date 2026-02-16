"""MCP tool use tests for Anthropic Messages API.

Tests for MCP (Model Context Protocol) tool execution through the SMG gateway,
including non-streaming and streaming modes.

Requirements:
- ANTHROPIC_API_KEY environment variable must be set
- MCP server must be running (configure via MCP_SERVER_URL env var, default: http://localhost:8001/sse)
"""

from __future__ import annotations

import json
import logging
import os

import pytest

logger = logging.getLogger(__name__)

# MCP server configuration
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8001/sse")
MCP_SERVER_NAME = os.environ.get("MCP_SERVER_NAME", "brave")

MCP_EXTRA_HEADERS = {"anthropic-beta": "mcp-client-2025-11-20"}


def _mcp_extra_body() -> dict:
    """Build extra_body for MCP requests."""
    return {
        "mcp_servers": [
            {
                "type": "url",
                "name": MCP_SERVER_NAME,
                "url": MCP_SERVER_URL,
            }
        ],
        "tools": [
            {
                "type": "mcp_toolset",
                "mcp_server_name": MCP_SERVER_NAME,
            }
        ],
    }


# =============================================================================
# MCP Tool Non-Streaming Tests
# =============================================================================


@pytest.mark.parametrize("setup_backend", ["anthropic"], indirect=True)
class TestMcpToolNonStream:
    """MCP tool use tests without streaming."""

    def test_mcp_tool_non_streaming(self, setup_backend):
        """Test MCP tool execution in non-streaming mode.

        Verifies:
        - mcp_tool_use blocks with server_name, id, name, input
        - mcp_tool_result blocks with tool_use_id and content
        - Final text summary block
        - Correct model and stop_reason
        """
        _, model, client, _ = setup_backend

        response = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": "search the web for 'Anthropic Claude AI' and give me a brief summary",
                }
            ],
            extra_headers=MCP_EXTRA_HEADERS,
            extra_body=_mcp_extra_body(),
        )

        assert response.id is not None
        assert response.model == model
        assert response.stop_reason == "end_turn"
        assert response.role == "assistant"
        assert len(response.content) > 0

        # Collect block types
        mcp_tool_use_blocks = [b for b in response.content if b.type == "mcp_tool_use"]
        mcp_tool_result_blocks = [b for b in response.content if b.type == "mcp_tool_result"]
        text_blocks = [b for b in response.content if b.type == "text"]

        # Should have at least one tool use + result pair
        assert len(mcp_tool_use_blocks) > 0, "Should have at least one mcp_tool_use block"
        assert len(mcp_tool_result_blocks) > 0, "Should have at least one mcp_tool_result block"
        assert len(text_blocks) > 0, "Should have a final text summary"

        # Validate mcp_tool_use structure
        tool_use = mcp_tool_use_blocks[0]
        assert tool_use.id.startswith("mcptoolu_")
        assert tool_use.name is not None
        assert tool_use.server_name == MCP_SERVER_NAME
        assert isinstance(tool_use.input, dict)

        # Validate all mcp_tool_result blocks match their corresponding tool_use
        assert len(mcp_tool_result_blocks) == len(mcp_tool_use_blocks), (
            f"Mismatch: {len(mcp_tool_result_blocks)} results vs {len(mcp_tool_use_blocks)} tool uses"
        )
        for i, tool_result in enumerate(mcp_tool_result_blocks):
            assert tool_result.tool_use_id == mcp_tool_use_blocks[i].id, (
                f"tool_result[{i}].tool_use_id mismatch: "
                f"{tool_result.tool_use_id} != {mcp_tool_use_blocks[i].id}"
            )
            assert tool_result.content is not None, f"tool_result[{i}].content is None"

        # Validate usage
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

        logger.info(
            "MCP non-stream: %d tool_use, %d tool_result, %d text blocks",
            len(mcp_tool_use_blocks),
            len(mcp_tool_result_blocks),
            len(text_blocks),
        )


# =============================================================================
# MCP Tool Streaming Tests
# =============================================================================


@pytest.mark.parametrize("setup_backend", ["anthropic"], indirect=True)
class TestMcpToolStream:
    """MCP tool use tests with SSE streaming."""

    def test_mcp_tool_streaming(self, setup_backend):
        """Test MCP tool execution with SSE streaming.

        Verifies:
        - message_start event with model info
        - content_block_start events for mcp_tool_use, mcp_tool_result, and text
        - content_block_delta events with text_delta and input_json_delta
        - content_block_stop events
        - message_delta with stop_reason
        - message_stop event
        """
        _, model, client, _ = setup_backend

        event_types = set()
        block_types = []
        input_json_deltas_by_index: dict[int, list[str]] = {}
        text_deltas = []
        mcp_tool_use_ids = []

        with client.messages.stream(
            model=model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": "search the web for 'Anthropic Claude AI' and give me a brief summary",
                }
            ],
            extra_headers=MCP_EXTRA_HEADERS,
            extra_body=_mcp_extra_body(),
        ) as stream:
            for event in stream:
                event_types.add(event.type)

                if event.type == "content_block_start":
                    block_types.append(event.content_block.type)
                    if event.content_block.type == "mcp_tool_use":
                        mcp_tool_use_ids.append(event.content_block.id)

                if event.type == "content_block_delta":
                    if event.delta.type == "input_json_delta":
                        idx = event.index
                        input_json_deltas_by_index.setdefault(idx, []).append(
                            event.delta.partial_json
                        )
                    elif event.delta.type == "text_delta":
                        text_deltas.append(event.delta.text)

        # Required SSE event types
        assert "message_start" in event_types, "Missing message_start event"
        assert "content_block_start" in event_types, "Missing content_block_start event"
        assert "content_block_stop" in event_types, "Missing content_block_stop event"
        assert "message_delta" in event_types, "Missing message_delta event"
        assert "message_stop" in event_types, "Missing message_stop event"

        # Should contain MCP tool blocks and text
        assert "mcp_tool_use" in block_types, "Should have mcp_tool_use content block"
        assert "mcp_tool_result" in block_types, "Should have mcp_tool_result content block"
        assert "text" in block_types, "Should have text content block"

        # Validate mcp_tool_use IDs
        assert len(mcp_tool_use_ids) > 0
        assert all(tid.startswith("mcptoolu_") for tid in mcp_tool_use_ids)

        # Should have input_json_delta events for tool input
        assert len(input_json_deltas_by_index) > 0, "Should have input_json_delta events"

        # Verify each tool's partial JSON forms valid JSON
        for idx, fragments in input_json_deltas_by_index.items():
            full_json = "".join(fragments)
            if full_json:
                try:
                    parsed = json.loads(full_json)
                except json.JSONDecodeError as exc:
                    pytest.fail(
                        f"Failed to parse tool input at index {idx}: {full_json!r} -> {exc}"
                    )
                assert isinstance(parsed, dict), f"Tool input at index {idx} should be a dict"

        # Should have text_delta events
        assert len(text_deltas) > 0, "Should have text_delta events"

        logger.info(
            "MCP stream: %d block_starts, %d tool calls, %d text deltas",
            len(block_types),
            len(mcp_tool_use_ids),
            len(text_deltas),
        )
