"""MCP tool use tests for Anthropic Messages API.

Tests for MCP (Model Context Protocol) tool execution through the SMG gateway,
including non-streaming and streaming modes, in both SMG-handled and passthrough modes.

Requirements:
- ANTHROPIC_API_KEY environment variable must be set
- For smg_handled mode: Brave MCP server must be running (see BRAVE_MCP_URL in infra.constants)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import pytest
from conftest import smg_compare
from infra import BRAVE_MCP_URL

logger = logging.getLogger(__name__)


# =============================================================================
# MCP Test Configurations
# =============================================================================


@dataclass
class McpTestConfig:
    """Configuration for an MCP test scenario."""

    server_name: str
    server_url: str
    headers: dict[str, str]
    prompt: str


# SMG-handled MCP: x-smg-mcp header present, SMG orchestrates tool loop
SMG_HANDLED = McpTestConfig(
    server_name="brave",
    server_url=BRAVE_MCP_URL,
    headers={
        "anthropic-beta": "mcp-client-2025-11-20",
        "x-smg-mcp": "enabled",
    },
    prompt=(
        "search the web for 'Anthropic Claude AI'. Set count to 1 to get "
        "only one result, and give a one sentence summary."
    ),
)

# Passthrough MCP: no x-smg-mcp header, request forwarded to Anthropic backend
PASSTHROUGH = McpTestConfig(
    server_name="dmcp",
    server_url="https://dmcp-server.deno.dev/sse",
    headers={"anthropic-beta": "mcp-client-2025-11-20"},
    prompt="Roll 2d4+1",
)


def _extra_body(cfg: McpTestConfig) -> dict:
    """Build extra_body for MCP requests."""
    return {
        "mcp_servers": [{"type": "url", "name": cfg.server_name, "url": cfg.server_url}],
        "tools": [{"type": "mcp_toolset", "mcp_server_name": cfg.server_name}],
    }


# =============================================================================
# Shared Assertion Helpers
# =============================================================================


def assert_non_streaming_mcp_response(response, model: str, server_name: str):
    """Validate a non-streaming MCP response structure."""
    assert response.id is not None
    assert response.model == model
    assert response.stop_reason == "end_turn"
    assert response.role == "assistant"
    assert len(response.content) > 0

    mcp_tool_use_blocks = [b for b in response.content if b.type == "mcp_tool_use"]
    mcp_tool_result_blocks = [b for b in response.content if b.type == "mcp_tool_result"]
    text_blocks = [b for b in response.content if b.type == "text"]

    assert len(mcp_tool_use_blocks) > 0, "Should have at least one mcp_tool_use block"
    assert len(mcp_tool_result_blocks) > 0, "Should have at least one mcp_tool_result block"
    assert len(text_blocks) > 0, "Should have a final text block"

    # Validate mcp_tool_use structure
    tool_use = mcp_tool_use_blocks[0]
    assert tool_use.id.startswith("mcptoolu_")
    assert tool_use.name is not None
    assert tool_use.server_name == server_name
    assert isinstance(tool_use.input, dict)

    # Validate tool_use / tool_result pairing
    assert len(mcp_tool_result_blocks) == len(mcp_tool_use_blocks), (
        f"Mismatch: {len(mcp_tool_result_blocks)} results vs {len(mcp_tool_use_blocks)} tool uses"
    )
    for i, tool_result in enumerate(mcp_tool_result_blocks):
        assert tool_result.tool_use_id == mcp_tool_use_blocks[i].id, (
            f"tool_result[{i}].tool_use_id mismatch: "
            f"{tool_result.tool_use_id} != {mcp_tool_use_blocks[i].id}"
        )
        assert tool_result.content is not None, f"tool_result[{i}].content is None"

    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0

    logger.info(
        "MCP non-stream: %d tool_use, %d tool_result, %d text blocks",
        len(mcp_tool_use_blocks),
        len(mcp_tool_result_blocks),
        len(text_blocks),
    )


def collect_streaming_events(stream):
    """Collect events from an MCP streaming response."""
    event_types = set()
    block_types = []
    input_json_deltas_by_index: dict[int, list[str]] = {}
    text_deltas = []
    mcp_tool_use_ids = []

    for event in stream:
        event_types.add(event.type)

        if event.type == "content_block_start":
            block_types.append(event.content_block.type)
            if event.content_block.type == "mcp_tool_use":
                mcp_tool_use_ids.append(event.content_block.id)

        if event.type == "content_block_delta":
            if event.delta.type == "input_json_delta":
                idx = event.index
                input_json_deltas_by_index.setdefault(idx, []).append(event.delta.partial_json)
            elif event.delta.type == "text_delta":
                text_deltas.append(event.delta.text)

    return event_types, block_types, mcp_tool_use_ids, input_json_deltas_by_index, text_deltas


def assert_streaming_mcp_response(
    event_types, block_types, mcp_tool_use_ids, input_json_deltas_by_index, text_deltas
):
    """Validate streaming MCP SSE events."""
    assert "message_start" in event_types, "Missing message_start event"
    assert "content_block_start" in event_types, "Missing content_block_start event"
    assert "content_block_stop" in event_types, "Missing content_block_stop event"
    assert "message_delta" in event_types, "Missing message_delta event"
    assert "message_stop" in event_types, "Missing message_stop event"

    assert "mcp_tool_use" in block_types, "Should have mcp_tool_use content block"
    assert "mcp_tool_result" in block_types, "Should have mcp_tool_result content block"
    assert "text" in block_types, "Should have text content block"

    assert len(mcp_tool_use_ids) > 0
    assert all(tid.startswith("mcptoolu_") for tid in mcp_tool_use_ids)

    assert len(input_json_deltas_by_index) > 0, "Should have input_json_delta events"

    for idx, fragments in input_json_deltas_by_index.items():
        full_json = "".join(fragments)
        if full_json:
            try:
                parsed = json.loads(full_json)
            except json.JSONDecodeError as exc:
                pytest.fail(f"Failed to parse tool input at index {idx}: {full_json!r} -> {exc}")
            assert isinstance(parsed, dict), f"Tool input at index {idx} should be a dict"

    assert len(text_deltas) > 0, "Should have text_delta events"

    logger.info(
        "MCP stream: %d block_starts, %d tool calls, %d text deltas",
        len(block_types),
        len(mcp_tool_use_ids),
        len(text_deltas),
    )


# =============================================================================
# MCP Tool Tests — SMG-handled (X-SMG-MCP: enabled)
# =============================================================================


@pytest.mark.vendor("anthropic")
@pytest.mark.gpu(0)
@pytest.mark.parametrize("setup_backend", ["anthropic"], indirect=True)
class TestMcpTool:
    """MCP tool use tests with SMG orchestration (X-SMG-MCP: enabled)."""

    def test_mcp_non_streaming(self, setup_backend, smg):
        """Test MCP tool execution in non-streaming mode."""
        cfg = SMG_HANDLED
        _, model, client, _ = setup_backend

        response = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": cfg.prompt}],
            extra_headers=cfg.headers,
            extra_body=_extra_body(cfg),
        )

        assert_non_streaming_mcp_response(response, model, cfg.server_name)

        # SmgClient comparison
        with smg_compare():
            smg_resp = smg.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": cfg.prompt}],
                extra_headers=cfg.headers,
                extra_body=_extra_body(cfg),
            )
            assert smg_resp.id is not None
            assert smg_resp.role == "assistant"
            assert len(smg_resp.content) > 0

    def test_mcp_streaming(self, setup_backend, smg):
        """Test MCP tool execution with SSE streaming."""
        cfg = SMG_HANDLED
        _, model, client, _ = setup_backend

        with client.messages.stream(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": cfg.prompt}],
            extra_headers=cfg.headers,
            extra_body=_extra_body(cfg),
        ) as stream:
            results = collect_streaming_events(stream)

        assert_streaming_mcp_response(*results)

        # SmgClient streaming comparison
        with smg_compare():
            with smg.messages.stream(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": cfg.prompt}],
                extra_headers=cfg.headers,
                extra_body=_extra_body(cfg),
            ) as smg_stream:
                smg_results = collect_streaming_events(smg_stream)

            assert_streaming_mcp_response(*smg_results)


# =============================================================================
# MCP Passthrough Tests — external DMCP server (run with: pytest -m external)
# =============================================================================


@pytest.mark.vendor("anthropic")
@pytest.mark.gpu(0)
@pytest.mark.external
@pytest.mark.parametrize("setup_backend", ["anthropic"], indirect=True)
class TestMcpToolPassthrough:
    """MCP passthrough tests using external DMCP server.

    No X-SMG-MCP header — request forwarded to Anthropic backend as-is.
    Requires external https://dmcp-server.deno.dev/sse to be reachable.
    """

    def test_mcp_passthrough_non_streaming(self, setup_backend, smg):
        """Test MCP passthrough in non-streaming mode."""
        cfg = PASSTHROUGH
        _, model, client, _ = setup_backend

        response = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": cfg.prompt}],
            extra_headers=cfg.headers,
            extra_body=_extra_body(cfg),
        )

        assert_non_streaming_mcp_response(response, model, cfg.server_name)

        # SmgClient comparison
        with smg_compare():
            smg_resp = smg.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": cfg.prompt}],
                extra_headers=cfg.headers,
                extra_body=_extra_body(cfg),
            )
            assert smg_resp.id is not None
            assert smg_resp.role == "assistant"
            assert len(smg_resp.content) > 0

    def test_mcp_passthrough_streaming(self, setup_backend, smg):
        """Test MCP passthrough with SSE streaming."""
        cfg = PASSTHROUGH
        _, model, client, _ = setup_backend

        with client.messages.stream(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": cfg.prompt}],
            extra_headers=cfg.headers,
            extra_body=_extra_body(cfg),
        ) as stream:
            results = collect_streaming_events(stream)

        assert_streaming_mcp_response(*results)

        # SmgClient streaming comparison
        with smg_compare():
            with smg.messages.stream(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": cfg.prompt}],
                extra_headers=cfg.headers,
                extra_body=_extra_body(cfg),
            ) as smg_stream:
                smg_results = collect_streaming_events(smg_stream)

            assert_streaming_mcp_response(*smg_results)
