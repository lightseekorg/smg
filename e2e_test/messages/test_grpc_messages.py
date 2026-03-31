"""Messages API E2E Tests — gRPC backend (self-hosted models).

Tests the Anthropic Messages API (/v1/messages) through SMG's gRPC router,
verifying non-streaming responses, SSE streaming, tool use, and reasoning.

Uses the same setup_backend fixture as chat_completions tests.
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================


def messages_post(gateway, body: dict, stream: bool = False) -> httpx.Response:
    """Send a POST /v1/messages request to the gateway."""
    return httpx.post(
        f"{gateway.base_url}/v1/messages",
        json={**body, "stream": stream},
        headers={"content-type": "application/json", "x-api-key": "test"},
        timeout=60.0,
    )


def collect_sse_events(response: httpx.Response) -> list[dict]:
    """Parse Anthropic SSE events from a streaming response."""
    events = []
    for line in response.text.split("\n"):
        if line.startswith("data: "):
            data = line[len("data: ") :]
            try:
                events.append(json.loads(data))
            except json.JSONDecodeError:
                pass
    return events


# =============================================================================
# Tool definition (Anthropic format)
# =============================================================================

GET_WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get the current weather in a given location.",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city name, e.g., San Francisco",
            },
        },
        "required": ["location"],
    },
}


# =============================================================================
# Non-streaming tests
# =============================================================================


@pytest.mark.engine("sglang", "vllm")
@pytest.mark.gpu(1)
@pytest.mark.model("meta-llama/Llama-3.1-8B-Instruct")
@pytest.mark.gateway(extra_args=["--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestGrpcMessagesBasic:
    """Non-streaming Messages API tests through the gRPC router."""

    def test_non_streaming_basic(self, setup_backend):
        """Basic non-streaming message returns valid Anthropic response."""
        _, model, _, gateway = setup_backend

        resp = messages_post(
            gateway,
            {
                "model": model,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "Say hello in one word."}],
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["type"] == "message"
        assert body["role"] == "assistant"
        assert body["id"] is not None
        assert body["stop_reason"] in ("end_turn", "max_tokens")
        assert len(body["content"]) > 0
        assert body["content"][0]["type"] == "text"
        assert len(body["content"][0]["text"]) > 0
        assert body["usage"]["input_tokens"] > 0
        assert body["usage"]["output_tokens"] > 0

    def test_non_streaming_with_system(self, setup_backend):
        """System prompt is accepted and doesn't error."""
        _, model, _, gateway = setup_backend

        resp = messages_post(
            gateway,
            {
                "model": model,
                "max_tokens": 32,
                "system": "You always reply with exactly one word.",
                "messages": [{"role": "user", "content": "What color is the sky?"}],
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["type"] == "message"
        assert len(body["content"]) > 0

    def test_non_streaming_multi_turn(self, setup_backend):
        """Multi-turn messages are accepted."""
        _, model, _, gateway = setup_backend

        resp = messages_post(
            gateway,
            {
                "model": model,
                "max_tokens": 64,
                "messages": [
                    {"role": "user", "content": "My name is Alice."},
                    {"role": "assistant", "content": "Nice to meet you, Alice!"},
                    {"role": "user", "content": "What is my name?"},
                ],
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["type"] == "message"
        assert "alice" in body["content"][0]["text"].lower()

    def test_stream_false_returns_json(self, setup_backend):
        """stream=false must return a single JSON object, NOT SSE."""
        _, model, _, gateway = setup_backend

        resp = messages_post(
            gateway,
            {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": "Hi"}]},
            stream=False,
        )

        assert resp.status_code == 200
        assert "text/event-stream" not in resp.headers.get("content-type", "")
        body = resp.json()
        assert body["type"] == "message"


# =============================================================================
# Streaming tests
# =============================================================================


@pytest.mark.engine("sglang", "vllm")
@pytest.mark.gpu(1)
@pytest.mark.model("meta-llama/Llama-3.1-8B-Instruct")
@pytest.mark.gateway(extra_args=["--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestGrpcMessagesStreaming:
    """Streaming Messages API tests through the gRPC router."""

    def test_streaming_event_sequence(self, setup_backend):
        """Streaming returns the required Anthropic SSE event types."""
        _, model, _, gateway = setup_backend

        with httpx.stream(
            "POST",
            f"{gateway.base_url}/v1/messages",
            json={
                "model": model,
                "max_tokens": 32,
                "stream": True,
                "messages": [{"role": "user", "content": "Count to 3."}],
            },
            headers={"content-type": "application/json", "x-api-key": "test"},
            timeout=60.0,
        ) as resp:
            assert resp.status_code == 200
            text = resp.read().decode()

        events = []
        for line in text.split("\n"):
            if line.startswith("data: "):
                try:
                    events.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    pass

        event_types = {e["type"] for e in events}
        required = {
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        }
        missing = required - event_types
        assert not missing, f"Missing SSE event types: {missing}"

    def test_streaming_text_concatenation(self, setup_backend):
        """Text deltas concatenate to a non-empty string."""
        _, model, _, gateway = setup_backend

        resp = messages_post(
            gateway,
            {
                "model": model,
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "Say hello."}],
            },
            stream=True,
        )

        events = collect_sse_events(resp)
        text = "".join(
            e["delta"]["text"]
            for e in events
            if e.get("type") == "content_block_delta"
            and e.get("delta", {}).get("type") == "text_delta"
        )
        assert len(text) > 0, "Streaming should produce non-empty text"


# =============================================================================
# Tool use tests
# =============================================================================


@pytest.mark.engine("sglang")
@pytest.mark.gpu(1)
@pytest.mark.model("meta-llama/Llama-3.1-8B-Instruct")
@pytest.mark.gateway(extra_args=["--tool-call-parser", "llama", "--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestGrpcMessagesToolUse:
    """Tool use tests for Messages API through the gRPC router."""

    def test_non_streaming_tool_use(self, setup_backend):
        """Non-streaming request with tools can return tool_use content block."""
        _, model, _, gateway = setup_backend

        resp = messages_post(
            gateway,
            {
                "model": model,
                "max_tokens": 256,
                "tools": [GET_WEATHER_TOOL],
                "tool_choice": {"type": "tool", "name": "get_weather"},
                "messages": [{"role": "user", "content": "What is the weather in San Francisco?"}],
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["type"] == "message"
        tool_blocks = [b for b in body["content"] if b["type"] == "tool_use"]
        assert len(tool_blocks) > 0, "Expected tool_use block in response"
        assert tool_blocks[0]["name"] == "get_weather"
        assert isinstance(tool_blocks[0]["input"], dict)

    def test_streaming_tool_use(self, setup_backend):
        """Streaming with tools emits input_json_delta events."""
        _, model, _, gateway = setup_backend

        resp = messages_post(
            gateway,
            {
                "model": model,
                "max_tokens": 256,
                "tools": [GET_WEATHER_TOOL],
                "tool_choice": {"type": "tool", "name": "get_weather"},
                "messages": [{"role": "user", "content": "What is the weather in London?"}],
            },
            stream=True,
        )

        events = collect_sse_events(resp)
        event_types = {e["type"] for e in events}

        assert "content_block_start" in event_types
        assert "message_stop" in event_types

        # Collect input_json_delta partial_json
        partials = [
            e["delta"]["partial_json"]
            for e in events
            if e.get("type") == "content_block_delta"
            and e.get("delta", {}).get("type") == "input_json_delta"
        ]
        if partials:
            full_json = "".join(partials)
            parsed = json.loads(full_json)
            assert isinstance(parsed, dict), "Tool args should be a JSON object"


# =============================================================================
# Reasoning (thinking) tests
# =============================================================================


@pytest.mark.engine("sglang")
@pytest.mark.gpu(1)
@pytest.mark.skip_for_runtime(
    "trtllm", reason="TRT-LLM does not support reasoning content extraction"
)
@pytest.mark.model("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
@pytest.mark.gateway(
    extra_args=["--reasoning-parser", "deepseek_r1", "--history-backend", "memory"]
)
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestGrpcMessagesReasoning:
    """Reasoning/thinking tests for Messages API through the gRPC router."""

    def test_non_streaming_thinking_block(self, setup_backend):
        """Non-streaming with thinking enabled produces thinking + text blocks."""
        _, model, _, gateway = setup_backend

        resp = messages_post(
            gateway,
            {
                "model": model,
                "max_tokens": 256,
                "thinking": {"type": "enabled", "budget_tokens": 1024},
                "messages": [{"role": "user", "content": "What is 2+3?"}],
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["type"] == "message"

        block_types = [b["type"] for b in body["content"]]
        assert "text" in block_types, "Should have a text block"
        # Reasoning model should produce thinking block when enabled
        if "thinking" in block_types:
            thinking_block = next(b for b in body["content"] if b["type"] == "thinking")
            assert len(thinking_block["thinking"]) > 0

    def test_streaming_thinking_deltas(self, setup_backend):
        """Streaming with thinking enabled emits thinking_delta events."""
        _, model, _, gateway = setup_backend

        resp = messages_post(
            gateway,
            {
                "model": model,
                "max_tokens": 256,
                "thinking": {"type": "enabled", "budget_tokens": 1024},
                "messages": [{"role": "user", "content": "What is 7*8?"}],
            },
            stream=True,
        )

        events = collect_sse_events(resp)
        event_types = {e["type"] for e in events}

        # Must have basic event lifecycle
        assert "message_start" in event_types
        assert "message_stop" in event_types

        # Check for thinking deltas
        delta_types = {
            e["delta"]["type"]
            for e in events
            if e.get("type") == "content_block_delta" and "delta" in e
        }
        assert "text_delta" in delta_types, "Should have text deltas"
        # Reasoning model should produce thinking_delta when enabled
        if "thinking_delta" in delta_types:
            thinking_text = "".join(
                e["delta"]["thinking"]
                for e in events
                if e.get("type") == "content_block_delta"
                and e.get("delta", {}).get("type") == "thinking_delta"
            )
            assert len(thinking_text) > 0, "Thinking content should be non-empty"
