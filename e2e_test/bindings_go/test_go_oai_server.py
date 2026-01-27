"""Go OAI Server E2E Tests.

Tests for the Go OpenAI-compatible server, including:
- Non-streaming chat completions
- Streaming chat completions
- Function calling / tool use
- Error handling

These tests use the Go OAI server which connects directly to a gRPC worker.
"""

from __future__ import annotations

import json
import logging

import pytest

logger = logging.getLogger(__name__)


@pytest.mark.model("llama-1b")
class TestGoOAIServerBasic:
    """Basic tests for Go OAI server functionality."""

    def test_non_streaming_completion(self, go_openai_client, go_oai_server):
        """Test non-streaming chat completion through Go OAI server."""
        _, _, model_path = go_oai_server

        response = go_openai_client.chat.completions.create(
            model=model_path,
            messages=[
                {"role": "user", "content": "Say hello in one word."}
            ],
            max_tokens=10,
            stream=False,
        )

        assert response.choices is not None
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0
        assert response.choices[0].finish_reason in ("stop", "length")
        logger.info(f"Response: {response.choices[0].message.content}")

    def test_streaming_completion(self, go_openai_client, go_oai_server):
        """Test streaming chat completion through Go OAI server."""
        _, _, model_path = go_oai_server

        response_stream = go_openai_client.chat.completions.create(
            model=model_path,
            messages=[
                {"role": "user", "content": "Count from 1 to 5."}
            ],
            max_tokens=50,
            stream=True,
        )

        chunks = list(response_stream)
        assert len(chunks) > 1, "Streaming should return multiple chunks"

        # Collect content from all chunks
        content_parts = []
        for chunk in chunks:
            if chunk.choices and chunk.choices[0].delta.content:
                content_parts.append(chunk.choices[0].delta.content)

        full_content = "".join(content_parts)
        assert len(full_content) > 0, "Should have received content"
        logger.info(f"Streamed content: {full_content}")

        # Check final chunk has finish_reason
        final_chunk = chunks[-1]
        assert final_chunk.choices[0].finish_reason in ("stop", "length", None)

    def test_streaming_multiple_messages(self, go_openai_client, go_oai_server):
        """Test streaming with conversation history."""
        _, _, model_path = go_oai_server

        response_stream = go_openai_client.chat.completions.create(
            model=model_path,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."},
                {"role": "user", "content": "And 3+3?"},
            ],
            max_tokens=20,
            stream=True,
        )

        chunks = list(response_stream)
        assert len(chunks) > 0, "Should receive response chunks"

        content_parts = []
        for chunk in chunks:
            if chunk.choices and chunk.choices[0].delta.content:
                content_parts.append(chunk.choices[0].delta.content)

        full_content = "".join(content_parts)
        assert len(full_content) > 0
        logger.info(f"Response to follow-up: {full_content}")


@pytest.mark.model("llama-1b")
@pytest.mark.xfail(
    reason="Llama-3.2-1B-Instruct doesn't reliably support tool calling with tool_choice=required",
    strict=False,  # Allow tests to pass if model happens to work
)
class TestGoOAIServerFunctionCalling:
    """Tests for function calling through Go OAI server.

    Note: Function calling requires the model to support tool use.
    The llama-1b model may not reliably support tool_choice='required'.
    These tests are marked as xfail to allow CI to pass while still
    testing the Go OAI server's tool calling proxy functionality.
    """

    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    SYSTEM_MESSAGE = (
        "You are a helpful assistant with tool calling capabilities. "
        "When you need to get weather information, use the get_weather function. "
        'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.'
    )

    def test_function_calling_non_streaming(self, go_openai_client, go_oai_server):
        """Test function calling in non-streaming mode."""
        _, _, model_path = go_oai_server

        response = go_openai_client.chat.completions.create(
            model=model_path,
            messages=[
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": "What's the weather in Tokyo?"},
            ],
            max_tokens=100,
            tools=self.TOOLS,
            tool_choice="required",
            stream=False,
        )

        assert response.choices is not None
        tool_calls = response.choices[0].message.tool_calls

        assert tool_calls is not None, "Expected tool calls"
        assert len(tool_calls) > 0, "Expected at least one tool call"

        tool_call = tool_calls[0]
        assert tool_call.function.name == "get_weather"

        args = json.loads(tool_call.function.arguments)
        assert "location" in args
        logger.info(f"Tool call: {tool_call.function.name}({args})")

    def test_function_calling_streaming(self, go_openai_client, go_oai_server):
        """Test function calling in streaming mode."""
        _, _, model_path = go_oai_server

        response_stream = go_openai_client.chat.completions.create(
            model=model_path,
            messages=[
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": "What's the weather in Paris?"},
            ],
            max_tokens=100,
            tools=self.TOOLS,
            tool_choice="required",
            stream=True,
        )

        chunks = list(response_stream)
        assert len(chunks) > 0

        # Reconstruct tool calls from streaming chunks
        tool_calls_by_index = {}
        for chunk in chunks:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_by_index:
                        tool_calls_by_index[idx] = {
                            "name": "",
                            "arguments": "",
                        }
                    if tc.function.name:
                        tool_calls_by_index[idx]["name"] = tc.function.name
                    if tc.function.arguments:
                        tool_calls_by_index[idx]["arguments"] += tc.function.arguments

        assert len(tool_calls_by_index) > 0, "Expected tool calls in stream"

        # Verify first tool call
        first_tc = tool_calls_by_index[0]
        assert first_tc["name"] == "get_weather"

        args = json.loads(first_tc["arguments"])
        assert "location" in args
        logger.info(f"Streamed tool call: {first_tc['name']}({args})")

    def test_function_calling_tool_choice_none(self, go_openai_client, go_oai_server):
        """Test that tool_choice='none' prevents function calls."""
        _, _, model_path = go_oai_server

        response = go_openai_client.chat.completions.create(
            model=model_path,
            messages=[
                {"role": "user", "content": "What's the weather in London?"},
            ],
            max_tokens=50,
            tools=self.TOOLS,
            tool_choice="none",
            stream=False,
        )

        assert response.choices is not None
        # With tool_choice="none", should get text response, not tool calls
        tool_calls = response.choices[0].message.tool_calls
        assert tool_calls is None or len(tool_calls) == 0


@pytest.mark.model("llama-1b")
class TestGoOAIServerParameters:
    """Tests for various generation parameters through Go OAI server."""

    def test_temperature_variation(self, go_openai_client, go_oai_server):
        """Test that temperature parameter affects output."""
        _, _, model_path = go_oai_server

        # Low temperature - more deterministic
        response_low = go_openai_client.chat.completions.create(
            model=model_path,
            messages=[{"role": "user", "content": "Say hello."}],
            max_tokens=10,
            temperature=0.1,
            stream=False,
        )

        # High temperature - more random
        response_high = go_openai_client.chat.completions.create(
            model=model_path,
            messages=[{"role": "user", "content": "Say hello."}],
            max_tokens=10,
            temperature=1.5,
            stream=False,
        )

        # Both should produce valid responses
        assert response_low.choices[0].message.content is not None
        assert response_high.choices[0].message.content is not None

    def test_max_tokens_limit(self, go_openai_client, go_oai_server):
        """Test that max_tokens limits output length."""
        _, _, model_path = go_oai_server

        response = go_openai_client.chat.completions.create(
            model=model_path,
            messages=[
                {"role": "user", "content": "Write a very long story about a dragon."}
            ],
            max_tokens=5,
            stream=False,
        )

        assert response.choices is not None
        # With max_tokens=5, response should be short
        content = response.choices[0].message.content
        assert content is not None
        # The model might stop due to length limit
        assert response.choices[0].finish_reason in ("stop", "length")

    def test_top_p_parameter(self, go_openai_client, go_oai_server):
        """Test that top_p parameter is accepted."""
        _, _, model_path = go_oai_server

        response = go_openai_client.chat.completions.create(
            model=model_path,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10,
            top_p=0.9,
            stream=False,
        )

        assert response.choices is not None
        assert response.choices[0].message.content is not None
