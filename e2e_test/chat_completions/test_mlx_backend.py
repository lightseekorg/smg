"""MLX backend E2E tests (Apple Silicon only).

Tests the SMG router → gRPC → MLX worker pipeline using mlx-lm's
BatchGenerator. The MLX backend only supports gRPC mode.

Run locally with:
    E2E_RUNTIME=mlx pytest e2e_test/chat_completions/test_mlx_backend.py -v

CI runs on macos-14 (Apple Silicon) with the smallest Qwen3 model (~400 MB).
See .github/workflows/pr-test-mlx.yml.
"""

from __future__ import annotations

import json
import platform
import sys

import pytest

# MLX is Apple Silicon only — skip the entire module elsewhere.
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="MLX backend requires Apple Silicon (macOS arm64)",
)


# ── Tools used for function calling tests ────────────────────────────────────

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name, e.g. 'Tokyo' or 'Paris'.",
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


@pytest.mark.engine("mlx")
@pytest.mark.gpu(0)  # MLX uses unified memory, no discrete GPU
@pytest.mark.model("mlx-community/Qwen3-0.6B-4bit")
@pytest.mark.gateway(extra_args=["--tool-call-parser", "qwen"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestMlxBackend:
    """End-to-end tests for the MLX gRPC backend.

    Uses mlx-community/Qwen3-0.6B-4bit (~400 MB) — small enough to download
    and run on a macos-latest GitHub Actions runner. Qwen3 supports both
    native tool calling and thinking mode in a single model.
    """

    # Qwen3 enters thinking mode by default; disable for the basic chat tests
    # so the small 0.6B model produces actual content within max_tokens budget.
    NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}

    def test_basic_chat(self, model, api_client):
        """Non-streaming chat completion goes through the full pipeline."""
        response = api_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with the single word 'OK'."}],
            max_tokens=20,
            temperature=0,
            extra_body=self.NO_THINKING,
        )
        assert response.id
        assert response.choices[0].message.role == "assistant"
        assert isinstance(response.choices[0].message.content, str)
        assert len(response.choices[0].message.content) > 0
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0

    def test_streaming_chat(self, model, api_client):
        """Streaming chat yields incremental delta chunks."""
        stream = api_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Count: 1, 2, 3."}],
            max_tokens=30,
            temperature=0,
            stream=True,
            extra_body=self.NO_THINKING,
        )
        chunks = list(stream)
        # First chunk has the role, subsequent chunks carry content deltas.
        assert len(chunks) >= 2
        text_pieces = [
            c.choices[0].delta.content for c in chunks if c.choices and c.choices[0].delta.content
        ]
        assert len(text_pieces) > 0
        assert "".join(text_pieces).strip()

    def test_tool_calling(self, model, api_client):
        """Qwen3 emits <tool_call> tags; SMG's qwen ToolParser extracts them."""
        response = api_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "What's the weather in Tokyo right now?"},
            ],
            tools=[WEATHER_TOOL],
            tool_choice="auto",
            max_tokens=300,
            temperature=0,
        )
        msg = response.choices[0].message
        assert response.choices[0].finish_reason == "tool_calls", (
            f"Expected finish_reason=tool_calls, got {response.choices[0].finish_reason!r} "
            f"with content={msg.content!r}"
        )
        assert msg.tool_calls, "Expected tool_calls to be populated"
        call = msg.tool_calls[0]
        assert call.function.name == "get_weather"
        # arguments is a JSON string per OpenAI spec
        args = json.loads(call.function.arguments)
        assert "tokyo" in args.get("location", "").lower()

    def test_reasoning_thinking_mode(self, model, api_client):
        """Qwen3 thinking mode populates reasoning_content separately from content."""
        response = api_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Alice has 5 apples. She gives 2 to Bob. "
                        "How many apples does Alice have? Think step by step."
                    ),
                }
            ],
            max_tokens=600,
            temperature=0,
        )
        msg = response.choices[0].message
        # Qwen3-0.6B may or may not always use thinking mode for trivial questions,
        # but at minimum the final answer must mention 3.
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", None) or ""
        full = content + reasoning
        assert (
            "3" in full
        ), f"Expected '3' in response, got content={content!r} reasoning={reasoning!r}"

    def test_max_tokens_finish_reason(self, model, api_client):
        """When max_tokens is reached, finish_reason is 'length'."""
        response = api_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Tell me a story."}],
            max_tokens=10,
            temperature=0,
            extra_body=self.NO_THINKING,
        )
        assert response.choices[0].finish_reason == "length"
        assert response.usage.completion_tokens == 10
