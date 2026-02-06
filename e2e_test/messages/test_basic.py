"""Basic tests for Anthropic Messages API.

Tests for non-streaming and streaming message creation, system prompts,
and multi-turn conversations via the Anthropic SDK.
"""

from __future__ import annotations

import logging

import pytest

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("setup_backend", ["anthropic"], indirect=True)
class TestMessagesBasic:
    """Basic message creation tests against the Anthropic Messages API."""

    def test_non_streaming_basic(self, setup_backend):
        """Test basic non-streaming message creation."""
        _, model, client, gateway = setup_backend

        response = client.messages.create(
            model=model,
            max_tokens=100,
            messages=[{"role": "user", "content": "Say hello in one sentence."}],
        )

        assert response.id is not None
        assert response.role == "assistant"
        assert response.stop_reason == "end_turn"
        assert response.content is not None
        assert len(response.content) > 0
        assert response.content[0].type == "text"
        assert len(response.content[0].text) > 0
        assert response.usage is not None
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

    def test_non_streaming_with_system(self, setup_backend):
        """Test non-streaming message with system prompt."""
        _, model, client, gateway = setup_backend

        response = client.messages.create(
            model=model,
            max_tokens=50,
            system="You are a helpful assistant. Always respond in exactly one word.",
            messages=[{"role": "user", "content": "What color is the sky?"}],
        )

        assert response.id is not None
        assert response.stop_reason == "end_turn"
        assert len(response.content) > 0
        assert response.content[0].type == "text"
        # With the "one word" instruction, response should be short
        assert len(response.content[0].text.split()) <= 10

    def test_non_streaming_multi_turn(self, setup_backend):
        """Test multi-turn conversation preserves context."""
        _, model, client, gateway = setup_backend

        response = client.messages.create(
            model=model,
            max_tokens=100,
            messages=[
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Nice to meet you, Alice!"},
                {"role": "user", "content": "What is my name?"},
            ],
        )

        assert response.id is not None
        assert response.stop_reason == "end_turn"
        assert len(response.content) > 0
        assert response.content[0].type == "text"
        assert "alice" in response.content[0].text.lower()

    def test_streaming_basic(self, setup_backend):
        """Test streaming message creation returns expected event types."""
        _, model, client, gateway = setup_backend

        expected_event_types = {
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        }

        with client.messages.stream(
            model=model,
            max_tokens=100,
            messages=[{"role": "user", "content": "Count from 1 to 3."}],
        ) as stream:
            event_types = set()
            for event in stream:
                event_types.add(event.type)

        missing = expected_event_types - event_types
        assert not missing, f"Missing expected event types: {missing}"

    def test_streaming_collects_full_message(self, setup_backend):
        """Test that streaming deltas concatenate to a non-empty message."""
        _, model, client, gateway = setup_backend

        with client.messages.stream(
            model=model,
            max_tokens=100,
            messages=[{"role": "user", "content": "Say hello in one sentence."}],
        ) as stream:
            full_text = stream.get_final_text()

        assert len(full_text) > 0
