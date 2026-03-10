"""Basic tests for Anthropic Messages API.

Tests for non-streaming and streaming message creation, system prompts,
and multi-turn conversations via the Anthropic SDK.
"""

from __future__ import annotations

import logging

import pytest
from conftest import smg_compare

logger = logging.getLogger(__name__)


@pytest.mark.vendor("anthropic")
@pytest.mark.gpu(0)
@pytest.mark.parametrize("setup_backend", ["anthropic"], indirect=True)
class TestMessagesBasic:
    """Basic message creation tests against the Anthropic Messages API."""

    def test_non_streaming_basic(self, setup_backend, smg):
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

        # SmgClient comparison
        with smg_compare():
            smg_resp = smg.messages.create(
                model=model,
                max_tokens=100,
                messages=[{"role": "user", "content": "Say hello in one sentence."}],
            )
            assert smg_resp.id is not None
            assert smg_resp.role == "assistant"
            assert smg_resp.stop_reason == "end_turn"
            assert len(smg_resp.content) > 0
            assert smg_resp.content[0].text is not None
            assert len(smg_resp.content[0].text) > 0
            assert smg_resp.usage.input_tokens > 0
            assert smg_resp.usage.output_tokens > 0

    def test_non_streaming_with_system(self, setup_backend, smg):
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

        # SmgClient comparison
        with smg_compare():
            smg_resp = smg.messages.create(
                model=model,
                max_tokens=50,
                system="You are a helpful assistant. Always respond in exactly one word.",
                messages=[{"role": "user", "content": "What color is the sky?"}],
            )
            assert smg_resp.id is not None
            assert smg_resp.stop_reason == "end_turn"
            assert len(smg_resp.content) > 0

    def test_non_streaming_multi_turn(self, setup_backend, smg):
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

        # SmgClient comparison
        with smg_compare():
            smg_resp = smg.messages.create(
                model=model,
                max_tokens=100,
                messages=[
                    {"role": "user", "content": "My name is Alice."},
                    {"role": "assistant", "content": "Nice to meet you, Alice!"},
                    {"role": "user", "content": "What is my name?"},
                ],
            )
            assert smg_resp.stop_reason == "end_turn"
            assert "alice" in smg_resp.content[0].text.lower()

    def test_streaming_basic(self, setup_backend, smg):
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

        # SmgClient streaming comparison
        with smg_compare():
            smg_event_types = set()
            with smg.messages.stream(
                model=model,
                max_tokens=100,
                messages=[{"role": "user", "content": "Count from 1 to 3."}],
            ) as stream:
                for event in stream:
                    smg_event_types.add(event.type)
            # SmgClient streams should have same event types
            smg_missing = expected_event_types - smg_event_types
            assert not smg_missing, f"SmgClient missing event types: {smg_missing}"

    def test_streaming_collects_full_message(self, setup_backend, smg):
        """Test that streaming deltas concatenate to a non-empty message."""
        _, model, client, gateway = setup_backend

        with client.messages.stream(
            model=model,
            max_tokens=100,
            messages=[{"role": "user", "content": "Say hello in one sentence."}],
        ) as stream:
            full_text = stream.get_final_text()

        assert len(full_text) > 0

        # SmgClient streaming comparison
        with smg_compare():
            smg_text_pieces = []
            with smg.messages.stream(
                model=model,
                max_tokens=100,
                messages=[{"role": "user", "content": "Say hello in one sentence."}],
            ) as stream:
                for event in stream:
                    if event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            smg_text_pieces.append(event.delta.text)
            smg_full_text = "".join(smg_text_pieces)
            assert len(smg_full_text) > 0, "SmgClient stream should produce text"
