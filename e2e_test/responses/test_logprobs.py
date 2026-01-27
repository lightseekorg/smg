"""Logprobs E2E Tests for Responses API.

Tests for logprobs support with GPT-OSS (Harmony) models through the Responses API.

The Responses API requires:
1. top_logprobs parameter in request
2. include field containing "message.output_text.logprobs"
"""

from __future__ import annotations

import logging

import pytest

logger = logging.getLogger(__name__)


# =============================================================================
# Responses API Logprobs Tests (GPT-OSS / Harmony)
# =============================================================================


@pytest.mark.e2e
@pytest.mark.model("gpt-oss")
@pytest.mark.gateway(
    extra_args=["--reasoning-parser=gpt-oss", "--history-backend", "memory"]
)
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestLogprobsGptOss:
    """Tests for logprobs support with GPT-OSS (Harmony) models via Responses API."""

    def test_responses_with_logprobs(self, setup_backend):
        """Test non-streaming Responses API returns logprobs when requested."""
        _, model, client, gateway = setup_backend

        response = client.responses.create(
            model=model,
            input="What is the capital of France? Answer in a few words.",
            max_output_tokens=50,
            top_logprobs=5,
            include=["message.output_text.logprobs"],
        )

        # Verify response structure
        assert response.id is not None
        assert response.status == "completed"
        assert response.error is None

        # Find message output items
        message_items = [item for item in response.output if item.type == "message"]
        assert len(message_items) > 0, "Should have at least one message output"

        message_item = message_items[0]
        assert message_item.content is not None
        assert len(message_item.content) > 0

        # Find OutputText content part with logprobs
        output_text_parts = [
            part for part in message_item.content if part.type == "output_text"
        ]
        assert len(output_text_parts) > 0, "Should have OutputText content"

        output_text = output_text_parts[0]
        assert output_text.text is not None
        assert isinstance(output_text.text, str)

        # Verify logprobs are returned
        logprobs = output_text.logprobs
        assert logprobs is not None, "logprobs should be present in OutputText"

        # Access as dict
        logprobs_content = logprobs["content"]
        assert logprobs_content is not None, "logprobs.content should be present"
        assert len(logprobs_content) > 0, "logprobs.content should not be empty"

        # Verify logprobs structure
        first_logprob = logprobs_content[0]
        assert isinstance(first_logprob["token"], str), "token should be a string"
        assert isinstance(first_logprob["logprob"], float), "logprob should be a float"
        assert first_logprob.get("bytes") is not None, "bytes should be present"

        # Verify top_logprobs count
        top_logprobs = first_logprob.get("top_logprobs")
        assert top_logprobs is not None, "top_logprobs should be present"
        assert len(top_logprobs) == 5, "should have 5 top_logprobs"

        for top_logprob in top_logprobs:
            assert isinstance(top_logprob["token"], str), "top_logprob token should be string"
            assert isinstance(top_logprob["logprob"], float), "top_logprob logprob should be float"
            assert top_logprob.get("bytes") is not None, "top_logprob bytes should be present"

        # Verify usage
        assert response.usage is not None
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0
        assert response.usage.total_tokens > 0

    def test_responses_without_logprobs(self, setup_backend):
        """Test non-streaming Responses API without logprobs (default behavior)."""
        _, model, client, gateway = setup_backend

        response = client.responses.create(
            model=model,
            input="What is the capital of France? Answer in a few words.",
            max_output_tokens=50,
        )

        # Verify response structure
        assert response.id is not None
        assert response.status == "completed"

        # Find message output items
        message_items = [item for item in response.output if item.type == "message"]
        assert len(message_items) > 0

        message_item = message_items[0]
        output_text_parts = [
            part for part in message_item.content if part.type == "output_text"
        ]
        assert len(output_text_parts) > 0

        # Verify logprobs are NOT returned when not requested
        output_text = output_text_parts[0]
        assert (
            output_text.logprobs is None
        ), "logprobs should be None when not requested"

    def test_responses_stream_with_logprobs(self, setup_backend):
        """Test streaming Responses API returns logprobs in events."""
        _, model, client, gateway = setup_backend

        generator = client.responses.create(
            model=model,
            input="What is the capital of France?",
            max_output_tokens=50,
            top_logprobs=3,
            include=["message.output_text.logprobs"],
            stream=True,
        )

        events = list(generator)
        assert len(events) > 0

        # Find response.completed event
        completed_events = [
            event for event in events if event.type == "response.completed"
        ]
        assert len(completed_events) == 1, "Should have exactly one completed event"

        # Get output from completed event and verify logprobs
        completed_event = completed_events[0]
        output_array = completed_event.response.output
        assert output_array is not None
        assert len(output_array) > 0

        # Find message item in output
        message_items = [item for item in output_array if item.type == "message"]
        assert len(message_items) > 0

        message_item = message_items[0]
        output_text_parts = [
            part for part in message_item.content if part.type == "output_text"
        ]
        assert len(output_text_parts) > 0, "Should have output_text content"

        # Verify logprobs in final response
        output_text = output_text_parts[0]
        logprobs = output_text.logprobs
        assert logprobs is not None, "logprobs should be present in streaming response"

        # Access as dict
        logprobs_content = logprobs["content"]
        assert logprobs_content is not None
        assert len(logprobs_content) > 0

        # Verify logprobs structure
        first_logprob = logprobs_content[0]
        assert isinstance(first_logprob["token"], str)
        assert isinstance(first_logprob["logprob"], float)

        top_logprobs = first_logprob.get("top_logprobs")
        assert top_logprobs is not None
        assert len(top_logprobs) == 3
