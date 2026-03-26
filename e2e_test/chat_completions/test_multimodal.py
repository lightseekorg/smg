"""Multimodal Chat Completions E2E Tests.

Tests for vision-language models through the gateway, verifying that
image content is correctly processed and the model produces meaningful
responses about the images.

Usage:
    pytest e2e_test/chat_completions/test_multimodal.py -v
"""

from __future__ import annotations

import logging

import pytest

logger = logging.getLogger(__name__)

# Test image URLs (stable, public images)
IMAGE_DOG_URL = "https://picsum.photos/id/237/300/200"  # Black labrador puppy
IMAGE_PUG_URL = "https://picsum.photos/id/1025/300/200"  # Pug in blanket


# =============================================================================
# Qwen3-VL multimodal tests
# =============================================================================


@pytest.mark.engine("vllm")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.model("Qwen/Qwen3-VL-8B-Instruct")
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestMultimodalQwen3VL:
    """Multimodal tests using Qwen3-VL via gRPC."""

    def test_single_image(self, model, setup_backend):
        """Test single image understanding."""
        _, _, client, *_ = setup_backend

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What animal is in this image?",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": IMAGE_DOG_URL},
                        },
                    ],
                }
            ],
            temperature=0,
            max_tokens=50,
        )

        text = response.choices[0].message.content
        assert text is not None
        assert len(text) > 0
        text_lower = text.lower()
        assert "dog" in text_lower or "puppy" in text_lower or "labrador" in text_lower, (
            f"Expected 'dog' or 'puppy' in response, got: {text}"
        )
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        logger.info("Single image response: %s", text)

    def test_multi_images(self, model, setup_backend):
        """Test multiple image understanding."""
        _, _, client, *_ = setup_backend

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "How many images did I send? Describe each. Are any of them the same?",
                        },
                        {"type": "image_url", "image_url": {"url": IMAGE_DOG_URL}},
                        {"type": "image_url", "image_url": {"url": IMAGE_PUG_URL}},
                        {"type": "image_url", "image_url": {"url": IMAGE_PUG_URL}},
                    ],
                }
            ],
            temperature=0,
            max_tokens=200,
        )

        text = response.choices[0].message.content
        assert text is not None
        assert len(text) > 0
        text_lower = text.lower()
        # Should identify dogs in the images
        assert "dog" in text_lower or "pug" in text_lower or "puppy" in text_lower, (
            f"Expected dog-related content, got: {text}"
        )
        # Should notice images 2 and 3 are identical
        assert "same" in text_lower or "identical" in text_lower or "duplicate" in text_lower, (
            f"Expected model to notice duplicate images, got: {text}"
        )
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        logger.info("Multi image response: %s", text)

    def test_streaming_with_image(self, model, setup_backend):
        """Test streaming response with image input."""
        _, _, client, *_ = setup_backend

        stream = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What animal is in this image?",
                        },
                        {"type": "image_url", "image_url": {"url": IMAGE_DOG_URL}},
                    ],
                }
            ],
            temperature=0,
            max_tokens=100,
            stream=True,
        )

        chunks = []
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)

        full_text = "".join(chunks)
        assert len(full_text) > 0, "Streaming should produce content"
        text_lower = full_text.lower()
        assert "dog" in text_lower or "puppy" in text_lower or "labrador" in text_lower, (
            f"Expected 'dog' or 'puppy' in streaming response, got: {full_text}"
        )
        logger.info("Streaming image response: %s", full_text)
