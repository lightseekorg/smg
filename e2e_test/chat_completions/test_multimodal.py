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
# Qwen3-VL multimodal tests (1 GPU)
# =============================================================================


@pytest.mark.engine("vllm")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.model("Qwen/Qwen3-VL-8B-Instruct")
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestMultimodalQwen3VL:
    """Multimodal tests using Qwen3-VL via gRPC."""

    @pytest.mark.parametrize("stream", [False, True], ids=["non_streaming", "streaming"])
    def test_single_image(self, model, setup_backend, stream):
        """Test single image understanding with and without streaming."""
        _, _, client, *_ = setup_backend

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What animal is in this image?"},
                        {"type": "image_url", "image_url": {"url": IMAGE_DOG_URL}},
                    ],
                }
            ],
            temperature=0,
            max_tokens=100,
            stream=stream,
        )

        if stream:
            chunks = [
                chunk.choices[0].delta.content
                for chunk in response
                if chunk.choices and chunk.choices[0].delta.content
            ]
            text = "".join(chunks)
            assert text, "Streaming should produce content"
            assert len(chunks) > 1, "Streaming should produce multiple chunks"
        else:
            text = response.choices[0].message.content
            assert text is not None and len(text) > 0
            assert response.usage.prompt_tokens > 0
            assert response.usage.completion_tokens > 0

        assert any(k in text.lower() for k in ["dog", "puppy", "labrador"]), (
            f"Expected dog-related content, got: {text}"
        )
        logger.info("Single image response (stream=%s): %s", stream, text)

    def test_multi_images(self, model, setup_backend):
        """Test multiple image understanding with duplicate detection."""
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
            max_tokens=300,
        )

        text = response.choices[0].message.content
        assert text is not None and len(text) > 0
        text_lower = text.lower()

        assert any(k in text_lower for k in ["dog", "pug", "puppy"]), (
            f"Expected dog-related content, got: {text}"
        )
        assert any(
            k in text_lower for k in ["same", "identical", "duplicate", "second", "third"]
        ), f"Expected model to notice duplicate images, got: {text}"
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        logger.info("Multi image response: %s", text)


# =============================================================================
# Llama-4-Scout multimodal tests (4 GPU)
# =============================================================================


@pytest.mark.engine("vllm")
@pytest.mark.gpu(4)
@pytest.mark.e2e
@pytest.mark.model("meta-llama/Llama-4-Scout-17B-16E-Instruct")
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestMultimodalLlama4Scout:
    """Multimodal tests using Llama-4-Scout via gRPC."""

    @pytest.mark.parametrize("stream", [False, True], ids=["non_streaming", "streaming"])
    def test_single_image(self, model, setup_backend, stream):
        """Test single image understanding with and without streaming."""
        _, _, client, *_ = setup_backend

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What animal is in this image?"},
                        {"type": "image_url", "image_url": {"url": IMAGE_DOG_URL}},
                    ],
                }
            ],
            temperature=0,
            max_tokens=100,
            stream=stream,
        )

        if stream:
            chunks = [
                chunk.choices[0].delta.content
                for chunk in response
                if chunk.choices and chunk.choices[0].delta.content
            ]
            text = "".join(chunks)
            assert text, "Streaming should produce content"
            assert len(chunks) > 1, "Streaming should produce multiple chunks"
        else:
            text = response.choices[0].message.content
            assert text is not None and len(text) > 0
            assert response.usage.prompt_tokens > 0
            assert response.usage.completion_tokens > 0

        assert any(k in text.lower() for k in ["dog", "puppy", "labrador"]), (
            f"Expected dog-related content, got: {text}"
        )
        logger.info("Single image response (stream=%s): %s", stream, text)

    def test_multi_images(self, model, setup_backend):
        """Test multiple image understanding with duplicate detection."""
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
            max_tokens=300,
        )

        text = response.choices[0].message.content
        assert text is not None and len(text) > 0
        text_lower = text.lower()

        assert any(k in text_lower for k in ["dog", "pug", "puppy"]), (
            f"Expected dog-related content, got: {text}"
        )
        assert any(
            k in text_lower for k in ["same", "identical", "duplicate", "second", "third"]
        ), f"Expected model to notice duplicate images, got: {text}"
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        logger.info("Multi image response: %s", text)
