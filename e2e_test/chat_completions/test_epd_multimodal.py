"""EPD (Encode-Prefill-Decode) multimodal Chat Completions E2E Tests.

Exercises TokenSpeed's EPD disaggregation on a vision-language model: the
encode worker runs the vision tower, prefill/decode run the LM, and the
gateway stitches encode -> prefill -> decode. The point of these tests is
that a disaggregated encode->prefill->decode path still produces a correct
multimodal answer.

EPD is TokenSpeed-only. On a small MoE VLM (Qwen3.6-35B-A3B, 3B active) at
tp=1 per worker, every topology fits the 4-GPU runner: 1e1p1d=3 GPUs and
1e2p1d/2e1p1d/1e1p2d=4 GPUs (EPD needs >=3 cards since encode/prefill/decode
are separate workers).

Usage:
    pytest e2e_test/chat_completions/test_epd_multimodal.py -v
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)

# Local test image (checked into repo) — a black labrador puppy.
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "images"
DOG_IMAGE_PATH = FIXTURES_DIR / "dog.jpg"


def _image_to_base64_url(path: Path) -> str:
    """Convert a local image file to a base64 data URL."""
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{data}"


def _make_image_content(image_source: str) -> dict:
    """Create an image_url content part from either a URL or local path."""
    return {"type": "image_url", "image_url": {"url": image_source}}


# The four EPD topologies to cover. Every worker runs at tp=1 (a 3B-active MoE
# that fits one card), so 1e1p1d uses 3 GPUs and the rest use 4 — all fit the
# 4-GPU runner. Each carries an ``epd`` marker consumed by setup_backend's EPD path.
_EPD_TOPOLOGIES = [
    pytest.param("epd_grpc", marks=pytest.mark.epd(encode=1, prefill=1, decode=1), id="1e1p1d"),
    pytest.param("epd_grpc", marks=pytest.mark.epd(encode=1, prefill=2, decode=1), id="1e2p1d"),
    pytest.param("epd_grpc", marks=pytest.mark.epd(encode=2, prefill=1, decode=1), id="2e1p1d"),
    pytest.param("epd_grpc", marks=pytest.mark.epd(encode=1, prefill=1, decode=2), id="1e1p2d"),
]


@pytest.mark.engine("tokenspeed")
@pytest.mark.gpu(4)
@pytest.mark.e2e
@pytest.mark.model("Qwen/Qwen3.6-35B-A3B-FP8")
@pytest.mark.parametrize("setup_backend", _EPD_TOPOLOGIES, indirect=True)
class TestEPDMultimodal:
    """Multimodal tests over TokenSpeed EPD disaggregation (4 GPU)."""

    def test_single_image_base64(self, model, setup_backend):
        """A single dog image travels encode -> prefill -> decode and is described."""
        _, _, client, *_ = setup_backend

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What animal is in this image?"},
                        _make_image_content(_image_to_base64_url(DOG_IMAGE_PATH)),
                    ],
                }
            ],
            temperature=0,
            max_tokens=100,
        )

        text = response.choices[0].message.content
        assert text is not None and len(text) > 0
        assert any(k in text.lower() for k in ["dog", "puppy", "labrador"]), (
            f"Expected dog-related content, got: {text}"
        )
        assert response.usage.prompt_tokens > 0
        logger.info("EPD single image base64: %s", text)
