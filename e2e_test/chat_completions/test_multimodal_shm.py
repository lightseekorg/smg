"""Multimodal SHM tensor-transport E2E test (actual model).

Launches the gateway with ``--multimodal-tensor-transport shm`` and runs a real
vision-language model (Qwen3-VL) over vLLM gRPC, then asserts both:

  1. the model still understands the image (SHM produces byte-identical tensors
     to the inline path, so the answer must be correct), and
  2. the SHM transport was *actually* exercised — scraped from the gateway's
     ``smg_mm_tensors_total{path="shm",runtime="vllm"}`` counter — so a silent
     fallback to inline (e.g. /dev/shm not writable) fails the test instead of
     passing on the inline path.

Inline transport for the same model/image is already covered by
``test_multimodal.py``; together they show the two paths are equivalent.

Usage:
    pytest e2e_test/chat_completions/test_multimodal_shm.py -v
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

import pytest
import requests

logger = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
DOG_IMAGE_PATH = FIXTURES_DIR / "images" / "dog.jpg"  # Black labrador puppy
DOG_VIDEO_PATH = FIXTURES_DIR / "videos" / "dog.mp4"  # Short clip of the same dog


def _image_to_base64_url(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{data}"


def _video_to_base64_url(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:video/mp4;base64,{data}"


def _shm_tensor_count(metrics_url: str, runtime: str = "vllm") -> float:
    """Sum the ``smg_mm_tensors_total`` counter for SHM-transported tensors."""
    resp = requests.get(f"{metrics_url}/metrics", timeout=10)
    resp.raise_for_status()
    total = 0.0
    for line in resp.text.splitlines():
        if not line.startswith("smg_mm_tensors_total"):
            continue
        if 'path="shm"' in line and f'runtime="{runtime}"' in line:
            total += float(line.rsplit(None, 1)[-1])
    return total


@pytest.mark.engine("vllm")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.model("Qwen/Qwen3-VL-8B-Instruct")
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
@pytest.mark.gateway(extra_args=["--multimodal-tensor-transport", "shm"])
class TestMultimodalVllmShmTransport:
    """Qwen3-VL over vLLM gRPC with the SHM multimodal tensor transport forced on."""

    def test_single_image_uses_shm_transport(self, model, setup_backend):
        _, _, client, gateway = setup_backend

        before = _shm_tensor_count(gateway.metrics_url)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What animal is in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": _image_to_base64_url(DOG_IMAGE_PATH)},
                        },
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
        assert response.usage.completion_tokens > 0

        after = _shm_tensor_count(gateway.metrics_url)
        assert after > before, (
            "expected the vLLM SHM tensor transport to be exercised "
            f"(smg_mm_tensors_total path=shm runtime=vllm did not increase: {before} -> {after})"
        )
        logger.info("SHM image response: %s (shm tensors %s -> %s)", text, before, after)

    def test_single_video_uses_shm_transport(self, model, setup_backend):
        """Video is the large-tensor case: pixel_values_videos is many frames of
        patches, so it decisively exercises the SHM transport for vLLM video."""
        _, _, client, gateway = setup_backend

        before = _shm_tensor_count(gateway.metrics_url)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What animal is in this video?"},
                        {
                            "type": "video_url",
                            "video_url": {"url": _video_to_base64_url(DOG_VIDEO_PATH)},
                        },
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
        assert response.usage.completion_tokens > 0

        after = _shm_tensor_count(gateway.metrics_url)
        assert after > before, (
            "expected the vLLM SHM tensor transport to be exercised for video "
            f"(smg_mm_tensors_total path=shm runtime=vllm did not increase: {before} -> {after})"
        )
        logger.info("SHM video response: %s (shm tensors %s -> %s)", text, before, after)
