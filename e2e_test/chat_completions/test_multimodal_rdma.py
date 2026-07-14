"""vLLM RDMA multimodal transport E2E test.

Exercises the full RDMA (NIXL) pixel lane end to end: the gateway stages
``pixel_values`` into a pre-registered arena and the vLLM worker pulls them with a
one-sided READ instead of receiving them inline. Verifies both that the model
still understands the image over RDMA AND that the pixels actually travelled the
remote path (``smg_mm_tensor_bytes_total{path="remote"}`` grew), so a silent
fall-back to inline fails the test rather than passing on correctness alone.

Requires a gateway built with ``--features mm-rdma`` and a NIXL/UCX-capable host.

Usage:
    pytest e2e_test/chat_completions/test_multimodal_rdma.py -v
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

import httpx
import pytest

logger = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "images"
DOG_IMAGE_PATH = FIXTURES_DIR / "dog.jpg"  # Black labrador puppy


def _image_to_base64_url(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{data}"


def _remote_pixel_bytes(metrics_url: str) -> float:
    """Sum ``smg_mm_tensor_bytes_total`` samples on the RDMA (remote) path."""
    text = httpx.get(f"{metrics_url}/metrics", timeout=10).text
    total = 0.0
    for line in text.splitlines():
        if line.startswith("smg_mm_tensor_bytes_total") and 'path="remote"' in line:
            total += float(line.rsplit(" ", 1)[1])
    return total


@pytest.fixture(scope="class", autouse=True)
def _rdma_env():
    """Turn the RDMA lane on for both the gateway and the vLLM worker, which
    inherit this process's environment when launched locally. Restored on teardown.

    - ``SMG_MM_TENSOR_TRANSPORT=rdma``: the first-class transport switch (also set
      as a gateway CLI flag) that enables the worker-side puller.
    - ``SMG_MM_PIXEL_RDMA=1``: the legacy puller switch, belt-and-suspenders.
    - ``SMG_RDMA_LISTEN_IP=127.0.0.1``: the gateway's NIXL listener (loopback).
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("SMG_MM_TENSOR_TRANSPORT", "rdma")
        mp.setenv("SMG_MM_PIXEL_RDMA", "1")
        mp.setenv("SMG_RDMA_LISTEN_IP", "127.0.0.1")
        yield


@pytest.mark.engine("vllm")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.model("Qwen/Qwen3-VL-8B-Instruct")
@pytest.mark.gateway(extra_args=["--multimodal-tensor-transport", "rdma"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestMultimodalRdmaQwen3VL:
    """vLLM multimodal over the RDMA pixel lane via gRPC."""

    def test_single_image_uses_rdma(self, model, setup_backend):
        _, _, client, gateway = setup_backend

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
        assert any(k in text.lower() for k in ["dog", "puppy", "labrador"]), (
            f"Expected dog-related content over the RDMA transport, got: {text}"
        )
        logger.info("RDMA multimodal response: %s", text)

        # Critical: the request must have actually used the remote (RDMA) path.
        # A silent inline fall-back would still answer correctly, so correctness
        # alone does not prove the lane worked — the transport metric does.
        remote_bytes = _remote_pixel_bytes(gateway.metrics_url)
        assert remote_bytes > 0, (
            'expected smg_mm_tensor_bytes_total{path="remote"} > 0; the pixels '
            "silently fell back to inline (RDMA lane not exercised)"
        )
        logger.info("RDMA remote pixel bytes: %s", remote_bytes)
