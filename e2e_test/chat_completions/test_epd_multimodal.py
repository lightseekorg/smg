"""EPD (Encode-Prefill-Decode) multimodal Chat Completions E2E Tests.

Exercises TokenSpeed's EPD disaggregation on a vision-language model: the
encode worker runs the vision tower, prefill/decode run the LM, and the gateway
stitches encode -> prefill -> decode.

Like the PD KV-transfer tests (``test_pd_mooncake``/``test_pd_nixl``), these do
NOT stop at "a plausible answer came back" — a single-worker fallback would pass
that. They assert a worker-side signal that the disaggregation actually happened:
the encode worker's own per-request accept log.

EPD is TokenSpeed-only. On a small MoE VLM (Qwen3.6-35B-A3B, 3B active) at tp=1
per worker, every topology fits the 4-GPU runner: 1e1p1d=3 GPUs and
1e2p1d/2e1p1d/1e1p2d=4 GPUs (EPD needs >=3 cards since encode/prefill/decode are
separate workers).

Usage:
    pytest e2e_test/chat_completions/test_epd_multimodal.py -v
"""

from __future__ import annotations

import base64
import logging
import os
import tempfile
from pathlib import Path

import pytest
from infra.pd_logs import assert_worker_logs_captured, wait_for_marker, worker_log_dir

logger = logging.getLogger(__name__)

# Local test image (checked into repo) — a black labrador puppy.
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "images"
DOG_IMAGE_PATH = FIXTURES_DIR / "dog.jpg"

# Router + worker logs land here (per-pid) via the gateway marker below. Worker
# logs actually go to E2E_LOG_DIR in CI; ``worker_log_dir`` resolves both.
_LOG_DIR = Path(tempfile.gettempdir()) / f"smg-e2e-epd-{os.getpid()}"
# Emitted once per Encode RPC by the TokenSpeed encode servicer
# (grpc_servicer/.../tokenspeed/encoder_servicer.py). Its presence proves the
# image reached a dedicated encode worker — the defining EPD step.
ENCODE_ACCEPTED_MARKER = "EPD encode: accepted"


def _image_to_base64_url(path: Path) -> str:
    """Convert a local image file to a base64 data URL."""
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{data}"


def _make_image_content(image_source: str) -> dict:
    """Create an image_url content part from a URL or data URL string.

    Local file paths must be pre-converted via ``_image_to_base64_url``.
    """
    return {"type": "image_url", "image_url": {"url": image_source}}


# The four EPD topologies to cover. Every worker runs at tp=1 (a 3B-active MoE
# that fits one card), so 1e1p1d uses 3 GPUs and the rest use 4 — all fit the
# 4-GPU runner. The (encode, prefill, decode) counts ride in the param tuple, not
# a marker: setup_backend is class-scoped, so per-param marks aren't visible there.
_EPD_TOPOLOGIES = [
    pytest.param(("epd_grpc", (1, 1, 1)), id="1e1p1d"),
    pytest.param(("epd_grpc", (1, 2, 1)), id="1e2p1d"),
    pytest.param(("epd_grpc", (2, 1, 1)), id="2e1p1d"),
    pytest.param(("epd_grpc", (1, 1, 2)), id="1e1p2d"),
]


@pytest.mark.engine("tokenspeed")
@pytest.mark.gpu(4)
@pytest.mark.e2e
@pytest.mark.model("Qwen/Qwen3.6-35B-A3B-FP8")
@pytest.mark.gateway(log_level="debug", log_dir=str(_LOG_DIR))
@pytest.mark.parametrize("setup_backend", _EPD_TOPOLOGIES, indirect=True)
class TestEPDMultimodal:
    """Verify the image really flows encode -> prefill -> decode.

    A naive content check can't distinguish a real 3-worker EPD pipeline from a
    single-worker fallback; the encode worker's own log can — so that's what this
    asserts, mirroring how the PD tests assert the KV transfer from logs.
    """

    def test_single_image_base64(self, model, setup_backend):
        """One dog image through encode -> prefill -> decode, with the encode
        worker's participation verified from its logs."""
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

        # (1) The model saw the image and described it correctly.
        text = response.choices[0].message.content
        assert text is not None and len(text) > 0
        assert any(k in text.lower() for k in ["dog", "puppy", "labrador"]), (
            f"Expected dog-related content, got: {text}"
        )
        # (2) The image was tokenized INTO the prompt: the bare text question is
        # ~10 tokens, so a large prompt confirms the vision tokens were spliced in
        # (encode -> prefill actually delivered the image), not dropped.
        assert response.usage.prompt_tokens > 50, (
            f"prompt_tokens={response.usage.prompt_tokens} is too low; "
            "the image tokens were likely not delivered to prefill"
        )
        assert response.usage.completion_tokens > 0

        # (3) REAL EPD: the encode worker itself logged accepting the dispatch, so
        # the vision stage ran on a separate encode worker rather than degrading to
        # a single-worker path. This is the EPD analog of the PD KV-transfer check.
        worker_dir = worker_log_dir(_LOG_DIR)
        worker_logs = wait_for_marker(worker_dir, "worker-*.log", ENCODE_ACCEPTED_MARKER)
        assert_worker_logs_captured(worker_logs, "EPD encode dispatch")
        assert ENCODE_ACCEPTED_MARKER in worker_logs, (
            "encode worker never logged accepting the request — the image did not "
            f"flow through the EPD encode stage; checked {worker_dir}/worker-*.log"
        )
        logger.info("EPD single image (encode worker engaged): %s", text)
