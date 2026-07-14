"""EPD (Encode-Prefill-Decode) multimodal Chat Completions E2E tests.

Exercises TokenSpeed's EPD disaggregation on a vision-language model: the
encode worker runs the vision tower, prefill/decode run the LM, and the gateway
stitches encode -> prefill -> decode across four worker-count topologies.

Like the PD KV-transfer tests, these do NOT stop at "a plausible answer came
back" — a single-worker fallback would pass that. They assert the encode
worker's own per-request accept log, proving the image really flowed through a
dedicated encode worker.

Usage:
    pytest e2e_test/chat_completions/test_epd_multimodal.py -v
"""

from __future__ import annotations

import base64
import logging
import os
import tempfile
import time
from pathlib import Path

import pytest
from infra.pd_logs import (
    LOG_FLUSH_TIMEOUT_S,
    assert_worker_logs_captured,
    read_logs,
    worker_log_dir,
)

logger = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "images"
DOG_IMAGE_PATH = FIXTURES_DIR / "dog.jpg"  # Black labrador puppy (checked in)

_LOG_DIR = Path(tempfile.gettempdir()) / f"smg-e2e-epd-{os.getpid()}"
# Emitted once per accepted Encode RPC by the TokenSpeed encode servicer
# (grpc_servicer/.../tokenspeed/encoder_servicer.py). Its presence proves the
# image reached a dedicated encode worker — the defining EPD step.
ENCODE_ACCEPTED_MARKER = "EPD encode: accepted"

# (encode, prefill, decode) worker counts. Every worker is tp=1, so 1e1p1d uses
# 3 GPUs and the rest use 4 — all fit the 4-GPU runner. Counts ride in the param
# because setup_backend is class-scoped and can't read per-param marks.
_EPD_TOPOLOGIES = [
    pytest.param(("epd_grpc", (1, 1, 1)), id="1e1p1d"),
    pytest.param(("epd_grpc", (1, 2, 1)), id="1e2p1d"),
    pytest.param(("epd_grpc", (2, 1, 1)), id="2e1p1d"),
    pytest.param(("epd_grpc", (1, 1, 2)), id="1e1p2d"),
]


def _image_to_base64_url(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{data}"


@pytest.mark.engine("tokenspeed")
@pytest.mark.gpu(4)
@pytest.mark.e2e
@pytest.mark.model("Qwen/Qwen3.5-9B")
@pytest.mark.gateway(log_level="debug", policy="cache_aware", log_dir=str(_LOG_DIR))
@pytest.mark.parametrize("setup_backend", _EPD_TOPOLOGIES, indirect=True)
class TestEPDMultimodal:
    """Verify the image really flows encode -> prefill -> decode for each topology."""

    def test_single_image(self, model, setup_backend):
        _, _, client, *_ = setup_backend

        # Baseline BEFORE the request: worker logs accumulate across the four
        # parametrized topologies in one shared dir and are never cleared, so a
        # plain "marker present" check would pass on a stale marker from an
        # earlier topology. Assert THIS request produced a NEW acceptance.
        worker_dir = worker_log_dir(_LOG_DIR)
        markers_before = read_logs(worker_dir, "worker-*.log").count(ENCODE_ACCEPTED_MARKER)

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

        # (1) The model saw the image and described it correctly.
        text = response.choices[0].message.content
        assert text is not None and len(text) > 0
        assert any(k in text.lower() for k in ["dog", "puppy", "labrador"]), (
            f"Expected dog-related content, got: {text}"
        )
        # (2) The image tokens were spliced into the prompt: the bare question is
        # ~10 tokens, so a large prompt confirms vision tokens reached prefill.
        assert response.usage.prompt_tokens > 50, (
            f"prompt_tokens={response.usage.prompt_tokens} too low; "
            "vision tokens likely not delivered to prefill"
        )
        assert response.usage.completion_tokens > 0

        # (3) REAL EPD: a NEW encode acceptance appeared for THIS request, proving
        # the image ran on a dedicated encode worker (not a single-worker fallback)
        # for THIS topology — robust to stale markers from earlier topologies.
        deadline = time.monotonic() + LOG_FLUSH_TIMEOUT_S
        worker_logs = read_logs(worker_dir, "worker-*.log")
        while (
            worker_logs.count(ENCODE_ACCEPTED_MARKER) <= markers_before
            and time.monotonic() < deadline
        ):
            time.sleep(0.5)
            worker_logs = read_logs(worker_dir, "worker-*.log")
        assert_worker_logs_captured(worker_logs, "EPD encode dispatch")
        assert worker_logs.count(ENCODE_ACCEPTED_MARKER) > markers_before, (
            "encode worker logged no NEW acceptance for this request — the image did "
            "not flow through a dedicated EPD encode worker for this topology; checked "
            f"{worker_dir}/worker-*.log (baseline count={markers_before})"
        )
        logger.info("EPD image OK (new encode acceptance observed): %s", text)
