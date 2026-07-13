"""Audio transcription E2E tests — TokenSpeed Qwen3-ASR.

Exercises the ``POST /v1/audio/transcriptions`` endpoint end-to-end against a
locally-hosted **TokenSpeed** gRPC worker serving ``Qwen/Qwen3-ASR-1.7B``.

This is the only serving path for audio transcription in SMG: the gRPC router's
transcription adapter (``model_gateway/src/routers/grpc/router.rs``) is Qwen3-ASR
only, and audio multimodal inputs are accepted only on TokenSpeed workers
(SGLang / vLLM / TRT-LLM reject audio batches in
``model_gateway/src/routers/grpc/multimodal/assemble.rs``). The realtime
WebSocket ASR path (``e2e_test/realtime/test_realtime_local.py``) is a separate
vLLM-only serving path and does not cover this REST endpoint.

The model is wired through the standard ``setup_backend`` gRPC fixture, so the
test runs in the ``e2e-1gpu-chat (tokenspeed)`` CI lane — the only lane with
TokenSpeed installed — and needs no bespoke gateway plumbing. Requests are sent
with raw ``httpx`` multipart (rather than the OpenAI SDK's transcription
helper), matching the gateway's ``multipart/form-data`` contract exactly and
keeping assertions independent of the SDK's streaming/response-format overloads.

Prerequisites:
- 1 GPU able to serve Qwen3-ASR under TokenSpeed.
- ``E2E_RUNTIME=tokenspeed`` (set by the CI lane); the ``engine`` marker keeps
  this test out of the sglang/vllm/trtllm lanes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx
import pytest

logger = logging.getLogger(__name__)

MODEL = "Qwen/Qwen3-ASR-1.7B"

# Reuse the committed 8s / 16 kHz mono PCM16 clip of "Mary had a little lamb"
# already used by the realtime ASR e2e (no new binary fixture needed).
AUDIO_WAV = Path(__file__).resolve().parents[1] / "realtime" / "fixtures" / "mary_had_lamb_16k.wav"

# Generous: TokenSpeed model warmup + whole-file ASR decode.
REQUEST_TIMEOUT = 120.0


def _audio_part() -> tuple[str, bytes, str]:
    """multipart ``file`` part: (filename, bytes, content-type)."""
    return (AUDIO_WAV.name, AUDIO_WAV.read_bytes(), "audio/wav")


def _post_transcription(base_url: str, data: dict[str, str]) -> httpx.Response:
    """POST the fixture clip to /v1/audio/transcriptions with form fields ``data``."""
    return httpx.post(
        f"{base_url}/v1/audio/transcriptions",
        files={"file": _audio_part()},
        data={"model": MODEL, **data},
        timeout=REQUEST_TIMEOUT,
    )


@pytest.fixture()
def gateway(setup_backend):
    """The Gateway launched by ``setup_backend`` (its 4th tuple element)."""
    _, _, _, gw = setup_backend
    return gw


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.gpu(1)
@pytest.mark.engine("tokenspeed")
@pytest.mark.model(MODEL)
@pytest.mark.gateway(extra_args=["--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestTokenSpeedTranscription:
    """``/v1/audio/transcriptions`` against a TokenSpeed Qwen3-ASR worker."""

    def test_transcription_round_trip(self, gateway):
        """A whole-file transcription returns HTTP 200 with non-empty ``text``.

        The exact wording is left to the ASR model (it varies by model
        revision / build), so the transcription is logged and only asserted to
        be non-empty — matching the realtime e2e's robustness stance. When the
        model behaves as expected the clip's words show up, so a best-effort
        substring check is logged without failing the test.
        """
        resp = _post_transcription(
            gateway.base_url,
            {"language": "en", "response_format": "json", "temperature": "0"},
        )
        assert resp.status_code == 200, resp.text

        text = resp.json()["text"]
        logger.info("transcription (json): %s", text)
        assert isinstance(text, str)
        assert text.strip(), "expected a non-empty transcription"
        if not any(word in text.lower() for word in ("mary", "lamb", "little")):
            logger.warning("transcription did not contain expected words: %r", text)

    def test_transcription_response_format_text(self, gateway):
        """``response_format="text"`` yields a plain-text (non-empty) 200 body."""
        resp = _post_transcription(gateway.base_url, {"response_format": "text"})
        assert resp.status_code == 200, resp.text
        assert resp.headers["content-type"].startswith("text/plain")

        text = resp.text
        logger.info("transcription (text): %s", text)
        assert text.strip(), "expected a non-empty transcription"

    def test_streaming_transcription_rejected(self, gateway):
        """TokenSpeed Qwen3-ASR is whole-file only — streaming must be rejected."""
        resp = _post_transcription(gateway.base_url, {"stream": "true"})
        assert resp.status_code == 400, resp.text

    def test_unsupported_language_rejected(self, gateway):
        """An out-of-allow-list language hint is rejected with 400."""
        resp = _post_transcription(gateway.base_url, {"language": "zz"})
        assert resp.status_code == 400, resp.text
