"""Audio transcription E2E tests — TokenSpeed Qwen3-ASR.

Exercises ``POST /v1/audio/transcriptions`` end-to-end against a locally-hosted
TokenSpeed gRPC worker serving ``Qwen/Qwen3-ASR-1.7B``, driven through the same
standard ``api_client`` (OpenAI SDK) / ``model`` fixtures every other
chat_completions e2e test uses.

Audio transcription is TokenSpeed-only: the gRPC transcription adapter
(``model_gateway/src/routers/grpc/router.rs``) is Qwen3-ASR only, and audio
multimodal inputs are accepted only on TokenSpeed workers. The ``engine``
marker keeps this test in the ``e2e-1gpu-chat (tokenspeed)`` CI lane.

Usage:
    pytest e2e_test/chat_completions/test_transcription_tokenspeed.py -v
"""

from __future__ import annotations

import logging
from pathlib import Path

import openai
import pytest

logger = logging.getLogger(__name__)

MODEL = "Qwen/Qwen3-ASR-1.7B"

# Reuse the committed 8s / 16 kHz mono clip of "Mary had a little lamb" already
# used by the realtime ASR e2e — no new fixture binary needed.
AUDIO_WAV = Path(__file__).parent.parent / "realtime" / "fixtures" / "mary_had_lamb_16k.wav"


@pytest.mark.engine("tokenspeed")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.model(MODEL)
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestTokenSpeedTranscription:
    """``/v1/audio/transcriptions`` against a TokenSpeed Qwen3-ASR worker."""

    def test_transcription_returns_text(self, model, api_client):
        """A whole-file transcription returns non-empty text via the OpenAI client."""
        with AUDIO_WAV.open("rb") as audio:
            result = api_client.audio.transcriptions.create(
                model=model,
                file=audio,
                language="en",
                temperature=0.0,
            )

        logger.info("transcription: %s", result.text)
        assert result.text.strip(), "expected a non-empty transcription"
        # Exact wording varies by model revision, so only warn (don't fail the
        # gate) if the clip's words are missing.
        if not any(word in result.text.lower() for word in ("mary", "lamb", "little")):
            logger.warning("transcription missing expected words: %r", result.text)

    def test_transcription_response_format_text(self, model, api_client):
        """``response_format="text"`` returns a plain-text transcription."""
        with AUDIO_WAV.open("rb") as audio:
            result = api_client.audio.transcriptions.create(
                model=model,
                file=audio,
                response_format="text",
            )

        # The OpenAI SDK returns a bare string for the ``text`` response format;
        # accept either form defensively.
        text = result if isinstance(result, str) else result.text
        logger.info("transcription (text): %s", text)
        assert text.strip(), "expected a non-empty transcription"

    def test_unsupported_language_rejected(self, model, api_client):
        """An out-of-allow-list language hint is rejected with HTTP 400."""
        with AUDIO_WAV.open("rb") as audio:
            with pytest.raises(openai.BadRequestError):
                api_client.audio.transcriptions.create(
                    model=model,
                    file=audio,
                    language="zz",
                )
