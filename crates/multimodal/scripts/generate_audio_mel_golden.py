#!/usr/bin/env python3
"""Generate a HuggingFace reference log-mel golden for Qwen3 audio preprocessing.

Reference extractor
-------------------
Qwen3-ASR and Qwen3-Omni consume audio through ``WhisperFeatureExtractor``
(``Qwen3OmniMoeProcessor.feature_extractor_class == "WhisperFeatureExtractor"``).
This script constructs that extractor locally with the parameters Qwen3's
``preprocessor_config.json`` sets (128 mel bins, 16 kHz, ``n_fft=400``,
``hop_length=160``), so it needs neither model weights nor a network download.

The extractor is run on deterministic, seeded synthetic waveforms so the golden
is reproducible without any external audio file. Every case's input PCM and the
reference outputs are dumped to a checked-in JSON fixture that the Rust
integration test loads with ``include_str!`` as an external correctness oracle.

Cases
-----
The fixture holds several cases so the Rust frontend is diffed across the paths
that actually matter, not just one 16 kHz clip:

* ``native_16k``     -- 16 kHz clip, no resample (the mel path in isolation).
* ``resample_44100`` -- 44.1 kHz clip resampled to 16 kHz, then mel.
* ``resample_48000`` -- 48 kHz clip resampled to 16 kHz, then mel.
* ``short_clip``     -- a clip shorter than ``n_fft`` (reflect-pad / frame edge).
* ``silence``        -- zeros (exercises the Whisper log-norm floor).
* ``batch_16k``      -- a batch of three 16 kHz clips of different lengths,
  carrying the batched-path metadata (``encoder_input`` shape,
  ``feature_attention_mask``, ``audio_feature_lengths``).

Parity note
-----------
SMG's frontend mirrors ``WhisperFeatureExtractor._torch_extract_fbank_features``
applied to the raw (un-padded) waveform: a centered STFT (reflect pad
``n_fft // 2``) yielding ``floor(len / hop) + 1`` frames, with the trailing frame
dropped (``stft[..., :-1]``), Slaney mel filters, and the Whisper log
normalization ``clamp(log10(x), peak - 8); (x + 4) / 4``. We therefore call the
extractor's internal fbank routine directly instead of its public ``__call__``,
which would otherwise pad/truncate every clip to the fixed 30 s window. For a
non-16 kHz clip we first resample with ``torchaudio.functional.resample`` -- the
same band-limited kernel SMG's ``bandlimited_resample`` targets -- so the golden
matches what Rust computes end to end from the original ``DecodedAudio``.

For the batch case the Rust path pads every (resampled) waveform to the longest
one, runs the per-clip log-mel over the padded waveform (its Whisper peak/floor
is therefore computed per padded clip), and reports ``feature_length =
floor(orig_len / hop)`` clamped to ``max_frames`` for the attention mask and
lengths. We reproduce exactly that here.

Usage::

    python3 crates/multimodal/scripts/generate_audio_mel_golden.py \
        > crates/multimodal/tests/fixtures/golden/audio_mel_reference.json
"""

import json

import numpy as np
import torch
import torchaudio.functional as AF
import transformers
from transformers import WhisperFeatureExtractor

# Matches Qwen3AudioParams::default() / Qwen3 preprocessor_config.json.
SAMPLE_RATE = 16000
FEATURE_SIZE = 128
N_FFT = 400
HOP_LENGTH = 160


def make_waveform(sample_rate: int, duration_seconds: float, seed: int) -> np.ndarray:
    """Build a deterministic mono f32 waveform at ``sample_rate``.

    A seeded mix of sinusoids plus a little reproducible noise, normalized to a
    stable peak so the log-mel floor is well defined, then cast to f32 -- the
    exact dtype the Rust side receives from decode.
    """
    num_samples = int(round(duration_seconds * sample_rate))
    t = np.arange(num_samples, dtype=np.float64) / sample_rate
    signal = (
        0.6 * np.sin(2.0 * np.pi * 220.0 * t)
        + 0.3 * np.sin(2.0 * np.pi * 440.0 * t)
        + 0.1 * np.sin(2.0 * np.pi * 3300.0 * t)
    )
    rng = np.random.default_rng(seed)
    signal += 0.01 * rng.standard_normal(num_samples)
    peak = float(np.max(np.abs(signal))) if num_samples else 0.0
    if peak > 0.0:
        signal = signal / peak * 0.95
    return signal.astype(np.float32)


def resample_to_16k(waveform: np.ndarray, orig_sr: int) -> np.ndarray:
    """Resample to 16 kHz with the kernel SMG's ``bandlimited_resample`` targets.

    ``torchaudio.functional.resample`` defaults to a Hann-windowed sinc filter,
    ``lowpass_filter_width=6``, ``rolloff=0.99`` -- the exact configuration the
    Rust resampler reproduces, so this is the oracle for the resample path.
    """
    if orig_sr == SAMPLE_RATE:
        return waveform.astype(np.float32)
    tensor = torch.from_numpy(waveform.astype(np.float32))
    resampled = AF.resample(tensor, orig_sr, SAMPLE_RATE)
    return resampled.numpy().astype(np.float32)


def reference_log_mel(extractor: WhisperFeatureExtractor, waveform: np.ndarray) -> np.ndarray:
    """Reference log-mel for the raw waveform (no fixed 30 s pad/truncate).

    ``_torch_extract_fbank_features`` performs the centered STFT, power spectrum,
    Slaney mel projection, and Whisper log normalization, and already drops the
    trailing STFT frame -- exactly the pipeline SMG reproduces per clip.
    """
    features = extractor._torch_extract_fbank_features(waveform.astype(np.float32))
    return np.asarray(features, dtype=np.float32)


def single_case(
    name: str,
    extractor: WhisperFeatureExtractor,
    *,
    sample_rate: int,
    duration_seconds: float,
    seed: int,
    silence: bool = False,
) -> dict:
    """Build one single-clip case: original PCM in, un-padded reference mel out."""
    if silence:
        num_samples = int(round(duration_seconds * sample_rate))
        pcm = np.zeros(num_samples, dtype=np.float32)
    else:
        pcm = make_waveform(sample_rate, duration_seconds, seed)

    waveform_16k = resample_to_16k(pcm, sample_rate)
    mel = reference_log_mel(extractor, waveform_16k)
    assert mel.ndim == 2 and mel.shape[0] == FEATURE_SIZE, mel.shape
    # HF drops the trailing frame, matching SMG's floor(len / hop).
    assert mel.shape[1] == waveform_16k.shape[0] // HOP_LENGTH, (
        mel.shape,
        waveform_16k.shape[0] // HOP_LENGTH,
    )

    return {
        "name": name,
        "kind": "single",
        "sample_rate": sample_rate,
        "pcm": [float(sample) for sample in pcm.tolist()],
        "mel_shape": list(mel.shape),
        # Row-major (mel, frame), matching the Rust Array2 layout.
        "mel": [float(value) for value in mel.reshape(-1).tolist()],
        "mel_sum": float(np.sum(mel, dtype=np.float64)),
    }


def batch_case(
    name: str,
    extractor: WhisperFeatureExtractor,
    *,
    specs: list[tuple[int, float, int]],
) -> dict:
    """Build the batched case reproducing SMG's ``preprocess_decoded_clips``.

    Each clip is resampled to 16 kHz, the batch is padded to the longest
    waveform, and the per-clip log-mel is taken over the padded waveform (so its
    Whisper peak/floor is per padded clip, exactly as Rust runs it).
    """
    waveforms = [resample_to_16k(make_waveform(sr, dur, seed), sr) for (sr, dur, seed) in specs]
    orig_lengths = [wave.shape[0] for wave in waveforms]
    max_samples = max(orig_lengths)
    max_frames = max_samples // HOP_LENGTH

    mels = []
    feature_lengths = []
    attention_mask = []
    for wave, orig_len in zip(waveforms, orig_lengths):
        padded = np.zeros(max_samples, dtype=np.float32)
        padded[:orig_len] = wave
        mel = reference_log_mel(extractor, padded)
        assert mel.shape == (FEATURE_SIZE, max_frames), (mel.shape, max_frames)
        mels.append(mel)

        feature_length = min(orig_len // HOP_LENGTH, max_frames)
        feature_lengths.append(int(feature_length))
        attention_mask.append([1 if frame < feature_length else 0 for frame in range(max_frames)])

    # Stack to [B, n_mels, max_frames], row-major to match the Rust Array3.
    stacked = np.stack(mels, axis=0)
    return {
        "name": name,
        "kind": "batch",
        "sample_rate": SAMPLE_RATE,
        "pcm_clips": [[float(sample) for sample in wave.tolist()] for wave in waveforms],
        "encoder_shape": [len(waveforms), FEATURE_SIZE, max_frames],
        "mel": [float(value) for value in stacked.reshape(-1).tolist()],
        "mel_sum": float(np.sum(stacked, dtype=np.float64)),
        "feature_attention_mask": attention_mask,
        "audio_feature_lengths": feature_lengths,
    }


def main() -> None:
    extractor = WhisperFeatureExtractor(
        feature_size=FEATURE_SIZE,
        sampling_rate=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT,
        chunk_length=30,
        padding_value=0.0,
        # dither defaults to 0.0; keep it explicit so the golden is deterministic.
        dither=0.0,
    )

    cases = [
        # 16 kHz, ~0.5 s -- the mel path in isolation, no resample.
        single_case(
            "native_16k",
            extractor,
            sample_rate=16000,
            duration_seconds=0.5,
            seed=1905,
        ),
        # 44.1 kHz -> 16 kHz, then mel.
        single_case(
            "resample_44100",
            extractor,
            sample_rate=44100,
            duration_seconds=0.35,
            seed=44100,
        ),
        # 48 kHz -> 16 kHz, then mel.
        single_case(
            "resample_48000",
            extractor,
            sample_rate=48000,
            duration_seconds=0.35,
            seed=48000,
        ),
        # Shorter than n_fft=400 samples: reflect-pad / frame-count edge.
        single_case(
            "short_clip",
            extractor,
            sample_rate=16000,
            # 300 samples < n_fft (400) but >= hop_length (160): 1 frame.
            duration_seconds=300 / 16000,
            seed=300,
        ),
        # Silence (~0.3 s of zeros): Whisper normalization floor.
        single_case(
            "silence",
            extractor,
            sample_rate=16000,
            duration_seconds=0.3,
            seed=0,
            silence=True,
        ),
        # Batch of three 16 kHz clips of different lengths: the batched path.
        # Kept short (longest ~0.35 s) so the padded [B, n_mels, max_frames]
        # slab -- the bulk of the fixture -- stays small.
        batch_case(
            "batch_16k",
            extractor,
            specs=[
                (16000, 0.35, 11),
                (16000, 0.25, 22),
                (16000, 0.15, 33),
            ],
        ),
    ]

    document = {
        "generator": "generate_audio_mel_golden.py",
        "reference": "transformers.WhisperFeatureExtractor",
        "resampler": "torchaudio.functional.resample",
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "sample_rate": SAMPLE_RATE,
        "n_mels": FEATURE_SIZE,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "cases": cases,
    }
    print(json.dumps(document))


if __name__ == "__main__":
    main()
