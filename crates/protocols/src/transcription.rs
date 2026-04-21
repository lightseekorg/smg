//! Audio transcription API protocol definitions.
//!
//! This module defines the request type for the OpenAI-compatible
//! `/v1/audio/transcriptions` endpoint. The wire format is
//! `multipart/form-data` (an audio file plus text form fields), so the
//! struct below carries only the non-file fields; the file bytes travel
//! through the router as a separate payload.

use serde::{Deserialize, Serialize};

use super::common::GenerationRequest;

/// Transcription request - compatible with OpenAI's /v1/audio/transcriptions API.
///
/// The audio file itself is carried out-of-band because the endpoint uses
/// multipart/form-data, not JSON.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, Deserialize, Serialize, schemars::JsonSchema)]
pub struct TranscriptionRequest {
    /// ID of the model to use (e.g. "whisper-large-v3").
    pub model: String,

    /// Optional ISO-639-1 language hint for the input audio.
    pub language: Option<String>,

    /// Optional text prompt to guide the model's style / preserve continuity.
    pub prompt: Option<String>,

    /// Response format: `json` (default), `text`, `srt`, `verbose_json`, `vtt`.
    pub response_format: Option<String>,

    /// Sampling temperature (0..=1).
    pub temperature: Option<f32>,

    /// Timestamp granularities for verbose_json: `word`, `segment`.
    pub timestamp_granularities: Option<Vec<String>>,

    /// If true, stream partial transcription events as SSE.
    pub stream: Option<bool>,
}

impl GenerationRequest for TranscriptionRequest {
    fn is_stream(&self) -> bool {
        self.stream.unwrap_or(false)
    }

    fn get_model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn extract_text_for_routing(&self) -> String {
        // Audio bytes are not visible here; use the optional prompt as a
        // rough cache-aware routing hint when present.
        self.prompt.clone().unwrap_or_default()
    }
}
