//! Transcription session types for the Realtime API.
//!
//! This module contains types specific to transcription sessions, which provide
//! speech-to-text functionality without full conversation capabilities.
//! Transcription sessions use a subset of the realtime session configuration.

use serde::{Deserialize, Serialize};

use super::session::{AudioFormat, InputAudioNoiseReduction, InputAudioTranscription};

// ============================================================================
// Main Types
// ============================================================================
/// Full transcription session object returned in server events.
///
/// Returned in `transcription_session.created` and `transcription_session.updated` events.
/// Matches the GA `RealtimeTranscriptionSessionCreateResponseGA` schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionSession {
    /// Unique identifier for the session
    pub id: String,
    /// Object type (always "realtime.transcription_session")
    #[serde(default = "transcription_session_object_type")]
    pub object: String,
    /// Session type (always "transcription")
    #[serde(rename = "type", default = "transcription_session_type_default")]
    pub session_type: String,
    /// Unix timestamp when the session expires
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<u64>,
    /// Additional fields included in outputs
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<String>>,
    /// Audio input configuration
    #[serde(default)]
    pub audio: TranscriptionAudioConfig,
}


/// Configuration for creating or updating a transcription session.
///
/// Used in `transcription_session.update` client events.
/// Matches the GA `RealtimeTranscriptionSessionCreateRequestGA` schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionSessionConfig {
    /// Session type (always "transcription", defaults to "transcription")
    #[serde(rename = "type", default = "transcription_session_object_type")]
    pub session_type: String,

    /// Audio input configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<TranscriptionAudioConfig>,

    /// Additional fields to include in server outputs
    /// (e.g., ["item.input_audio_transcription.logprobs"])
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<String>>,
}

impl Default for TranscriptionSessionConfig {
    fn default() -> Self {
        Self {
            session_type: "transcription".to_string(),
            audio: None,
            include: None,
        }
    }
}

fn transcription_session_object_type() -> String {
    "realtime.transcription_session".to_string()
}

fn transcription_session_type_default() -> String {
    "transcription".to_string()
}


// ============================================================================
// Supporting Types
// ============================================================================

/// Audio configuration for transcription sessions (input only).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TranscriptionAudioConfig {
    /// Input audio configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<TranscriptionAudioInputConfig>,
}

/// Audio input configuration for transcription sessions.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TranscriptionAudioInputConfig {
    /// Input audio format
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<AudioFormat>,
    /// Transcription configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transcription: Option<InputAudioTranscription>,
    /// Noise reduction configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub noise_reduction: Option<InputAudioNoiseReduction>,
    /// Turn detection configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_detection: Option<TranscriptionTurnDetection>,
}

/// Turn detection configuration for transcription sessions.
///
/// Only `server_vad` is supported for transcription sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TranscriptionTurnDetection {
    ServerVad {
        /// Activation threshold for VAD (0.0-1.0, default 0.5)
        #[serde(skip_serializing_if = "Option::is_none")]
        threshold: Option<f32>,
        /// Audio to include before speech starts (ms, default 300)
        #[serde(skip_serializing_if = "Option::is_none")]
        prefix_padding_ms: Option<u32>,
        /// Duration of silence to detect end of speech (ms, default 500)
        #[serde(skip_serializing_if = "Option::is_none")]
        silence_duration_ms: Option<u32>,
    },
}

/// A transcription segment returned in
/// `conversation.item.input_audio_transcription.segment` events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionSegment {
    /// Segment identifier
    pub id: String,
    /// The transcription text for this segment
    pub text: String,
    /// Detected speaker label for this segment
    pub speaker: String,
    /// Start time of the segment in seconds
    pub start: f32,
    /// End time of the segment in seconds
    pub end: f32,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transcription_session_config_minimal() {
        let config = TranscriptionSessionConfig {
            session_type: "transcription".to_string(),
            audio: None,
            include: None,
        };
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"type\":\"transcription\""));
        assert!(!json.contains("\"audio\""));
    }

    #[test]
    fn test_transcription_session_config_with_audio() {
        let config = TranscriptionSessionConfig {
            session_type: "transcription".to_string(),
            audio: Some(TranscriptionAudioConfig {
                input: Some(TranscriptionAudioInputConfig {
                    format: Some(AudioFormat::pcm()),
                    ..Default::default()
                }),
            }),
            include: Some(vec!["item.input_audio_transcription.logprobs".to_string()]),
        };
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"type\":\"transcription\""));
        assert!(json.contains("\"audio/pcm\""));
        assert!(json.contains("logprobs"));
    }

    #[test]
    fn test_transcription_session_deserialization() {
        let json = r#"{
            "id": "sess_abc",
            "object": "realtime.transcription_session",
            "type": "transcription",
            "expires_at": 1700000000,
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "transcription": {"model": "gpt-4o-transcribe"},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 500
                    }
                }
            },
            "include": ["item.input_audio_transcription.logprobs"]
        }"#;
        let session: TranscriptionSession = serde_json::from_str(json).unwrap();
        assert_eq!(session.id, "sess_abc");
        assert_eq!(session.session_type, "transcription");
        assert!(session.audio.input.is_some());
        let input = session.audio.input.unwrap();
        assert!(input.turn_detection.is_some());
        assert!(matches!(
            input.turn_detection.unwrap(),
            TranscriptionTurnDetection::ServerVad { .. }
        ));
    }

    #[test]
    fn test_transcription_segment() {
        let segment = TranscriptionSegment {
            id: "seg_001".to_string(),
            text: "Hello world".to_string(),
            speaker: "speaker_0".to_string(),
            start: 0.5,
            end: 1.2,
        };
        let json = serde_json::to_string(&segment).unwrap();
        assert!(json.contains("\"id\":\"seg_001\""));
        assert!(json.contains("\"speaker\":\"speaker_0\""));
        assert!(json.contains("\"start\":0.5"));
    }

    #[test]
    fn test_transcription_turn_detection_serialization() {
        let td = TranscriptionTurnDetection::ServerVad {
            threshold: Some(0.6),
            prefix_padding_ms: Some(300),
            silence_duration_ms: Some(200),
        };
        let json = serde_json::to_string(&td).unwrap();
        assert!(json.contains("\"type\":\"server_vad\""));
        assert!(json.contains("\"threshold\":0.6"));
    }
}
