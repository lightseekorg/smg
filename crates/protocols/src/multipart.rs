//! Axum extractors for multipart/form-data inference endpoints.
//!
//! JSON endpoints get [`crate::validated::ValidatedJson`]; the
//! `/v1/audio/transcriptions` endpoint uses multipart/form-data and gets
//! [`AudioTranscriptionMultipart`], which parses the form into a typed
//! `(TranscriptionRequest, AudioFile)` pair before the handler runs.

#[cfg(feature = "axum")]
use axum::{
    extract::{multipart::MultipartError, FromRequest, Multipart, Request},
    http::StatusCode,
    response::{IntoResponse, Response},
};

#[cfg(feature = "axum")]
use crate::transcription::{AudioFile, TranscriptionRequest};

/// Extractor for `/v1/audio/transcriptions` requests.
///
/// Parses `multipart/form-data` into a [`TranscriptionRequest`] (text fields)
/// plus an [`AudioFile`] (the `file` part). Returns `400 Bad Request` on
/// malformed parts, missing/empty `file`, missing/blank `model`, or
/// out-of-range `temperature`.
#[cfg(feature = "axum")]
pub struct AudioTranscriptionMultipart {
    pub request: TranscriptionRequest,
    pub audio: AudioFile,
}

#[cfg(feature = "axum")]
impl<S: Send + Sync> FromRequest<S> for AudioTranscriptionMultipart {
    type Rejection = Response;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        let mut multipart = Multipart::from_request(req, state)
            .await
            .map_err(IntoResponse::into_response)?;

        let mut file_bytes: Option<bytes::Bytes> = None;
        let mut file_name: Option<String> = None;
        let mut file_content_type: Option<String> = None;
        let mut request = TranscriptionRequest::default();
        let mut timestamp_granularities: Vec<String> = Vec::new();

        loop {
            let field = match multipart.next_field().await {
                Ok(Some(f)) => f,
                Ok(None) => break,
                Err(e) => {
                    return Err(bad_request(format!("Failed to read multipart field: {e}")));
                }
            };

            let name = field.name().unwrap_or("").to_string();
            match name.as_str() {
                "file" => {
                    file_name = field.file_name().map(str::to_string);
                    file_content_type = field.content_type().map(str::to_string);
                    match field.bytes().await {
                        Ok(b) => file_bytes = Some(b),
                        Err(e) => {
                            return Err(bad_request(format!(
                                "Failed to read audio file bytes: {e}"
                            )));
                        }
                    }
                }
                "model" => match field.text().await {
                    Ok(t) => request.model = t,
                    Err(e) => return Err(bad_text_field("model", e)),
                },
                "language" => match field.text().await {
                    Ok(t) => request.language = Some(t),
                    Err(e) => return Err(bad_text_field("language", e)),
                },
                "prompt" => match field.text().await {
                    Ok(t) => request.prompt = Some(t),
                    Err(e) => return Err(bad_text_field("prompt", e)),
                },
                "response_format" => match field.text().await {
                    Ok(t) => request.response_format = Some(t),
                    Err(e) => return Err(bad_text_field("response_format", e)),
                },
                "temperature" => match field.text().await {
                    Ok(t) => match t.trim().parse::<f32>() {
                        Ok(v) if v.is_finite() && (0.0..=1.0).contains(&v) => {
                            request.temperature = Some(v);
                        }
                        Ok(v) => {
                            return Err(bad_request(format!(
                                "Invalid 'temperature' value: {v} (must be a finite number in [0.0, 1.0])"
                            )));
                        }
                        Err(e) => {
                            return Err(bad_request(format!("Invalid 'temperature' value: {e}")));
                        }
                    },
                    Err(e) => return Err(bad_text_field("temperature", e)),
                },
                "timestamp_granularities" | "timestamp_granularities[]" => {
                    match field.text().await {
                        Ok(t) => timestamp_granularities.push(t),
                        Err(e) => return Err(bad_text_field("timestamp_granularities", e)),
                    }
                }
                "stream" => match field.text().await {
                    Ok(t) => match t.as_str() {
                        "true" | "True" | "TRUE" | "1" => request.stream = Some(true),
                        "false" | "False" | "FALSE" | "0" => request.stream = Some(false),
                        other => {
                            return Err(bad_request(format!(
                                "Invalid 'stream' value: '{other}' (expected true/false/1/0)"
                            )));
                        }
                    },
                    Err(e) => return Err(bad_text_field("stream", e)),
                },
                _ => {
                    // Unknown field; drain to free resources but otherwise ignore.
                    let _ = field.bytes().await;
                }
            }
        }

        if request.model.trim().is_empty() {
            return Err(bad_request("Missing required 'model' field".to_string()));
        }
        request.model = request.model.trim().to_string();

        let bytes = match file_bytes {
            Some(b) if !b.is_empty() => b,
            Some(_) => {
                return Err(bad_request("Uploaded 'file' part is empty".to_string()));
            }
            None => {
                return Err(bad_request("Missing required 'file' part".to_string()));
            }
        };

        if !timestamp_granularities.is_empty() {
            request.timestamp_granularities = Some(timestamp_granularities);
        }

        let audio = AudioFile {
            bytes,
            file_name: file_name.unwrap_or_else(|| "audio".to_string()),
            content_type: file_content_type,
        };

        Ok(AudioTranscriptionMultipart { request, audio })
    }
}

#[cfg(feature = "axum")]
fn bad_request(message: String) -> Response {
    (StatusCode::BAD_REQUEST, message).into_response()
}

#[cfg(feature = "axum")]
fn bad_text_field(field: &str, e: MultipartError) -> Response {
    bad_request(format!("Failed to read '{field}' field: {e}"))
}
