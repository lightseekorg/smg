//! Preprocessing for Responses API input content parts (R2).
//!
//! The P1 protocol schema added three new [`ResponseContentPart`] variants to
//! the gRPC regular Responses router: `InputImage`, `InputFile`, and `Refusal`.
//! The downstream chat pipeline only understands OpenAI Chat Completions
//! [`ContentPart`] (`text` + `image_url`), so this module runs BEFORE
//! [`super::conversions::responses_to_chat`] and normalizes every new variant
//! into something the chat pipeline already handles — or rejects it with a
//! 400 `unsupported_content` when SMG cannot service the request.
//!
//! Specifically:
//!
//! - `InputImage` with an HTTP or data URL → left in place; `conversions` maps
//!   it to [`ContentPart::ImageUrl`] which [`crate::routers::grpc::multimodal`]
//!   then downloads/decodes.
//! - `InputImage` with `file_id` → rejected (SMG has no Files API backend).
//! - `InputFile` with `file_data` (base64) → decoded, sniffed for magic bytes;
//!   images are re-emitted as `InputImage` with a `data:` URL so they flow
//!   through the existing image pipeline; PDFs are rejected pending R4.
//! - `InputFile` with `file_url` → downloaded via [`MediaConnector`] (same
//!   allowlist + timeout envelope as the image pipeline), sniffed, and then
//!   either re-emitted as an image data URL or rejected (PDF / non-image).
//! - `InputFile` with `file_id` → rejected.
//! - `Refusal` on an input (user) role → rejected; refusals are output-only.
//!
//! After preprocessing, `responses_to_chat` only has to map the narrower
//! surface of text + image-url content parts.

use std::sync::Arc;

use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine};
use llm_multimodal::MediaConnector;
use openai_protocol::responses::{
    ResponseContentPart, ResponseInput, ResponseInputOutputItem, ResponsesRequest,
    StringOrContentParts,
};
use tracing::warn;

/// Typed error returned by preprocessing and
/// [`super::conversions::responses_to_chat`]. Callers map each variant to an
/// appropriate HTTP response (400 with a specific error code, or 500).
#[derive(Debug, Clone)]
pub(super) enum ConversionError {
    /// The request carries a content part that SMG cannot handle. Should
    /// surface as HTTP 400 with `error.code = "unsupported_content"`.
    UnsupportedContent(String),
    /// The request itself is malformed (e.g. empty after normalization).
    /// Should surface as HTTP 400 with `error.code = "invalid_request"`.
    InvalidRequest(String),
}

impl ConversionError {
    pub(super) fn error_code(&self) -> &'static str {
        match self {
            Self::UnsupportedContent(_) => "unsupported_content",
            Self::InvalidRequest(_) => "invalid_request",
        }
    }

    pub(super) fn message(&self) -> &str {
        match self {
            Self::UnsupportedContent(m) | Self::InvalidRequest(m) => m,
        }
    }
}

impl std::fmt::Display for ConversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.error_code(), self.message())
    }
}

/// Shared mapping from [`ConversionError`] to an HTTP 400 response. Both
/// [`ConversionError::UnsupportedContent`] and [`ConversionError::InvalidRequest`]
/// are client-side issues; the typed code in the response body lets clients
/// distinguish them without parsing free-form messages.
pub(super) fn conversion_error_to_response(err: &ConversionError) -> axum::response::Response {
    use crate::routers::error::bad_request;
    bad_request(err.error_code(), err.message())
}

/// Walk the request's structured input and normalize every `InputImage` /
/// `InputFile` / `Refusal` content part. Mutates `req.input` in place.
///
/// Network I/O happens only for `InputFile` with `file_url`. `media_connector`
/// may be `None` when the router is running without multimodal support; in
/// that case any content part that would have required a download is
/// rejected with `unsupported_content` rather than silently downloaded
/// through an out-of-envelope client.
pub(super) async fn preprocess_responses_input(
    req: &mut ResponsesRequest,
    media_connector: Option<&Arc<MediaConnector>>,
) -> Result<(), ConversionError> {
    // Text input has no rich content parts so there is nothing to preprocess.
    let items = match &mut req.input {
        ResponseInput::Text(_) => return Ok(()),
        ResponseInput::Items(items) => items,
    };

    for item in items.iter_mut() {
        match item {
            ResponseInputOutputItem::SimpleInputMessage {
                role,
                content: StringOrContentParts::Array(parts),
                ..
            } => {
                normalize_content_parts(parts, role.as_str(), media_connector).await?;
            }
            ResponseInputOutputItem::Message { role, content, .. } => {
                normalize_content_parts(content, role.as_str(), media_connector).await?;
            }
            _ => {}
        }
    }

    Ok(())
}

/// Normalize an in-place slice of content parts. Parts are rewritten rather
/// than re-collected so callers can keep the original order and any
/// surrounding `InputText` entries.
async fn normalize_content_parts(
    parts: &mut [ResponseContentPart],
    role: &str,
    media_connector: Option<&Arc<MediaConnector>>,
) -> Result<(), ConversionError> {
    for part in parts.iter_mut() {
        normalize_one(part, role, media_connector).await?;
    }
    Ok(())
}

async fn normalize_one(
    part: &mut ResponseContentPart,
    role: &str,
    media_connector: Option<&Arc<MediaConnector>>,
) -> Result<(), ConversionError> {
    match part {
        // Text / output text are passed through untouched.
        ResponseContentPart::InputText { .. } | ResponseContentPart::OutputText { .. } => Ok(()),

        ResponseContentPart::Refusal { .. } => {
            // Refusals are an output-only spec type. An input-role refusal is
            // a client bug; preserving it as text would silently change the
            // prompt semantics, so reject it explicitly.
            if is_input_role(role) {
                Err(ConversionError::UnsupportedContent(
                    "Refusal content parts are output-only; reject on input role".to_string(),
                ))
            } else {
                // On an assistant/other output role a refusal is legitimate
                // and downstream text extraction preserves it verbatim.
                Ok(())
            }
        }

        ResponseContentPart::InputImage {
            file_id,
            image_url,
            detail: _,
        } => {
            if file_id.is_some() {
                // SMG does not host a Files API, so file_id handles are not
                // resolvable here. Reject cleanly; the caller can forward the
                // request to an OpenAI-compat backend if it wants the
                // passthrough behavior (R1).
                return Err(ConversionError::UnsupportedContent(
                    "input_image.file_id is not supported by the gRPC regular router; \
                     pass image_url (HTTP or data URL) instead"
                        .to_string(),
                ));
            }
            if image_url.is_none() {
                return Err(ConversionError::InvalidRequest(
                    "input_image requires either image_url or file_id".to_string(),
                ));
            }
            // image_url (HTTP or data URL) passes through — the chat pipeline
            // already knows how to fetch both shapes via MediaConnector.
            Ok(())
        }

        ResponseContentPart::InputFile {
            file_data,
            file_id,
            file_url,
            filename,
            detail: _,
        } => {
            if let Some(name) = filename.as_deref() {
                // Filename is informational metadata. Log but do not fail,
                // since the spec treats it as a hint not a handle.
                warn!(filename = %name, "input_file filename is informational — ignored");
            }

            if file_id.is_some() {
                return Err(ConversionError::UnsupportedContent(
                    "input_file.file_id is not supported by the gRPC regular router; \
                     pass file_data (base64) or file_url instead"
                        .to_string(),
                ));
            }

            if let Some(data) = file_data.take() {
                let bytes = BASE64_STANDARD.decode(data.as_bytes()).map_err(|e| {
                    ConversionError::InvalidRequest(format!(
                        "input_file.file_data is not valid base64: {e}"
                    ))
                })?;
                let normalized = sniff_and_build_image_part(&bytes)?;
                *part = normalized;
                return Ok(());
            }

            if let Some(url) = file_url.take() {
                let connector = media_connector.ok_or_else(|| {
                    ConversionError::UnsupportedContent(
                        "input_file.file_url requires the multimodal pipeline, which is not \
                         configured on this gateway instance"
                            .to_string(),
                    )
                })?;
                let (bytes, _resolved) = connector.fetch_raw_bytes(&url).await.map_err(|e| {
                    ConversionError::UnsupportedContent(format!(
                        "failed to download input_file.file_url: {e}"
                    ))
                })?;
                let normalized = sniff_and_build_image_part(bytes.as_ref())?;
                *part = normalized;
                return Ok(());
            }

            Err(ConversionError::InvalidRequest(
                "input_file requires one of file_data, file_url, or file_id".to_string(),
            ))
        }
    }
}

/// Classify a byte payload by its file signature and build the matching
/// content part. Images become `InputImage` with a `data:` URL; PDFs are
/// rejected pending R4; everything else is rejected as `unsupported_content`.
fn sniff_and_build_image_part(bytes: &[u8]) -> Result<ResponseContentPart, ConversionError> {
    match sniff_media_kind(bytes) {
        MediaKind::Image(mime) => {
            let encoded = BASE64_STANDARD.encode(bytes);
            Ok(ResponseContentPart::InputImage {
                detail: None,
                file_id: None,
                image_url: Some(format!("data:{mime};base64,{encoded}")),
            })
        }
        MediaKind::Pdf => Err(ConversionError::UnsupportedContent(
            "input_file with application/pdf is not yet supported; PDF extraction pending R4"
                .to_string(),
        )),
        MediaKind::Unknown => Err(ConversionError::UnsupportedContent(
            "input_file payload is not a recognized image (jpeg/png/webp/gif) or PDF".to_string(),
        )),
    }
}

/// Minimal magic-byte classifier covering every MIME type the downstream
/// image pipeline supports plus the PDF sentinel we need to reject.
enum MediaKind {
    Image(&'static str),
    Pdf,
    Unknown,
}

fn sniff_media_kind(bytes: &[u8]) -> MediaKind {
    // JPEG: FF D8 FF
    if bytes.len() >= 3 && bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return MediaKind::Image("image/jpeg");
    }
    // PNG: 89 50 4E 47 0D 0A 1A 0A
    if bytes.len() >= 8
        && bytes.starts_with(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
    {
        return MediaKind::Image("image/png");
    }
    // GIF: "GIF87a" / "GIF89a"
    if bytes.len() >= 6 && (bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a")) {
        return MediaKind::Image("image/gif");
    }
    // WebP: "RIFF" ... "WEBP"
    if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP" {
        return MediaKind::Image("image/webp");
    }
    // PDF: "%PDF-"
    if bytes.starts_with(b"%PDF-") {
        return MediaKind::Pdf;
    }
    MediaKind::Unknown
}

/// Whether a message role denotes client-authored input (as opposed to
/// server/assistant output). Spec §Messages distinguishes `user`,
/// `developer`, `system`, and `critic` as input-bearing roles.
fn is_input_role(role: &str) -> bool {
    matches!(role, "user" | "developer" | "system" | "critic")
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // sniff_media_kind
    // ------------------------------------------------------------------

    #[test]
    fn sniff_detects_jpeg_magic() {
        let bytes = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10];
        assert!(matches!(sniff_media_kind(&bytes), MediaKind::Image(m) if m == "image/jpeg"));
    }

    #[test]
    fn sniff_detects_png_magic() {
        let bytes = [
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00,
        ];
        assert!(matches!(sniff_media_kind(&bytes), MediaKind::Image(m) if m == "image/png"));
    }

    #[test]
    fn sniff_detects_gif_magic() {
        let bytes = b"GIF89a....";
        assert!(matches!(sniff_media_kind(bytes), MediaKind::Image(m) if m == "image/gif"));
    }

    #[test]
    fn sniff_detects_webp_magic() {
        // RIFF + size bytes + WEBP
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&[0, 0, 0, 0]);
        bytes.extend_from_slice(b"WEBP");
        bytes.extend_from_slice(b"VP8 ");
        assert!(matches!(sniff_media_kind(&bytes), MediaKind::Image(m) if m == "image/webp"));
    }

    #[test]
    fn sniff_detects_pdf_magic() {
        let bytes = b"%PDF-1.7\n...";
        assert!(matches!(sniff_media_kind(bytes), MediaKind::Pdf));
    }

    #[test]
    fn sniff_returns_unknown_for_unrecognized_payload() {
        let bytes = b"not a real file";
        assert!(matches!(sniff_media_kind(bytes), MediaKind::Unknown));
    }

    #[test]
    fn sniff_guards_short_buffers() {
        // Buffer shorter than every signature must not panic and must not
        // match any known kind.
        for n in 0..3usize {
            let bytes = vec![0u8; n];
            assert!(matches!(sniff_media_kind(&bytes), MediaKind::Unknown));
        }
    }

    // ------------------------------------------------------------------
    // sniff_and_build_image_part
    // ------------------------------------------------------------------

    #[test]
    fn build_image_part_from_jpeg_bytes() {
        let bytes = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46];
        let part = sniff_and_build_image_part(&bytes).expect("jpeg must be accepted");
        match part {
            ResponseContentPart::InputImage {
                image_url: Some(url),
                file_id: None,
                detail: None,
            } => {
                assert!(
                    url.starts_with("data:image/jpeg;base64,"),
                    "expected jpeg data URL, got: {url}"
                );
                let encoded = url.strip_prefix("data:image/jpeg;base64,").unwrap();
                let roundtrip = BASE64_STANDARD.decode(encoded).unwrap();
                assert_eq!(roundtrip, bytes.to_vec());
            }
            other => panic!("expected InputImage, got {other:?}"),
        }
    }

    #[test]
    fn build_image_part_rejects_pdf_pending_r4() {
        let bytes = b"%PDF-1.4\n%\xc7\xec\x8f\xa2\n";
        let err = sniff_and_build_image_part(bytes).expect_err("PDF must be rejected");
        assert_eq!(err.error_code(), "unsupported_content");
        assert!(
            err.message().contains("R4"),
            "PDF error must reference R4, got: {}",
            err.message()
        );
    }

    #[test]
    fn build_image_part_rejects_unknown_bytes() {
        let err = sniff_and_build_image_part(b"garbage").expect_err("unknown kind must reject");
        assert_eq!(err.error_code(), "unsupported_content");
    }

    // ------------------------------------------------------------------
    // preprocess_responses_input — error paths (no HTTP required)
    // ------------------------------------------------------------------

    fn user_message_with_parts(parts: Vec<ResponseContentPart>) -> ResponsesRequest {
        ResponsesRequest {
            input: ResponseInput::Items(vec![ResponseInputOutputItem::Message {
                id: "msg_1".to_string(),
                role: "user".to_string(),
                content: parts,
                status: None,
                phase: None,
            }]),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn preprocess_text_input_is_noop() {
        let mut req = ResponsesRequest {
            input: ResponseInput::Text("hello".to_string()),
            ..Default::default()
        };
        preprocess_responses_input(&mut req, None).await.unwrap();
        match req.input {
            ResponseInput::Text(t) => assert_eq!(t, "hello"),
            ResponseInput::Items(_) => panic!("text input must stay text"),
        }
    }

    #[tokio::test]
    async fn preprocess_input_image_data_url_passes_through() {
        let url = "data:image/png;base64,iVBORw0KGgoAAAANS".to_string();
        let mut req = user_message_with_parts(vec![ResponseContentPart::InputImage {
            detail: None,
            file_id: None,
            image_url: Some(url.clone()),
        }]);
        preprocess_responses_input(&mut req, None).await.unwrap();
        let parts = match &req.input {
            ResponseInput::Items(items) => match &items[0] {
                ResponseInputOutputItem::Message { content, .. } => content,
                other => panic!("unexpected item: {other:?}"),
            },
            ResponseInput::Text(_) => panic!("unexpected text input"),
        };
        match &parts[0] {
            ResponseContentPart::InputImage {
                image_url: Some(u), ..
            } => assert_eq!(u, &url),
            other => panic!("expected InputImage, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn preprocess_rejects_input_image_file_id() {
        let mut req = user_message_with_parts(vec![ResponseContentPart::InputImage {
            detail: None,
            file_id: Some("file_abc".to_string()),
            image_url: None,
        }]);
        let err = preprocess_responses_input(&mut req, None)
            .await
            .expect_err("file_id must be rejected");
        assert_eq!(err.error_code(), "unsupported_content");
    }

    #[tokio::test]
    async fn preprocess_rejects_input_image_missing_source() {
        let mut req = user_message_with_parts(vec![ResponseContentPart::InputImage {
            detail: None,
            file_id: None,
            image_url: None,
        }]);
        let err = preprocess_responses_input(&mut req, None)
            .await
            .expect_err("missing source must be rejected");
        assert_eq!(err.error_code(), "invalid_request");
    }

    #[tokio::test]
    async fn preprocess_rewrites_input_file_base64_jpeg() {
        // Minimal JPEG magic bytes — the downstream image decoder is not
        // involved at this stage; we only care that preprocessing accepts
        // JPEG magic and rewrites to an InputImage data URL.
        let jpeg_bytes = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A];
        let file_data = BASE64_STANDARD.encode(jpeg_bytes);
        let mut req = user_message_with_parts(vec![ResponseContentPart::InputFile {
            detail: None,
            file_data: Some(file_data),
            file_id: None,
            file_url: None,
            filename: Some("photo.jpg".to_string()),
        }]);

        preprocess_responses_input(&mut req, None).await.unwrap();

        let parts = match &req.input {
            ResponseInput::Items(items) => match &items[0] {
                ResponseInputOutputItem::Message { content, .. } => content,
                other => panic!("unexpected item: {other:?}"),
            },
            ResponseInput::Text(_) => panic!("unexpected text input"),
        };

        match &parts[0] {
            ResponseContentPart::InputImage {
                image_url: Some(url),
                ..
            } => {
                assert!(url.starts_with("data:image/jpeg;base64,"));
                let encoded = url.strip_prefix("data:image/jpeg;base64,").unwrap();
                let decoded = BASE64_STANDARD.decode(encoded).unwrap();
                assert_eq!(decoded, jpeg_bytes);
            }
            other => panic!("expected InputImage after rewrite, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn preprocess_rejects_input_file_pdf_pending_r4() {
        let pdf_bytes = b"%PDF-1.4\n%\xc7\xec\x8f\xa2\n";
        let file_data = BASE64_STANDARD.encode(pdf_bytes);
        let mut req = user_message_with_parts(vec![ResponseContentPart::InputFile {
            detail: None,
            file_data: Some(file_data),
            file_id: None,
            file_url: None,
            filename: Some("spec.pdf".to_string()),
        }]);

        let err = preprocess_responses_input(&mut req, None)
            .await
            .expect_err("PDF must be rejected");
        assert_eq!(err.error_code(), "unsupported_content");
        assert!(
            err.message().contains("R4"),
            "PDF error must cite R4, got: {}",
            err.message()
        );
    }

    #[tokio::test]
    async fn preprocess_rejects_input_file_file_id() {
        let mut req = user_message_with_parts(vec![ResponseContentPart::InputFile {
            detail: None,
            file_data: None,
            file_id: Some("file_abc".to_string()),
            file_url: None,
            filename: None,
        }]);
        let err = preprocess_responses_input(&mut req, None)
            .await
            .expect_err("file_id must be rejected");
        assert_eq!(err.error_code(), "unsupported_content");
    }

    #[tokio::test]
    async fn preprocess_rejects_input_file_empty() {
        let mut req = user_message_with_parts(vec![ResponseContentPart::InputFile {
            detail: None,
            file_data: None,
            file_id: None,
            file_url: None,
            filename: None,
        }]);
        let err = preprocess_responses_input(&mut req, None)
            .await
            .expect_err("empty input_file must be rejected");
        assert_eq!(err.error_code(), "invalid_request");
    }

    #[tokio::test]
    async fn preprocess_rejects_input_file_url_without_media_connector() {
        let mut req = user_message_with_parts(vec![ResponseContentPart::InputFile {
            detail: None,
            file_data: None,
            file_id: None,
            file_url: Some("https://example.com/x.jpg".to_string()),
            filename: None,
        }]);
        let err = preprocess_responses_input(&mut req, None)
            .await
            .expect_err("file_url without connector must be rejected");
        assert_eq!(err.error_code(), "unsupported_content");
    }

    #[tokio::test]
    async fn preprocess_rejects_refusal_on_input_role() {
        let mut req = user_message_with_parts(vec![ResponseContentPart::Refusal {
            refusal: "I can't help with that.".to_string(),
        }]);
        let err = preprocess_responses_input(&mut req, None)
            .await
            .expect_err("input-role refusal must be rejected");
        assert_eq!(err.error_code(), "unsupported_content");
    }

    #[tokio::test]
    async fn preprocess_allows_refusal_on_assistant_role() {
        // On an assistant (output) role a refusal is a legitimate prior turn
        // (e.g. loaded from conversation history). Preprocessing must not
        // reject it; the downstream text extractor preserves it verbatim.
        let mut req = ResponsesRequest {
            input: ResponseInput::Items(vec![ResponseInputOutputItem::Message {
                id: "msg_1".to_string(),
                role: "assistant".to_string(),
                content: vec![ResponseContentPart::Refusal {
                    refusal: "previous refusal".to_string(),
                }],
                status: None,
                phase: None,
            }]),
            ..Default::default()
        };
        preprocess_responses_input(&mut req, None).await.unwrap();
    }

    #[tokio::test]
    async fn preprocess_walks_simple_input_message_arrays() {
        // SimpleInputMessage carries a StringOrContentParts::Array — the
        // preprocess pass must walk that array too.
        use openai_protocol::responses::SimpleInputMessageTypeTag;
        let mut req = ResponsesRequest {
            input: ResponseInput::Items(vec![ResponseInputOutputItem::SimpleInputMessage {
                role: "user".to_string(),
                content: StringOrContentParts::Array(vec![ResponseContentPart::InputImage {
                    detail: None,
                    file_id: Some("file_abc".to_string()),
                    image_url: None,
                }]),
                r#type: Some(SimpleInputMessageTypeTag::Message),
                phase: None,
            }]),
            ..Default::default()
        };
        let err = preprocess_responses_input(&mut req, None)
            .await
            .expect_err("file_id in SimpleInputMessage must be rejected too");
        assert_eq!(err.error_code(), "unsupported_content");
    }

    // ------------------------------------------------------------------
    // HTTP fetch tests — spin up a minimal axum server on an ephemeral
    // port so we can exercise the `file_url` branch end-to-end without
    // needing a crate-level HTTP mock fixture.
    // ------------------------------------------------------------------

    use axum::{body::Bytes as AxumBytes, extract::State, response::IntoResponse, routing::get};
    use llm_multimodal::{MediaConnector, MediaConnectorConfig};
    use std::net::SocketAddr;
    use tokio::{net::TcpListener, task::JoinHandle};

    struct MockFileServer {
        addr: SocketAddr,
        _handle: JoinHandle<()>,
    }

    impl MockFileServer {
        /// Spin up an axum server that serves `bytes` at `GET /file` on a
        /// random 127.0.0.1 port. Returns the server and its base URL.
        #[expect(
            clippy::disallowed_methods,
            reason = "test fixture — spawned task ends with the test"
        )]
        async fn start(bytes: Vec<u8>) -> Self {
            let state = Arc::new(bytes);
            let app = axum::Router::new()
                .route("/file", get(serve_file))
                .with_state(state);

            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            let handle = tokio::spawn(async move {
                axum::serve(listener, app).await.unwrap();
            });
            // Give the server a tick to bind.
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            Self {
                addr,
                _handle: handle,
            }
        }

        fn url(&self, path: &str) -> String {
            format!("http://{}{}", self.addr, path)
        }
    }

    async fn serve_file(State(bytes): State<Arc<Vec<u8>>>) -> impl IntoResponse {
        AxumBytes::from(bytes.as_ref().clone())
    }

    fn default_connector() -> Arc<MediaConnector> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .unwrap();
        Arc::new(MediaConnector::new(client, MediaConnectorConfig::default()).unwrap())
    }

    #[tokio::test]
    async fn preprocess_rewrites_input_file_url_jpeg() {
        // Serve a tiny JPEG-magic payload over an ephemeral port and
        // verify preprocess downloads it, sniffs the magic, and rewrites
        // the InputFile into an InputImage with a `data:image/jpeg` URL.
        let jpeg = vec![0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46];
        let server = MockFileServer::start(jpeg.clone()).await;
        let url = server.url("/file");
        let connector = default_connector();

        let mut req = user_message_with_parts(vec![ResponseContentPart::InputFile {
            detail: None,
            file_data: None,
            file_id: None,
            file_url: Some(url),
            filename: Some("photo.bin".to_string()),
        }]);

        preprocess_responses_input(&mut req, Some(&connector))
            .await
            .expect("jpeg file_url must be accepted");

        let parts = match &req.input {
            ResponseInput::Items(items) => match &items[0] {
                ResponseInputOutputItem::Message { content, .. } => content,
                other => panic!("unexpected item: {other:?}"),
            },
            ResponseInput::Text(_) => panic!("unexpected text input"),
        };
        match &parts[0] {
            ResponseContentPart::InputImage {
                image_url: Some(url),
                ..
            } => {
                assert!(
                    url.starts_with("data:image/jpeg;base64,"),
                    "expected rewritten to data URL, got: {url}"
                );
                let encoded = url.strip_prefix("data:image/jpeg;base64,").unwrap();
                let decoded = BASE64_STANDARD.decode(encoded).unwrap();
                assert_eq!(decoded, jpeg);
            }
            other => panic!("expected InputImage after file_url fetch, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn preprocess_rejects_input_file_url_pdf_pending_r4() {
        // file_url that resolves to a PDF must get the same R4-pending
        // rejection as base64 file_data with PDF magic.
        let pdf = b"%PDF-1.4\n%\xc7\xec\x8f\xa2\n".to_vec();
        let server = MockFileServer::start(pdf).await;
        let url = server.url("/file");
        let connector = default_connector();

        let mut req = user_message_with_parts(vec![ResponseContentPart::InputFile {
            detail: None,
            file_data: None,
            file_id: None,
            file_url: Some(url),
            filename: None,
        }]);

        let err = preprocess_responses_input(&mut req, Some(&connector))
            .await
            .expect_err("pdf file_url must be rejected");
        assert_eq!(err.error_code(), "unsupported_content");
        assert!(
            err.message().contains("R4"),
            "PDF error must cite R4, got: {}",
            err.message()
        );
    }

    #[tokio::test]
    async fn preprocess_rejects_input_file_url_non_image() {
        // file_url that resolves to neither an image nor a PDF must be
        // rejected rather than silently forwarded.
        let server = MockFileServer::start(b"arbitrary binary blob".to_vec()).await;
        let url = server.url("/file");
        let connector = default_connector();

        let mut req = user_message_with_parts(vec![ResponseContentPart::InputFile {
            detail: None,
            file_data: None,
            file_id: None,
            file_url: Some(url),
            filename: None,
        }]);

        let err = preprocess_responses_input(&mut req, Some(&connector))
            .await
            .expect_err("non-image file_url must be rejected");
        assert_eq!(err.error_code(), "unsupported_content");
    }
}
