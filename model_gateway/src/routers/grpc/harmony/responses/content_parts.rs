//! Harmony Responses content-part validation (R3).
//!
//! The harmony Responses pipeline builds gpt-oss prompt tokens directly via
//! [`openai_harmony`]; unlike the regular gRPC Responses path, it never
//! converts to Chat Completions, so the shared multimodal wire in
//! [`crate::routers::grpc::multimodal`] is *not* reachable from this router.
//! The Chat harmony path in `harmony/stages/request_building.rs` explicitly
//! passes `None` for multimodal on every backend, which matches the fact
//! that gpt-oss was not trained against image/file tokens at the
//! `<|message|>` level.
//!
//! Prior to R3 the harmony builder silently dropped `InputImage`,
//! `InputFile`, and (indirectly) `Refusal` content parts when rendering the
//! harmony prompt (see the `None` arms at `builder.rs` circa the pre-R3
//! `parse_response_item_to_harmony_message` / `SimpleInputMessage` paths).
//! Silent data loss on multimodal prompts is a much worse failure mode
//! than a clear 400: the caller believes the model saw their image, the
//! model responds from text context alone, and the discrepancy is
//! invisible at every observable layer.
//!
//! R3 upgrades that silent drop into an explicit `400 unsupported_content`
//! at the router entry point so callers get the same P1 parity R1/R2 give
//! on their respective surfaces. This module is the single source of
//! truth for the rejection policy — both the non-streaming and streaming
//! entry points in this directory call [`validate_harmony_responses_input`]
//! before any pipeline dispatch.
//!
//! Scope of rejection (per R3 task spec):
//!
//! - `InputImage` on a user/system/developer/simple message → 400
//!   `unsupported_content`. Applies regardless of whether the payload
//!   references `file_id`, `image_url` (absolute or `data:` URL). Harmony
//!   cannot forward the image to a backend capable of consuming it.
//! - `InputFile` on any fresh input message → 400 `unsupported_content`.
//!   The error code specialises on `file_data` payloads whose magic bytes
//!   match `%PDF-`: those return the PDF-specific "pending R4" message so
//!   the caller understands it is a known gap, not a generic rejection.
//! - `Refusal` on a *non-assistant* role → 400 `unsupported_content`. On
//!   `assistant` roles the part is a legitimate replay of a prior model
//!   refusal (the P2-refined assistant-message variant) and is left alone
//!   for the builder to render as text — rejecting it there would break
//!   multi-turn conversations that faithfully echo the prior assistant
//!   output.
//!
//! Text-only parts (`InputText`, `OutputText`) and assistant-role
//! `Refusal` replays pass through untouched; the builder already renders
//! them correctly. PDF text extraction itself remains Task R4 and is
//! explicitly out of scope here.

use axum::response::Response;
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use openai_protocol::responses::{
    ResponseContentPart, ResponseInput, ResponseInputOutputItem, ResponsesRequest,
    StringOrContentParts,
};

use crate::routers::error;

/// Magic-byte prefix for PDF files (`%PDF-`). PDFs may include a small
/// leading BOM or whitespace but the spec requires the `%PDF-` token
/// within the first 1024 bytes; we only sniff the decoded prefix of the
/// content-part payload here, which is sufficient to catch the common
/// case where a PDF is base64-encoded without additional wrapping.
const PDF_MAGIC: &[u8] = b"%PDF-";

/// Validate that the user-supplied input in a ResponsesRequest contains
/// only content parts the harmony Responses router can faithfully render.
///
/// Returns `Ok(())` when the request is safe to dispatch, or a
/// `400 unsupported_content` response otherwise. The error message is
/// deliberately descriptive so callers can distinguish the rejection
/// reason (image, file, PDF, refusal, file_id) at the HTTP layer.
///
/// Only the caller-submitted `request.input` is inspected. Persisted
/// history loaded via `previous_response_id` is skipped: it was
/// validated when it was first submitted, may legitimately carry
/// assistant-refusal replays, and is not something the caller can
/// mutate on this request.
#[expect(
    clippy::result_large_err,
    reason = "axum Response is the standard error type on the router-entry boundary; \
              matches the rest of the harmony Responses entry points"
)]
pub(super) fn validate_harmony_responses_input(
    request: &ResponsesRequest,
) -> Result<(), Response> {
    let items = match &request.input {
        ResponseInput::Text(_) => return Ok(()),
        ResponseInput::Items(items) => items,
    };

    for item in items {
        match item {
            ResponseInputOutputItem::Message { role, content, .. } => {
                validate_content_parts(role.as_str(), content)?;
            }
            ResponseInputOutputItem::SimpleInputMessage {
                role,
                content: StringOrContentParts::Array(parts),
                ..
            } => {
                validate_content_parts(role.as_str(), parts)?;
            }
            // The `StringOrContentParts::String` variant has no
            // content parts to inspect; and non-message items
            // (function calls, reasoning, tool outputs, etc.) never
            // carry `ResponseContentPart` today so there is nothing
            // to reject here. The harmony builder handles unsupported
            // item variants with its own `Unsupported input item type`
            // errors at the prep stage.
            _ => {}
        }
    }

    Ok(())
}

/// Apply the rejection policy to a single message's content-parts list.
#[expect(
    clippy::result_large_err,
    reason = "axum Response is the standard error type on the router-entry boundary; \
              matches the rest of the harmony Responses entry points"
)]
fn validate_content_parts(role: &str, parts: &[ResponseContentPart]) -> Result<(), Response> {
    for part in parts {
        match part {
            ResponseContentPart::InputText { .. } | ResponseContentPart::OutputText { .. } => {
                // Text parts — the builder already renders these.
            }
            ResponseContentPart::InputImage {
                image_url, file_id, ..
            } => {
                // file_id is always unsupported (no Files API backend in
                // harmony gRPC mode). image_url (absolute or data URL) is
                // also unsupported because the harmony pipeline does not
                // thread multimodal bytes to the backend — the chat
                // harmony path itself passes `None` for multimodal. See
                // `harmony/stages/request_building.rs` for the
                // corresponding dispatch code.
                let detail = if file_id.is_some() {
                    "input_image content parts with file_id are not supported on the harmony Responses \
                     backend (no Files API resolver in gRPC mode)"
                } else if let Some(url) = image_url {
                    if is_data_url(url) {
                        "input_image content parts with a data: URL are not supported on the harmony \
                         Responses backend (the gpt-oss pipeline has no multimodal wire)"
                    } else {
                        "input_image content parts with absolute image_url are not supported on the \
                         harmony Responses backend (the gpt-oss pipeline has no multimodal wire)"
                    }
                } else {
                    "input_image content parts are not supported on the harmony Responses backend \
                     (the gpt-oss pipeline has no multimodal wire)"
                };
                return Err(unsupported(detail));
            }
            ResponseContentPart::InputFile {
                file_data,
                file_id,
                file_url,
                ..
            } => {
                // `file_data` PDF payloads get the R4-specific message
                // so callers can correlate a known gap instead of a
                // blanket rejection. We only sniff magic bytes on the
                // decoded prefix (spec-shaped `data:application/pdf;...`
                // prefixes may or may not be present — the magic bytes
                // are authoritative).
                if let Some(data) = file_data.as_deref() {
                    if looks_like_pdf(data) {
                        return Err(unsupported(
                            "input_file content parts carrying PDF data (magic bytes `%PDF-`) are \
                             not yet supported on the harmony Responses backend. PDF text \
                             extraction is pending R4.",
                        ));
                    }
                    return Err(unsupported(
                        "input_file content parts with file_data are not supported on the harmony \
                         Responses backend",
                    ));
                }
                if file_id.is_some() {
                    return Err(unsupported(
                        "input_file content parts with file_id are not supported on the harmony \
                         Responses backend (no Files API resolver in gRPC mode)",
                    ));
                }
                if file_url.is_some() {
                    return Err(unsupported(
                        "input_file content parts with file_url are not supported on the harmony \
                         Responses backend",
                    ));
                }
                return Err(unsupported(
                    "input_file content parts are not supported on the harmony Responses backend",
                ));
            }
            ResponseContentPart::Refusal { .. } => {
                // Only assistant-role replays are allowed — those
                // legitimately echo a prior model refusal in multi-turn
                // history. Any other role carrying a refusal is a new
                // input that the harmony backend cannot consume.
                if role != "assistant" {
                    return Err(unsupported(
                        "refusal content parts are only allowed on assistant-role messages \
                         (e.g. replaying a prior response); other roles cannot carry refusals \
                         on the harmony Responses backend",
                    ));
                }
            }
        }
    }

    Ok(())
}

/// Detect a `data:` URL prefix. Matches `data:` case-insensitively to
/// mirror RFC 2397 permissiveness without pulling in a dedicated parser.
fn is_data_url(url: &str) -> bool {
    url.len() >= 5 && url.as_bytes()[..5].eq_ignore_ascii_case(b"data:")
}

/// Number of base64 characters that decode to at least [`PDF_MAGIC`]'s
/// length (5 bytes). base64 encodes 3 bytes per 4 chars, so 8 chars
/// decode to 6 bytes — the smallest multiple-of-4 window that still
/// covers the magic-byte prefix. Sniffing only this window avoids the
/// DOS-adjacent case where a 500 MB rejected `file_data` would otherwise
/// be fully base64-decoded just to check its leading bytes (see
/// `max_payload_size` default in `model_gateway/src/config/types.rs`).
const PDF_SNIFF_BASE64_LEN: usize = 8;
/// Decoded buffer size needed for [`PDF_SNIFF_BASE64_LEN`] base64
/// characters. base64 decodes 4 input chars to 3 output bytes.
const PDF_SNIFF_DECODED_LEN: usize = PDF_SNIFF_BASE64_LEN / 4 * 3;

/// Return `true` if the decoded prefix of `file_data` contains the PDF
/// magic bytes `%PDF-`. Accepts either a raw base64 payload or a
/// `data:<mime>;base64,<payload>` wrapper for robustness against
/// clients that send the latter in a `file_data` field.
///
/// Only decodes [`PDF_SNIFF_BASE64_LEN`] base64 chars into a fixed-size
/// stack buffer; the full payload is never materialized. This matters
/// because the router's default `max_payload_size` permits multi-hundred
/// megabyte `file_data` inputs that would otherwise be fully decoded
/// just to be rejected.
fn looks_like_pdf(file_data: &str) -> bool {
    let payload = strip_data_url_prefix(file_data).trim_start();
    let Some(prefix) = payload.get(..PDF_SNIFF_BASE64_LEN) else {
        // Input shorter than the sniff window — safe to fall through
        // to the generic "file_data unsupported" message. An 8-byte
        // `file_data` payload can't be a real PDF anyway.
        return false;
    };
    let mut decoded = [0u8; PDF_SNIFF_DECODED_LEN];
    let Ok(len) = BASE64_STANDARD.decode_slice(prefix.as_bytes(), &mut decoded) else {
        // Malformed base64 is a separate rejection kind; fall through
        // so callers see the generic "file_data unsupported" message.
        return false;
    };
    decoded[..len].starts_with(PDF_MAGIC)
}

/// Strip a `data:<mime>;base64,` prefix from `input` if present, returning
/// the remainder; otherwise return the input unchanged. Case-insensitive
/// on the `;base64,` separator to stay consistent with [`is_data_url`],
/// which is case-insensitive on the `data:` scheme per RFC 2397 §3; without
/// this, a payload like `data:application/pdf;Base64,JVBERi...` would pass
/// the data-URL gate but miss the strip, falling through to the generic
/// "file_data unsupported" message instead of the PDF-specific R4 message.
fn strip_data_url_prefix(input: &str) -> &str {
    if !is_data_url(input) {
        return input;
    }
    match find_base64_separator(input) {
        Some(idx) => &input[idx + ";base64,".len()..],
        // `data:<text>,<payload>` (non-base64) is not something we can
        // meaningfully magic-byte sniff; return the full input so the
        // caller falls through to the generic file-unsupported path.
        None => input,
    }
}

/// Locate the `;base64,` separator inside a data URL, matching
/// case-insensitively. Uses a byte-window walk (no allocation) so the
/// hot path stays allocation-free even on large payloads.
fn find_base64_separator(input: &str) -> Option<usize> {
    const NEEDLE: &[u8] = b";base64,";
    input
        .as_bytes()
        .windows(NEEDLE.len())
        .position(|window| window.eq_ignore_ascii_case(NEEDLE))
}

/// Build the standard `400 unsupported_content` response used by every
/// rejection branch in this module.
fn unsupported(message: &str) -> Response {
    error::bad_request("unsupported_content", message.to_string())
}

#[cfg(test)]
mod tests {
    //! Unit coverage for R3. These tests lock in the rejection policy and
    //! assert every branch of the content-part taxonomy the harmony
    //! backend cannot execute — so the silent-drop regression that R3
    //! fixed cannot quietly come back.

    use axum::{body::to_bytes, http::StatusCode};
    use openai_protocol::{
        common::Detail,
        responses::{
            FileDetail, ResponseContentPart, ResponseInput, ResponseInputOutputItem,
            ResponsesRequest, StringOrContentParts,
        },
    };
    use serde_json::Value;

    use super::*;
    use crate::routers::error::HEADER_X_SMG_ERROR_CODE;

    /// Extract the JSON error message body from a validation error
    /// response. Tests use this instead of only checking status +
    /// `X-SMG-Error-Code` so that PDF-specific and generic rejections
    /// are distinguishable at the message-body level — otherwise both
    /// shapes satisfy the same status/code assertion and the
    /// R4-specific path would be untested.
    async fn error_message(response: Response) -> String {
        let (_parts, body) = response.into_parts();
        let bytes = to_bytes(body, usize::MAX)
            .await
            .expect("error body fits in memory");
        let value: Value =
            serde_json::from_slice(&bytes).expect("error body is always valid JSON");
        value
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(|m| m.as_str())
            .map(str::to_owned)
            .expect("error.message is always present on a bad_request response")
    }

    fn request_with_items(items: Vec<ResponseInputOutputItem>) -> ResponsesRequest {
        ResponsesRequest {
            model: "gpt-oss".to_string(),
            input: ResponseInput::Items(items),
            ..Default::default()
        }
    }

    fn user_message(content: Vec<ResponseContentPart>) -> ResponseInputOutputItem {
        ResponseInputOutputItem::Message {
            id: "msg_test".to_string(),
            role: "user".to_string(),
            content,
            status: Some("completed".to_string()),
            phase: None,
        }
    }

    fn assistant_message(content: Vec<ResponseContentPart>) -> ResponseInputOutputItem {
        ResponseInputOutputItem::Message {
            id: "msg_test_asst".to_string(),
            role: "assistant".to_string(),
            content,
            status: Some("completed".to_string()),
            phase: None,
        }
    }

    fn assert_is_unsupported(response: Response) {
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let header = response
            .headers()
            .get(HEADER_X_SMG_ERROR_CODE)
            .expect("error code header is always set");
        assert_eq!(header.to_str().unwrap(), "unsupported_content");
    }

    #[test]
    fn text_only_input_is_accepted() {
        // Baseline: `ResponseInput::Text` never carries non-text parts
        // so it must always validate. Guards against accidental
        // overreach in the validator.
        let request = ResponsesRequest {
            model: "gpt-oss".to_string(),
            input: ResponseInput::Text("hello".to_string()),
            ..Default::default()
        };
        validate_harmony_responses_input(&request).expect("text-only request must pass");
    }

    #[test]
    fn pure_input_text_items_are_accepted() {
        let request = request_with_items(vec![user_message(vec![ResponseContentPart::InputText {
            text: "Describe the weather.".to_string(),
        }])]);
        validate_harmony_responses_input(&request).expect("InputText must pass");
    }

    #[test]
    fn input_image_data_url_is_rejected() {
        // R3 scope item: InputImage with a `data:` URL. The pre-R3
        // builder dropped this silently; R3 mandates a 400.
        let request = request_with_items(vec![user_message(vec![ResponseContentPart::InputImage {
            detail: Some(Detail::Auto),
            file_id: None,
            image_url: Some("data:image/jpeg;base64,/9j/4AAQSkZJRg==".to_string()),
        }])]);
        let err =
            validate_harmony_responses_input(&request).expect_err("data-URL images must be rejected");
        assert_is_unsupported(err);
    }

    #[test]
    fn input_image_absolute_url_is_rejected() {
        let request = request_with_items(vec![user_message(vec![ResponseContentPart::InputImage {
            detail: None,
            file_id: None,
            image_url: Some("https://example.com/dog.jpg".to_string()),
        }])]);
        let err = validate_harmony_responses_input(&request)
            .expect_err("absolute-URL images must be rejected");
        assert_is_unsupported(err);
    }

    #[test]
    fn input_image_file_id_is_rejected() {
        // R3 scope item: file_id is always unsupported in gRPC mode
        // (no Files API resolver).
        let request = request_with_items(vec![user_message(vec![ResponseContentPart::InputImage {
            detail: None,
            file_id: Some("file-abc".to_string()),
            image_url: None,
        }])]);
        let err = validate_harmony_responses_input(&request)
            .expect_err("file_id images must be rejected");
        assert_is_unsupported(err);
    }

    #[tokio::test]
    async fn input_file_pdf_magic_bytes_are_rejected_with_r4_message() {
        // `%PDF-1.4\n` base64-encodes to `JVBERi0xLjQK`. Asserts the
        // PDF-specific error path so callers know the gap is tracked
        // under R4 rather than a blanket rejection. We assert on the
        // message body (not just status/code) so the PDF-specific
        // branch is distinguishable from a generic file-data rejection.
        let request = request_with_items(vec![user_message(vec![ResponseContentPart::InputFile {
            detail: Some(FileDetail::High),
            file_data: Some("JVBERi0xLjQK".to_string()),
            file_id: None,
            file_url: None,
            filename: Some("report.pdf".to_string()),
        }])]);
        let err =
            validate_harmony_responses_input(&request).expect_err("PDF file_data must be rejected");
        assert_eq!(err.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            err.headers()
                .get(HEADER_X_SMG_ERROR_CODE)
                .and_then(|h| h.to_str().ok()),
            Some("unsupported_content"),
        );
        let message = error_message(err).await;
        assert!(
            message.contains("PDF") && message.contains("R4"),
            "PDF rejections must cite R4 so callers can correlate the paused task; got: {message}"
        );
    }

    #[tokio::test]
    async fn input_file_non_pdf_file_data_is_rejected() {
        // A JPEG magic-byte prefix (FFD8FFE0 ... base64 `/9j/4A==`) is
        // not a PDF; it must still be rejected because the harmony
        // pipeline cannot route it — but *not* with the R4 message.
        // Asserts the generic file-data branch is taken so we do not
        // misdirect clients to the paused R4 task.
        let request = request_with_items(vec![user_message(vec![ResponseContentPart::InputFile {
            detail: None,
            file_data: Some("/9j/4AAQSkZJRg==".to_string()),
            file_id: None,
            file_url: None,
            filename: Some("photo.jpg".to_string()),
        }])]);
        let err = validate_harmony_responses_input(&request)
            .expect_err("JPEG file_data must be rejected");
        let message = error_message(err).await;
        assert!(
            !message.contains("R4"),
            "non-PDF file_data must not cite R4; got: {message}"
        );
        assert!(
            message.contains("file_data"),
            "generic file-data rejections must mention file_data; got: {message}"
        );
    }

    #[test]
    fn input_file_file_url_is_rejected() {
        let request = request_with_items(vec![user_message(vec![ResponseContentPart::InputFile {
            detail: None,
            file_data: None,
            file_id: None,
            file_url: Some("https://example.com/report.pdf".to_string()),
            filename: None,
        }])]);
        let err = validate_harmony_responses_input(&request)
            .expect_err("file_url attachments must be rejected");
        assert_is_unsupported(err);
    }

    #[test]
    fn input_file_file_id_is_rejected() {
        let request = request_with_items(vec![user_message(vec![ResponseContentPart::InputFile {
            detail: None,
            file_data: None,
            file_id: Some("file-abc".to_string()),
            file_url: None,
            filename: None,
        }])]);
        let err = validate_harmony_responses_input(&request)
            .expect_err("file_id attachments must be rejected");
        assert_is_unsupported(err);
    }

    #[test]
    fn refusal_on_user_role_is_rejected() {
        // Refusals on user/system/developer messages are nonsensical
        // inputs for the harmony pipeline. R3 rejects them explicitly
        // instead of letting the builder silently fold the text into
        // the prompt.
        let request = request_with_items(vec![user_message(vec![ResponseContentPart::Refusal {
            refusal: "I cannot process that request.".to_string(),
        }])]);
        let err = validate_harmony_responses_input(&request)
            .expect_err("user-role refusals must be rejected");
        assert_is_unsupported(err);
    }

    #[test]
    fn refusal_on_assistant_role_is_accepted() {
        // Assistant-role replays legitimately carry a prior refusal
        // when the client rehydrates a multi-turn conversation. The
        // builder already renders them correctly as text, so R3 must
        // not regress this.
        let request = request_with_items(vec![assistant_message(vec![
            ResponseContentPart::Refusal {
                refusal: "I cannot process that request.".to_string(),
            },
        ])]);
        validate_harmony_responses_input(&request)
            .expect("assistant-role refusals must still be accepted for multi-turn replay");
    }

    #[test]
    fn simple_input_message_with_content_parts_is_validated() {
        // SimpleInputMessage uses the `StringOrContentParts::Array`
        // shape; the validator must descend into it symmetrically with
        // the `Message` case. Regression guard: before R3 the builder
        // silently dropped images on both variants.
        let request = request_with_items(vec![ResponseInputOutputItem::SimpleInputMessage {
            content: StringOrContentParts::Array(vec![ResponseContentPart::InputImage {
                detail: None,
                file_id: None,
                image_url: Some("https://example.com/cat.jpg".to_string()),
            }]),
            role: "user".to_string(),
            r#type: None,
            phase: None,
        }]);
        let err = validate_harmony_responses_input(&request)
            .expect_err("SimpleInputMessage.Array images must be rejected");
        assert_is_unsupported(err);
    }

    #[test]
    fn simple_input_message_string_variant_is_accepted() {
        // Sibling of the Array case above; must not regress the bare
        // `StringOrContentParts::String` path.
        let request = request_with_items(vec![ResponseInputOutputItem::SimpleInputMessage {
            content: StringOrContentParts::String("Describe the weather.".to_string()),
            role: "user".to_string(),
            r#type: None,
            phase: None,
        }]);
        validate_harmony_responses_input(&request)
            .expect("SimpleInputMessage.String must pass unchanged");
    }

    #[tokio::test]
    async fn data_url_prefix_pdf_magic_still_rejected_with_r4_message() {
        // Robustness: if a client wraps the base64 payload in a
        // `data:application/pdf;base64,` envelope inside `file_data`
        // (non-canonical but observed in the wild), the magic-byte
        // sniff must still catch it so the caller sees the R4-specific
        // message rather than a generic rejection.
        let request = request_with_items(vec![user_message(vec![ResponseContentPart::InputFile {
            detail: None,
            file_data: Some("data:application/pdf;base64,JVBERi0xLjQK".to_string()),
            file_id: None,
            file_url: None,
            filename: None,
        }])]);
        let err = validate_harmony_responses_input(&request)
            .expect_err("data-URL-wrapped PDF file_data must be rejected");
        let message = error_message(err).await;
        assert!(
            message.contains("PDF") && message.contains("R4"),
            "data-URL-wrapped PDF must reach the R4-specific message; got: {message}"
        );
    }

    #[tokio::test]
    async fn data_url_prefix_is_case_insensitive_on_base64_separator() {
        // `is_data_url` matches `data:` case-insensitively, so
        // `strip_data_url_prefix` must match `;base64,` with the same
        // flexibility. Before the R3 follow-up both `Data:` and
        // `;Base64,` would skip the strip, mis-routing the request into
        // the generic "file_data unsupported" path instead of the
        // PDF-specific R4 message.
        let request = request_with_items(vec![user_message(vec![ResponseContentPart::InputFile {
            detail: None,
            file_data: Some("Data:application/pdf;Base64,JVBERi0xLjQK".to_string()),
            file_id: None,
            file_url: None,
            filename: None,
        }])]);
        let err = validate_harmony_responses_input(&request)
            .expect_err("mixed-case data-URL wrapper must still reach the PDF sniffer");
        let message = error_message(err).await;
        assert!(
            message.contains("PDF") && message.contains("R4"),
            "mixed-case data-URL wrapper must reach the R4-specific message; got: {message}"
        );
    }

    #[tokio::test]
    async fn pdf_sniff_decodes_only_magic_prefix_not_full_payload() {
        // Regression guard for the bounded-decode path: a large PDF
        // `file_data` must still route to the R4-specific message
        // without the validator materializing the entire payload. We
        // construct a payload whose first 6 bytes base64-decode to
        // `%PDF-1` and pad the remainder with a long run of `A`s; the
        // prefix-only decoder reaches the magic bytes before looking
        // at anything beyond the first 8 base64 chars.
        //
        // `%PDF-1` → base64 `JVBERi0x` (the canonical PDF header).
        let mut payload = String::from("JVBERi0x");
        payload.push_str(&"A".repeat(65_536));
        let request = request_with_items(vec![user_message(vec![ResponseContentPart::InputFile {
            detail: None,
            file_data: Some(payload),
            file_id: None,
            file_url: None,
            filename: None,
        }])]);
        let err = validate_harmony_responses_input(&request)
            .expect_err("large PDF file_data must still be rejected with the R4 message");
        let message = error_message(err).await;
        assert!(
            message.contains("PDF") && message.contains("R4"),
            "bounded-decode path must still emit the R4-specific message; got: {message}"
        );
    }

    #[tokio::test]
    async fn refusal_role_rejection_cites_assistant_exception() {
        // The non-assistant refusal branch has a distinct
        // human-readable message; lock it in so future refactors do
        // not regress the rationale the caller sees.
        let request = request_with_items(vec![user_message(vec![ResponseContentPart::Refusal {
            refusal: "I cannot process that.".to_string(),
        }])]);
        let err = validate_harmony_responses_input(&request)
            .expect_err("user-role refusals must be rejected");
        let message = error_message(err).await;
        assert!(
            message.contains("refusal") && message.contains("assistant"),
            "refusal rejection must name the assistant-role exception; got: {message}"
        );
    }
}
