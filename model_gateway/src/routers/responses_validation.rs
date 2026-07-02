//! Shared request validation for the Responses API.
//!
//! Provides input normalization and validation shared across the HTTP and
//! WebSocket Responses paths. The HTTP `POST /v1/responses` route still uses
//! the generic `ValidatedJson` extractor; the WebSocket path calls
//! [`normalize_and_validate_responses_request`] directly on the per-event
//! request payload.

use openai_protocol::{
    responses::{ResponseInput, ResponseInputOutputItem, ResponsesRequest},
    validated::Normalizable,
};
use validator::{Validate, ValidationErrors, ValidationErrorsKind};

/// Normalize and validate a Responses request.
///
/// Tolerates the one OpenAI-spec relaxation that the strict schema rejects: a
/// continuation (`previous_response_id` set) whose `input` is composed solely
/// of `function_call_output` items and therefore has no leading user message.
pub(crate) fn normalize_and_validate_responses_request(
    request: &mut ResponsesRequest,
) -> Result<(), ValidationErrors> {
    request.normalize();

    match request.validate() {
        Ok(()) => Ok(()),
        Err(errors) if allows_function_call_output_only_continuation(request, &errors) => Ok(()),
        Err(errors) => Err(errors),
    }
}

/// Whether the only validation failure is the missing-user-message rule on a
/// `previous_response_id` continuation that supplies just function-call outputs.
fn allows_function_call_output_only_continuation(
    request: &ResponsesRequest,
    errors: &ValidationErrors,
) -> bool {
    if request.previous_response_id.is_none() {
        return false;
    }

    let ResponseInput::Items(items) = &request.input else {
        return false;
    };

    if items.is_empty() {
        return false;
    }

    if !items
        .iter()
        .all(|item| matches!(item, ResponseInputOutputItem::FunctionCallOutput { .. }))
    {
        return false;
    }

    let all_errors = errors.errors();
    if all_errors.len() != 1 {
        return false;
    }

    let Some(schema_errors) = all_errors.get("__all__") else {
        return false;
    };

    let ValidationErrorsKind::Field(schema_errors) = schema_errors else {
        return false;
    };

    schema_errors.len() == 1 && schema_errors[0].code == "input_missing_user_message"
}

#[cfg(test)]
mod tests {
    use openai_protocol::responses::{ResponseInput, ResponseInputOutputItem, ResponsesRequest};

    use super::normalize_and_validate_responses_request;

    #[test]
    fn allows_function_call_output_only_continuation_with_previous_response_id() {
        let mut request = ResponsesRequest {
            model: "mock-model".to_string(),
            previous_response_id: Some("resp_prev".to_string()),
            input: ResponseInput::Items(vec![ResponseInputOutputItem::FunctionCallOutput {
                id: None,
                call_id: "call_123".to_string(),
                output: r#"{"result":345}"#.to_string(),
                status: None,
            }]),
            ..Default::default()
        };

        assert!(normalize_and_validate_responses_request(&mut request).is_ok());
    }

    #[test]
    fn still_rejects_function_call_output_only_without_previous_response_id() {
        let mut request = ResponsesRequest {
            model: "mock-model".to_string(),
            input: ResponseInput::Items(vec![ResponseInputOutputItem::FunctionCallOutput {
                id: None,
                call_id: "call_123".to_string(),
                output: r#"{"result":345}"#.to_string(),
                status: None,
            }]),
            ..Default::default()
        };

        assert!(normalize_and_validate_responses_request(&mut request).is_err());
    }
}
