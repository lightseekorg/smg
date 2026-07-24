//! Builds the tenant-rate-limit rejection response: `429`, the standard
//! SMG error envelope, and `Retry-After` when a wait time is known.

use axum::{
    http::{header::RETRY_AFTER, HeaderValue},
    response::Response,
};

use crate::routers::error;

pub const RATE_LIMIT_ERROR_CODE: &str = "tenant_rate_limit_exceeded";

/// `retry_after_secs` of `0` omits the `Retry-After` header (nothing to
/// wait for is a caller bug, not a real rejection state).
pub fn rejection_response(retry_after_secs: u64) -> Response {
    let mut response = error::too_many_requests(
        RATE_LIMIT_ERROR_CODE,
        "Tenant rate limit exceeded for this request",
    );
    if retry_after_secs > 0 {
        if let Ok(value) = HeaderValue::from_str(&retry_after_secs.to_string()) {
            response.headers_mut().insert(RETRY_AFTER, value);
        }
    }
    response
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;

    use super::*;

    #[test]
    fn sets_status_code_and_retry_after() {
        let response = rejection_response(30);
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(response.headers().get(RETRY_AFTER).unwrap(), "30");
        assert_eq!(
            response
                .headers()
                .get(error::HEADER_X_SMG_ERROR_CODE)
                .unwrap(),
            RATE_LIMIT_ERROR_CODE
        );
    }

    #[test]
    fn zero_retry_after_omits_header() {
        let response = rejection_response(0);
        assert!(response.headers().get(RETRY_AFTER).is_none());
    }
}
