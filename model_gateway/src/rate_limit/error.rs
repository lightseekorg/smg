use axum::{
    http::{self, header::RETRY_AFTER, HeaderValue},
    response::Response,
};

use super::local::TERMINAL_REJECTION_RETRY_AFTER_SECS;
use crate::routers::error::create_error;

pub fn rate_limit_exceeded_response(retry_after_secs: u64) -> Response {
    let mut response = create_error(
        http::StatusCode::TOO_MANY_REQUESTS,
        "tenant_rate_limit_exceeded",
        "Tenant rate limit exceeded for this request",
    );

    if retry_after_secs == TERMINAL_REJECTION_RETRY_AFTER_SECS {
        return response;
    }

    if let Ok(v) = HeaderValue::from_str(&retry_after_secs.max(1).to_string()) {
        response.headers_mut().insert(RETRY_AFTER, v);
    }

    response
}

#[cfg(test)]
mod tests {
    use axum::http::header::RETRY_AFTER;

    use super::{rate_limit_exceeded_response, TERMINAL_REJECTION_RETRY_AFTER_SECS};

    #[test]
    fn omits_retry_after_for_terminal_rate_limit_rejection() {
        let response = rate_limit_exceeded_response(TERMINAL_REJECTION_RETRY_AFTER_SECS);

        assert!(response.headers().get(RETRY_AFTER).is_none());
    }
}
