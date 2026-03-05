//! Extension traits for tonic gRPC types.

use axum::response::Response;
use http::StatusCode;
use tonic::Code;

use crate::{core::is_retryable_status, routers::error};

/// Extension methods for `tonic::Status`.
pub(crate) trait TonicStatusExt {
    /// Map gRPC status code to the corresponding HTTP status code.
    fn http_status(&self) -> StatusCode;

    /// Returns `true` if this error should trip the circuit breaker.
    ///
    /// Delegates to `is_retryable_status(http_status())` so the circuit breaker
    /// uses the same predicate for both HTTP and gRPC paths. This covers
    /// 408, 429, 500, 502, 503, 504 (i.e. ResourceExhausted, DeadlineExceeded,
    /// Internal, Unavailable, Unknown, DataLoss, etc.).
    fn is_cb_failure(&self) -> bool;

    /// Convert this gRPC error into an HTTP error response with the appropriate status code.
    fn to_http_error(&self, code: &str, msg: String) -> Response;
}

impl TonicStatusExt for tonic::Status {
    fn http_status(&self) -> StatusCode {
        match self.code() {
            Code::Ok => StatusCode::OK,
            Code::InvalidArgument
            | Code::FailedPrecondition
            | Code::OutOfRange
            | Code::Cancelled => StatusCode::BAD_REQUEST,
            Code::Unauthenticated => StatusCode::UNAUTHORIZED,
            Code::PermissionDenied => StatusCode::FORBIDDEN,
            Code::NotFound => StatusCode::NOT_FOUND,
            Code::AlreadyExists | Code::Aborted => StatusCode::CONFLICT,
            Code::ResourceExhausted => StatusCode::TOO_MANY_REQUESTS,
            Code::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
            Code::DeadlineExceeded => StatusCode::GATEWAY_TIMEOUT,
            Code::Unimplemented => StatusCode::NOT_IMPLEMENTED,
            // Internal, Unknown, DataLoss
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    fn is_cb_failure(&self) -> bool {
        is_retryable_status(self.http_status())
    }

    fn to_http_error(&self, code: &str, msg: String) -> Response {
        error::create_error(self.http_status(), code, msg)
    }
}

/// Extension for `Result<T, tonic::Status>` to check circuit breaker health.
pub(crate) trait TonicResultExt {
    /// Returns `true` if the result is healthy for the circuit breaker.
    /// `Ok` and client-error results are healthy; only retryable errors are failures.
    fn is_healthy(&self) -> bool;
}

impl<T> TonicResultExt for Result<T, tonic::Status> {
    fn is_healthy(&self) -> bool {
        self.as_ref().map_or_else(|e| !e.is_cb_failure(), |_| true)
    }
}
