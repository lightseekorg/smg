//! Extension traits for tonic gRPC types.

use axum::response::Response;
use http::StatusCode;
use tonic::Code;

use crate::routers::error;

/// Extension methods for `tonic::Status`.
pub(crate) trait TonicStatusExt {
    /// Map gRPC status code to the corresponding HTTP status code.
    fn http_status(&self) -> StatusCode;

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

    fn to_http_error(&self, code: &str, msg: String) -> Response {
        error::create_error(self.http_status(), code, msg)
    }
}

/// Extension for `Result<T, tonic::Status>` to extract HTTP status for CB recording.
pub(crate) trait TonicResultExt {
    /// Returns the HTTP status code for circuit breaker recording.
    /// `Ok` → 200, `Err(status)` → mapped HTTP status code.
    fn cb_status_code(&self) -> u16;
}

impl<T> TonicResultExt for Result<T, tonic::Status> {
    fn cb_status_code(&self) -> u16 {
        self.as_ref()
            .map_or_else(|e| e.http_status().as_u16(), |_| 200)
    }
}
