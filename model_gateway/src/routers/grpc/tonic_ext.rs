//! Extension traits for tonic gRPC types.

use axum::response::Response;
use http::StatusCode;
use tonic::Code;

use crate::routers::error;

/// Extension methods for `tonic::Status`.
pub(crate) trait TonicStatusExt {
    /// Map gRPC status code to the corresponding HTTP status code.
    fn http_status(&self) -> StatusCode;

    /// Returns `true` if this is a server-side error (should trip the circuit breaker).
    /// Client errors (InvalidArgument, NotFound, etc.) return `false`.
    fn is_server_error(&self) -> bool;

    /// Convert this gRPC error into an HTTP error response with the appropriate status code.
    fn to_http_error(&self, code: &str, msg: String) -> Response;
}

impl TonicStatusExt for tonic::Status {
    fn http_status(&self) -> StatusCode {
        match self.code() {
            Code::InvalidArgument | Code::FailedPrecondition | Code::OutOfRange => {
                StatusCode::BAD_REQUEST
            }
            Code::Unauthenticated => StatusCode::UNAUTHORIZED,
            Code::PermissionDenied => StatusCode::FORBIDDEN,
            Code::NotFound => StatusCode::NOT_FOUND,
            Code::AlreadyExists | Code::Aborted => StatusCode::CONFLICT,
            Code::ResourceExhausted => StatusCode::TOO_MANY_REQUESTS,
            Code::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
            Code::DeadlineExceeded => StatusCode::GATEWAY_TIMEOUT,
            // Internal, Unknown, Unimplemented, DataLoss, Cancelled
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    fn is_server_error(&self) -> bool {
        matches!(
            self.code(),
            Code::Internal
                | Code::Unavailable
                | Code::Unknown
                | Code::DataLoss
                | Code::DeadlineExceeded
        )
    }

    fn to_http_error(&self, code: &str, msg: String) -> Response {
        error::create_error(self.http_status(), code, msg)
    }
}

/// Extension for `Result<T, tonic::Status>` to check circuit breaker health.
pub(crate) trait TonicResultExt {
    /// Returns `true` if the result is healthy for the circuit breaker.
    /// `Ok` and client-error results are healthy; only server errors are failures.
    fn is_healthy(&self) -> bool;
}

impl<T> TonicResultExt for Result<T, tonic::Status> {
    fn is_healthy(&self) -> bool {
        self.as_ref()
            .map_or_else(|e| !e.is_server_error(), |_| true)
    }
}
