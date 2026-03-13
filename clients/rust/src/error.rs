use openai_protocol::common::ErrorResponse;

/// All errors that can occur when using the SMG client.
#[derive(Debug, thiserror::Error)]
pub enum SmgError {
    /// HTTP transport error (connection, DNS, TLS, timeout).
    #[error("connection error: {0}")]
    Connection(#[from] reqwest::Error),

    /// Server returned a 400 Bad Request.
    #[error("bad request ({status}): {message}")]
    BadRequest {
        message: String,
        status: u16,
        body: Option<ErrorResponse>,
    },

    /// Server returned a 401 Unauthorized.
    #[error("authentication error (401): {message}")]
    Authentication {
        message: String,
        status: u16,
        body: Option<ErrorResponse>,
    },

    /// Server returned a 403 Forbidden.
    #[error("permission denied (403): {message}")]
    PermissionDenied {
        message: String,
        status: u16,
        body: Option<ErrorResponse>,
    },

    /// Server returned a 404 Not Found.
    #[error("not found (404): {message}")]
    NotFound {
        message: String,
        status: u16,
        body: Option<ErrorResponse>,
    },

    /// Server returned a 429 Too Many Requests.
    #[error("rate limited (429): {message}")]
    RateLimit {
        message: String,
        status: u16,
        body: Option<ErrorResponse>,
    },

    /// Server returned a 500+ error.
    #[error("server error ({status}): {message}")]
    Server {
        message: String,
        status: u16,
        body: Option<ErrorResponse>,
    },

    /// Failed to deserialize the response body.
    #[error("deserialization error: {0}")]
    Deserialization(#[from] serde_json::Error),

    /// SSE stream ended unexpectedly or contained invalid data.
    #[error("stream error: {0}")]
    Stream(String),
}

impl SmgError {
    /// Build the appropriate error variant from an HTTP status code and body.
    pub fn from_status(status: u16, body: &str) -> Self {
        let parsed: Option<ErrorResponse> = serde_json::from_str(body).ok();
        let message = parsed
            .as_ref()
            .map(|e| e.error.message.clone())
            .unwrap_or_else(|| body.to_string());

        match status {
            400 => Self::BadRequest {
                message,
                status,
                body: parsed,
            },
            401 => Self::Authentication {
                message,
                status,
                body: parsed,
            },
            403 => Self::PermissionDenied {
                message,
                status,
                body: parsed,
            },
            404 => Self::NotFound {
                message,
                status,
                body: parsed,
            },
            429 => Self::RateLimit {
                message,
                status,
                body: parsed,
            },
            // Catch-all for other 4xx client errors (e.g. 422).
            402 | 405..=428 | 430..=499 => Self::BadRequest {
                message,
                status,
                body: parsed,
            },
            _ => Self::Server {
                message,
                status,
                body: parsed,
            },
        }
    }
}
