"""Error types for the SMG client."""

from __future__ import annotations


class SmgError(Exception):
    """Base exception for all SMG client errors."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class ApiError(SmgError):
    """Error returned by the SMG API."""

    def __init__(
        self,
        message: str,
        status_code: int,
        error_type: str | None = None,
        param: str | None = None,
        code: str | None = None,
        body: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.error_type = error_type
        self.param = param
        self.code = code
        self.body = body
        super().__init__(message)


class BadRequestError(ApiError):
    """400 Bad Request."""


class AuthenticationError(ApiError):
    """401 Unauthorized."""


class PermissionDeniedError(ApiError):
    """403 Forbidden."""


class NotFoundError(ApiError):
    """404 Not Found."""


class RateLimitError(ApiError):
    """429 Too Many Requests."""


class InternalServerError(ApiError):
    """500 Internal Server Error."""


class ServiceUnavailableError(ApiError):
    """503 Service Unavailable."""


class ConnectionError(SmgError):
    """Network connection error."""


class TimeoutError(SmgError):
    """Request timeout error."""


_STATUS_MAP: dict[int, type[ApiError]] = {
    400: BadRequestError,
    401: AuthenticationError,
    403: PermissionDeniedError,
    404: NotFoundError,
    429: RateLimitError,
    500: InternalServerError,
    503: ServiceUnavailableError,
}


def raise_for_status(status_code: int, body: str) -> None:
    """Raise an appropriate ApiError for non-2xx responses."""
    if 200 <= status_code < 300:
        return

    # Try to parse OpenAI-style error body
    message = body
    error_type = None
    param = None
    code = None

    try:
        import json

        data = json.loads(body)
        if "error" in data and isinstance(data["error"], dict):
            err = data["error"]
            message = err.get("message", body)
            error_type = err.get("type")
            param = err.get("param")
            code = err.get("code")
        elif "error" in data and isinstance(data["error"], str):
            message = data["error"]
        elif "type" in data and data.get("type") == "error":
            # Anthropic-style error
            err = data.get("error", {})
            message = err.get("message", body)
            error_type = err.get("type")
    except (json.JSONDecodeError, KeyError):
        pass

    cls = _STATUS_MAP.get(status_code, ApiError)
    raise cls(
        message=message,
        status_code=status_code,
        error_type=error_type,
        param=param,
        code=code,
        body=body,
    )
