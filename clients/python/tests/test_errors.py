"""Tests for error handling."""

import pytest

from smg_client._errors import (
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    ServiceUnavailableError,
    raise_for_status,
)


def test_raise_for_status_2xx():
    # Should not raise
    raise_for_status(200, "OK")
    raise_for_status(201, "Created")


def test_raise_for_status_400_openai_format():
    body = '{"error": {"message": "Invalid model", "type": "invalid_request_error"}}'
    with pytest.raises(BadRequestError) as exc_info:
        raise_for_status(400, body)
    assert exc_info.value.message == "Invalid model"
    assert exc_info.value.error_type == "invalid_request_error"
    assert exc_info.value.status_code == 400


def test_raise_for_status_401():
    body = '{"error": {"message": "Unauthorized", "type": "authentication_error"}}'
    with pytest.raises(AuthenticationError):
        raise_for_status(401, body)


def test_raise_for_status_404():
    with pytest.raises(NotFoundError):
        raise_for_status(404, "Not Found")


def test_raise_for_status_429():
    body = '{"error": {"message": "Rate limited"}}'
    with pytest.raises(RateLimitError):
        raise_for_status(429, body)


def test_raise_for_status_500():
    with pytest.raises(InternalServerError):
        raise_for_status(500, "Internal Server Error")


def test_raise_for_status_anthropic_format():
    body = '{"type": "error", "error": {"type": "invalid_request_error", "message": "Bad input"}}'
    with pytest.raises(BadRequestError) as exc_info:
        raise_for_status(400, body)
    assert exc_info.value.message == "Bad input"


def test_raise_for_status_403():
    body = '{"error": {"message": "Forbidden", "type": "permission_error"}}'
    with pytest.raises(PermissionDeniedError) as exc_info:
        raise_for_status(403, body)
    assert exc_info.value.message == "Forbidden"
    assert exc_info.value.status_code == 403


def test_raise_for_status_503():
    with pytest.raises(ServiceUnavailableError) as exc_info:
        raise_for_status(503, "Service Unavailable")
    assert exc_info.value.status_code == 503


def test_raise_for_status_plain_text():
    with pytest.raises(BadRequestError) as exc_info:
        raise_for_status(400, "plain error text")
    assert exc_info.value.message == "plain error text"


def test_raise_for_status_scalar_json():
    """Scalar JSON bodies (null, number, string, bool) must not crash."""
    for body in ["null", "42", '"just a string"', "true"]:
        with pytest.raises(BadRequestError) as exc_info:
            raise_for_status(400, body)
        # Falls through to raw body as message since it's not a dict
        assert exc_info.value.message == body
