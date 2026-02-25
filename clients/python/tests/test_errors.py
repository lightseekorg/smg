"""Tests for error handling."""

import pytest

from smg_client._errors import (
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
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


def test_raise_for_status_plain_text():
    with pytest.raises(BadRequestError) as exc_info:
        raise_for_status(400, "plain error text")
    assert exc_info.value.message == "plain error text"
