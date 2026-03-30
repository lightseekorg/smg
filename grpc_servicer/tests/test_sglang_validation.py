"""
Unit tests for SGLang request validation helpers.

These tests focus on validating tokenized request input in isolation.
"""

import pytest
from smg_grpc_servicer.sglang.validation import validate_tokenized_input


class _Tokenized:
    def __init__(self, original_text: str, input_ids: list[int]):
        self.original_text = original_text
        self.input_ids = input_ids


class _GrpcReqStub:
    def __init__(self, tokenized: _Tokenized | None):
        self.tokenized = tokenized

    def HasField(self, field_name: str) -> bool:  # noqa: N802
        return field_name == "tokenized" and self.tokenized is not None


class TestSGLangValidation:
    """Test tokenized input validation behavior."""

    def test_validate_tokenized_input_rejects_missing_tokenized(self):
        """Reject requests without tokenized payload."""
        req = _GrpcReqStub(tokenized=None)
        with pytest.raises(ValueError, match="Tokenized input must be provided"):
            validate_tokenized_input(req, "generate")

    def test_validate_tokenized_input_rejects_empty_generate(self):
        """Reject generate requests with empty input_ids."""
        req = _GrpcReqStub(tokenized=_Tokenized("hello", []))
        with pytest.raises(ValueError, match="input_ids cannot be empty for generate requests"):
            validate_tokenized_input(req, "generate")

    def test_validate_tokenized_input_rejects_empty_embed(self):
        """Reject embed requests with empty input_ids."""
        req = _GrpcReqStub(tokenized=_Tokenized("hello", []))
        with pytest.raises(ValueError, match="input_ids cannot be empty for embed requests"):
            validate_tokenized_input(req, "embed")

    def test_validate_tokenized_input_returns_text_and_ids(self):
        """Return original text and ids for valid tokenized payload."""
        req = _GrpcReqStub(tokenized=_Tokenized("hello", [1, 2]))
        input_text, input_ids = validate_tokenized_input(req, "embed")
        assert input_text == "hello"
        assert input_ids == [1, 2]
