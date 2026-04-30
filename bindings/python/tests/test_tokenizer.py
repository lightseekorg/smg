"""Tests for smg.Tokenizer PyO3 binding."""

import pytest

from smg import smg_rs as smg


@pytest.mark.unit
def test_tokenizer_class_exists():
    """The Tokenizer class must be exported from the smg_rs module."""
    assert hasattr(smg, "Tokenizer")
    assert callable(smg.Tokenizer.from_file)


@pytest.mark.unit
def test_from_file_constructs_tokenizer(hf_tokenizer_path):
    """from_file should return a Tokenizer instance for a valid local path."""
    tok = smg.Tokenizer.from_file(hf_tokenizer_path)
    assert tok is not None
    assert "Tokenizer" in repr(tok)


@pytest.mark.unit
def test_from_file_rejects_invalid_path():
    """from_file should raise on input that's neither a valid path nor a real HF repo."""
    with pytest.raises((ValueError, RuntimeError)) as exc:
        smg.Tokenizer.from_file("/this/path/does/not/exist/tokenizer.json")
    msg = str(exc.value).lower()
    # Either path-not-found OR HF Hub repo-not-found / 404 messages should match.
    expected = ("tokenizer", "not found", "exist", "no such file", "404", "repo", "repository")
    assert any(token in msg for token in expected), f"unexpected error message: {msg!r}"
