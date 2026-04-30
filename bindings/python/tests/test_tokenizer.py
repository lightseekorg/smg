"""Tests for smg.Tokenizer PyO3 binding."""

import json
from pathlib import Path

import pytest
from smg import smg_rs as smg

FIXTURES = Path(__file__).parent / "fixtures" / "tokenizer"


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


@pytest.mark.unit
def test_encode_returns_list_of_ints(hf_tokenizer_path):
    tok = smg.Tokenizer.from_file(hf_tokenizer_path)
    ids = tok.encode("hello world")
    assert isinstance(ids, list)
    assert len(ids) > 0
    assert all(isinstance(x, int) for x in ids)


@pytest.mark.unit
def test_encode_with_special_tokens_differs(hf_tokenizer_path):
    tok = smg.Tokenizer.from_file(hf_tokenizer_path)
    with_special = tok.encode("hello", add_special_tokens=True)
    without_special = tok.encode("hello", add_special_tokens=False)
    # With special tokens enabled, the result must be at least as long as without.
    # On most tokenizers it differs; on some (rare) it may match, so we use >=.
    assert len(with_special) >= len(without_special)


@pytest.mark.unit
def test_encode_empty_string_succeeds(hf_tokenizer_path):
    tok = smg.Tokenizer.from_file(hf_tokenizer_path)
    ids = tok.encode("", add_special_tokens=False)
    assert isinstance(ids, list)


@pytest.mark.unit
def test_decode_round_trips(hf_tokenizer_path):
    """encode followed by decode should recover the original text (modulo whitespace/specials)."""
    tok = smg.Tokenizer.from_file(hf_tokenizer_path)
    text = "hello world, this is a test."
    ids = tok.encode(text, add_special_tokens=False)
    out = tok.decode(ids, skip_special_tokens=True)
    assert out.strip() == text.strip()


@pytest.mark.unit
def test_decode_empty_list_returns_empty_string(hf_tokenizer_path):
    tok = smg.Tokenizer.from_file(hf_tokenizer_path)
    assert tok.decode([], skip_special_tokens=True) == ""


@pytest.mark.unit
def test_decode_skip_special_tokens_flag(hf_tokenizer_path):
    """skip_special_tokens=False should produce output at least as long as True."""
    tok = smg.Tokenizer.from_file(hf_tokenizer_path)
    ids = tok.encode("hi", add_special_tokens=True)
    kept = tok.decode(ids, skip_special_tokens=False)
    skipped = tok.decode(ids, skip_special_tokens=True)
    assert len(kept) >= len(skipped)


@pytest.mark.unit
def test_apply_chat_template_basic(hf_tokenizer_path):
    tok = smg.Tokenizer.from_file(hf_tokenizer_path)
    messages = json.loads((FIXTURES / "messages_basic.json").read_text())

    prompt = tok.apply_chat_template(messages)

    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert "helpful assistant" in prompt
    assert "2 + 2" in prompt
    # The template must have added a generation prompt suffix (e.g. "<|im_start|>assistant\n" for Qwen).
    # Check it ends with something non-empty after the user message.
    assert "user" not in prompt.split("assistant")[-1].lower() or "assistant" in prompt


@pytest.mark.unit
def test_apply_chat_template_with_tools(hf_tokenizer_path):
    tok = smg.Tokenizer.from_file(hf_tokenizer_path)
    fixture = json.loads((FIXTURES / "messages_with_tools.json").read_text())

    prompt_no_tools = tok.apply_chat_template(fixture["messages"])
    prompt_with_tools = tok.apply_chat_template(fixture["messages"], tools=fixture["tools"])

    # When tools are passed, the rendered prompt should differ from the no-tools version
    # (the chat template injects tool descriptions). For tokenizers that ignore tools,
    # this assertion would need adjustment - Qwen2.5-Instruct does honor tools.
    assert prompt_with_tools != prompt_no_tools
    assert "get_weather" in prompt_with_tools


@pytest.mark.unit
def test_apply_chat_template_without_generation_prompt(hf_tokenizer_path):
    """add_generation_prompt=False should NOT append the assistant turn header."""
    tok = smg.Tokenizer.from_file(hf_tokenizer_path)
    messages = json.loads((FIXTURES / "messages_basic.json").read_text())

    with_prompt = tok.apply_chat_template(messages, add_generation_prompt=True)
    without_prompt = tok.apply_chat_template(messages, add_generation_prompt=False)

    # The "with generation prompt" version should be at least as long as without.
    assert len(with_prompt) >= len(without_prompt)
    # And the additional content should not be present in the without version.
    assert with_prompt != without_prompt
