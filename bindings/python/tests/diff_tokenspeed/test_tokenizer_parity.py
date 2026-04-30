"""smg.Tokenizer vs tokenspeed.get_tokenizer parity tests."""

import json
from pathlib import Path

import pytest

import smg

FIXTURES = Path(__file__).parent.parent / "fixtures" / "tokenizer"


@pytest.mark.diff_tokenspeed
def test_apply_chat_template_matches_tokenspeed(hf_tokenizer_path):
    """smg.Tokenizer.apply_chat_template should produce the same prompt as tokenspeed's
    AutoTokenizer wrapper for the same input."""
    from tokenspeed.runtime.utils.hf_transformers_utils import get_tokenizer

    smg_tok = smg.Tokenizer.from_file(hf_tokenizer_path)
    ts_tok = get_tokenizer(hf_tokenizer_path)

    messages = json.loads((FIXTURES / "messages_basic.json").read_text())

    smg_prompt = smg_tok.apply_chat_template(messages, add_generation_prompt=True)
    ts_prompt = ts_tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    assert smg_prompt == ts_prompt, (
        f"Chat-template output diverges between smg and tokenspeed:\n"
        f"  smg: {smg_prompt!r}\n"
        f"  tokenspeed: {ts_prompt!r}\n"
    )


@pytest.mark.diff_tokenspeed
def test_encode_matches_tokenspeed(hf_tokenizer_path):
    from tokenspeed.runtime.utils.hf_transformers_utils import get_tokenizer

    smg_tok = smg.Tokenizer.from_file(hf_tokenizer_path)
    ts_tok = get_tokenizer(hf_tokenizer_path)

    text = "Hello, world! Testing parity."

    smg_ids = smg_tok.encode(text, add_special_tokens=True)
    ts_ids = ts_tok.encode(text, add_special_tokens=True)

    assert smg_ids == ts_ids, (
        f"encode diverges between smg and tokenspeed:\n"
        f"  smg: {smg_ids[:20]}... (len={len(smg_ids)})\n"
        f"  tokenspeed: {ts_ids[:20]}... (len={len(ts_ids)})\n"
    )
