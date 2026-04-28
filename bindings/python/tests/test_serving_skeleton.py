"""Smoke tests for the smg-as-tokenspeed-dependency skeleton.

These verify that the protocol-layer entry points exposed via
:py:mod:`smg_rs.serving` (registered into the top-level ``smg_rs`` module)
work end-to-end with the existing ``tool_parser`` / ``reasoning_parser``
crates — the same code paths tokenspeed will call once it imports smg as
a dependency for tokenization, function calling, reasoning parsing, and
the OAI server.

The ``serve_oai`` HTTP entry is still a stub; we just assert it raises a
clear error so callers know the integration hook is wired but the body
isn't implemented yet.
"""

from __future__ import annotations

import json

import pytest


smg_rs = pytest.importorskip("smg_rs")


# ---------------------------------------------------------------------------
# Tool-call parsing
# ---------------------------------------------------------------------------


def test_parse_tool_call_complete_json() -> None:
    """The ``json`` parser passes raw JSON tool calls through verbatim."""
    payload = '{"name": "get_weather", "arguments": {"city": "SF", "unit": "celsius"}}'
    result = smg_rs.parse_tool_call_complete(payload, "json")
    assert isinstance(result, dict)
    assert "tool_calls" in result
    assert "normal_text" in result
    assert len(result["tool_calls"]) == 1
    call = result["tool_calls"][0]
    assert call["name"] == "get_weather"
    args = json.loads(call["arguments"])
    assert args == {"city": "SF", "unit": "celsius"}


def test_parse_tool_call_complete_unknown_parser_raises() -> None:
    with pytest.raises(ValueError, match="unknown tool parser"):
        smg_rs.parse_tool_call_complete("anything", "definitely-not-a-real-parser")


# ---------------------------------------------------------------------------
# Reasoning parsing
# ---------------------------------------------------------------------------


def test_parse_reasoning_qwen3_thinking_block() -> None:
    """Qwen3 emits ``<think>...</think>`` around reasoning content."""
    text = "<think>let me think step by step</think>The answer is 42."
    result = smg_rs.parse_reasoning_complete(text, "qwen3")
    assert isinstance(result, dict)
    assert result["reasoning_text"].strip() == "let me think step by step"
    assert "42" in result["normal_text"]


def test_parse_reasoning_unknown_parser_raises() -> None:
    with pytest.raises(ValueError, match="unknown reasoning parser"):
        smg_rs.parse_reasoning_complete("anything", "definitely-not-a-real-parser")


# ---------------------------------------------------------------------------
# OAI HTTP server (stub) — just verify the hook is wired.
# ---------------------------------------------------------------------------


def test_serve_oai_stub_raises_with_pointer() -> None:
    """``serve_oai`` will host smg's axum HTTP server in-process, driving
    the supplied engine via PyO3 callbacks. Until that lands, the entry
    point is a stub that raises a RuntimeError pointing at the follow-up.
    """
    sentinel_engine = object()
    with pytest.raises(RuntimeError, match="serve_oai is not implemented yet"):
        smg_rs.serve_oai(engine=sentinel_engine, host="127.0.0.1", port=8000)
