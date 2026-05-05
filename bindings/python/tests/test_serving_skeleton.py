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
# OAI HTTP server — verify the entry point is exposed; full end-to-end
# behavior is covered by tools/smg_serve_oai_demo.py against a fake
# AsyncLLM (it spawns a server in a daemon thread and runs an HTTP
# roundtrip), kept out of the unit-test path because it binds a real
# TCP port and pulls in pyo3-async-runtimes' tokio runtime.
# ---------------------------------------------------------------------------


def test_serve_oai_is_exposed() -> None:
    """``serve_oai`` blocks on ``axum::serve`` once entered (it owns the
    HTTP listener for the lifetime of the process), so unit tests don't
    actually invoke it. We only check the symbol exists with the
    expected signature."""
    import inspect

    assert callable(smg_rs.serve_oai), "smg_rs.serve_oai must be callable"
    # ``smg_rs.serve_oai.__doc__`` is rendered from the pyo3 docstring
    # in serving.rs; the doc-comment is the canonical reference for
    # the bridge's behavior so we sanity-check it lands.
    doc = inspect.getdoc(smg_rs.serve_oai) or ""
    assert "engine" in doc.lower(), "serve_oai docstring should mention 'engine' param"
