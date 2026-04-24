"""End-to-end tests for the ``web_search_preview`` built-in tool.

Validates the MCP-as-builtin pattern previously exercised for
``image_generation``: a request that opts into ``{"type": "web_search_preview"}``
is intercepted by the gateway, dispatched to the in-process ``MockMcpServer``
configured with ``builtin_type: web_search_preview``, and the result is
shaped into a ``web_search_call`` output item that matches the production
OpenAI Responses API wire shape (``id``, ``status``, ``action``, ``type``).

The cloud lane uses ``gpt-5-nano`` against the OpenAI backend; the local
gRPC lane uses ``openai/gpt-oss-20b`` over the harmony tool path. Both
exercise the same gateway code and assert the same response item shape.

Note: ``{"type": "web_search"}`` (the non-preview hosted tool) is **not**
routed through MCP-builtin in the current code path. See
``model_gateway/src/routers/common/mcp_utils.rs::extract_builtin_types``,
which only forwards ``WebSearchPreview``. Requests with that tool type
fall through to the upstream model directly. ``TestWebSearchNonPreview``
documents the gap with a skip rather than asserting upstream behaviour.

See ``e2e_test/responses/conftest.py`` for the gateway/MCP fixtures and
``e2e_test/infra/mock_mcp_server.py`` for the mock tool implementation.
"""

from __future__ import annotations

import logging

import pytest

logger = logging.getLogger(__name__)


# =============================================================================
# Shared constants
# =============================================================================

_WEB_SEARCH_PROMPT = "What's the weather in Paris today?"

# Force the model to invoke ``web_search_preview`` rather than auto-planning
# its way to a text-only answer; without this the assertions below would be
# vacuous on a turn that skipped the tool. The wire shape matches
# ``BuiltInToolChoiceType::WebSearchPreview`` in
# ``crates/protocols/src/responses.rs``.
_FORCED_TOOL_CHOICE = {"type": "web_search_preview"}

# Streaming events emitted for a successful ``web_search_call`` per
# ``crates/protocols/src/event_types.rs::WebSearchCallEvent``. The OpenAI
# ground-truth capture (``/tmp/openai_ground_truth/results/web_search_preview.json``)
# shows the full sequence:
#   created → in_progress
#   → output_item.added(reasoning) → output_item.done(reasoning)
#   → output_item.added(web_search_call)
#   → web_search_call.in_progress → searching → completed
#   → output_item.done(web_search_call)
#   → ... reasoning + message ... → completed
_WS_IN_PROGRESS = "response.web_search_call.in_progress"
_WS_SEARCHING = "response.web_search_call.searching"
_WS_COMPLETED = "response.web_search_call.completed"

_REQUIRED_STREAM_EVENTS = (
    "response.created",
    "response.output_item.added",
    _WS_IN_PROGRESS,
    _WS_SEARCHING,
    _WS_COMPLETED,
    "response.output_item.done",
    "response.completed",
)


# =============================================================================
# Helpers
# =============================================================================


def _find_web_search_call(output) -> object | None:
    """Return the first ``web_search_call`` item in a response's output."""
    for item in output:
        if getattr(item, "type", None) == "web_search_call":
            return item
    return None


def _action_query(action) -> str | None:
    """Return ``action.query`` whether ``action`` is a dict or a Pydantic model."""
    if action is None:
        return None
    if isinstance(action, dict):
        return action.get("query")
    return getattr(action, "query", None)


def _action_type(action) -> str | None:
    """Return ``action.type`` whether ``action`` is a dict or a Pydantic model."""
    if action is None:
        return None
    if isinstance(action, dict):
        return action.get("type")
    return getattr(action, "type", None)


def _assert_web_search_call_item(item) -> None:
    """Assert the documented field shape on a ``web_search_call`` item.

    Spec (``crates/protocols/src/responses.rs::ResponseOutputItem::WebSearchCall``):
    ``{ id, action, status, type: "web_search_call", results? }``. Per the
    production capture (``/tmp/openai_ground_truth/results/web_search_preview.json``),
    ``action`` is the ``Search`` variant ``{type: "search", query, queries, sources}``
    and ``status`` is ``"completed"``. The optional ``results`` array stays
    omitted unless the caller asks for it via top-level ``include[]``.
    """
    assert item is not None, "Expected a web_search_call output item"
    assert item.type == "web_search_call", f"wrong item type: {item.type!r}"
    assert item.id.startswith("ws_"), f"id should be prefixed 'ws_'; got {item.id!r}"
    assert item.status == "completed", f"status should be 'completed'; got {item.status!r}"

    # The action carries the search query — Pydantic deserializes it into a
    # tagged-union model. We verify both the discriminator and the populated
    # ``query`` string regardless of model representation.
    assert item.action is not None, "web_search_call.action must be populated"
    assert _action_type(item.action) == "search", (
        f"action.type should be 'search'; got {_action_type(item.action)!r}"
    )
    query = _action_query(item.action)
    assert isinstance(query, str) and query, (
        f"action.query should be a non-empty string; got {query!r}"
    )

    # ``arguments`` is a function-call-style field that must NOT appear on
    # hosted-tool output items. The production capture confirms the shape is
    # ``{id, action, status, type}`` — no ``arguments``, no ``output``.
    assert not hasattr(item, "arguments") or getattr(item, "arguments", None) is None, (
        "web_search_call must not carry an 'arguments' field"
    )
    assert not hasattr(item, "output") or getattr(item, "output", None) is None, (
        "web_search_call must not carry an 'output' field"
    )


def _collect_stream_events(events) -> list:
    """Return the full ordered list of events from a streaming response."""
    collected = list(events)
    assert collected, "Streaming response produced no events"
    return collected


def _final_output_from_stream(events: list) -> list:
    """Pull the final ``response.output`` from the ``response.completed`` event."""
    completed = [e for e in events if getattr(e, "type", None) == "response.completed"]
    assert len(completed) == 1, (
        f"Expected exactly one response.completed event; got {len(completed)}"
    )
    return completed[0].response.output


def _assert_streaming_envelope(events: list) -> None:
    """Assert the ordered streaming envelope for a ``web_search_call`` turn.

    Mirrors the image_generation envelope check but scoped to the
    ``web_search_call`` item — the cloud lane wraps the call in a reasoning
    item beforehand, so a naive ``index()`` would match the reasoning
    envelope rather than the tool's.
    """
    types_in_order: list[str] = [str(getattr(e, "type", "")) for e in events]

    def first_idx(evt: str) -> int:
        try:
            return types_in_order.index(evt)
        except ValueError:
            return -1

    def first_ws_envelope_idx(evt: str) -> int:
        for i, e in enumerate(events):
            if getattr(e, "type", None) == evt and (
                getattr(getattr(e, "item", None), "type", None) == "web_search_call"
            ):
                return i
        return -1

    def scoped_idx(evt: str) -> int:
        if evt in ("response.output_item.added", "response.output_item.done"):
            return first_ws_envelope_idx(evt)
        return first_idx(evt)

    missing = [evt for evt in _REQUIRED_STREAM_EVENTS if scoped_idx(evt) < 0]
    assert not missing, (
        f"Missing required streaming events: {missing}. "
        f"Observed event types: {sorted(set(types_in_order))}"
    )

    idxs = [scoped_idx(evt) for evt in _REQUIRED_STREAM_EVENTS]
    for earlier, later in zip(_REQUIRED_STREAM_EVENTS, _REQUIRED_STREAM_EVENTS[1:]):
        assert scoped_idx(earlier) < scoped_idx(later), (
            f"Events out of order: {earlier!r} (first@{scoped_idx(earlier)}) must precede "
            f"{later!r} (first@{scoped_idx(later)}). Required-event indices: "
            f"{dict(zip(_REQUIRED_STREAM_EVENTS, idxs))}"
        )

    # Count bounds: exactly one created/completed envelope and one
    # added/done pair scoped to the web_search_call item.
    assert types_in_order.count("response.created") == 1, (
        "Expected exactly one response.created event"
    )
    assert types_in_order.count("response.completed") == 1, (
        "Expected exactly one response.completed event"
    )
    ws_added = sum(
        1
        for e in events
        if getattr(e, "type", None) == "response.output_item.added"
        and getattr(getattr(e, "item", None), "type", None) == "web_search_call"
    )
    ws_done = sum(
        1
        for e in events
        if getattr(e, "type", None) == "response.output_item.done"
        and getattr(getattr(e, "item", None), "type", None) == "web_search_call"
    )
    assert ws_added == ws_done == 1, (
        f"Expected exactly one output_item.added + one output_item.done for the "
        f"web_search_call; got added={ws_added}, done={ws_done}"
    )


# =============================================================================
# Shared test mix-in body
# =============================================================================


class _WebSearchPreviewAssertions:
    """Concrete test bodies shared by cloud + local fixture classes.

    Subclasses supply a class-level ``_fixture_name`` pointing at the
    pytest fixture that yields ``(gateway, client, mock_mcp, model)``.
    """

    _fixture_name: str = ""  # overridden by subclasses

    def _ctx(self, request):
        """Pull ``(gateway, client, mock_mcp, model)`` from the concrete fixture."""
        return request.getfixturevalue(self._fixture_name)

    def test_web_search_non_streaming(self, request) -> None:
        """Non-streaming: assert every documented field on the output item."""
        _, client, mock_mcp, model = self._ctx(request)

        baseline_calls = len(mock_mcp.call_log)

        resp = client.responses.create(
            model=model,
            input=_WEB_SEARCH_PROMPT,
            tools=[{"type": "web_search_preview"}],
            tool_choice=_FORCED_TOOL_CHOICE,
            stream=False,
        )

        assert resp.error is None, f"Response error: {resp.error}"
        assert resp.output is not None

        # Non-vacuity guard: confirm the mock saw a fresh dispatch. Otherwise
        # a regression that bypasses MCP entirely would still let the cloud
        # answer pass through and the test could green spuriously on a
        # production-OpenAI ``web_search_call`` shape.
        assert len(mock_mcp.call_log) > baseline_calls, (
            f"Mock MCP server saw no new calls (baseline={baseline_calls}); "
            "the gateway did not dispatch web_search_preview through MCP."
        )
        last_args = mock_mcp.last_call_args
        assert last_args is not None and last_args.get("tool") == "web_search", (
            f"Last MCP call should be web_search; got {last_args!r}"
        )

        item = _find_web_search_call(resp.output)
        _assert_web_search_call_item(item)

    def test_web_search_streaming(self, request) -> None:
        """Streaming: assert the full envelope sequence and field payload."""
        _, client, mock_mcp, model = self._ctx(request)

        baseline_calls = len(mock_mcp.call_log)

        resp = client.responses.create(
            model=model,
            input=_WEB_SEARCH_PROMPT,
            tools=[{"type": "web_search_preview"}],
            tool_choice=_FORCED_TOOL_CHOICE,
            stream=True,
        )

        events = _collect_stream_events(resp)
        _assert_streaming_envelope(events)

        assert len(mock_mcp.call_log) > baseline_calls, (
            f"Mock MCP server saw no new calls (baseline={baseline_calls}); "
            "streaming dispatch did not reach MCP."
        )

        # Final payload on the response.completed envelope must round-trip.
        final_output = _final_output_from_stream(events)
        item = _find_web_search_call(final_output)
        _assert_web_search_call_item(item)


# =============================================================================
# Engine-specific classes
# =============================================================================


@pytest.mark.vendor("openai")
@pytest.mark.gpu(0)
class TestWebSearchPreviewCloud(_WebSearchPreviewAssertions):
    """``web_search_preview`` against the OpenAI cloud backend."""

    _fixture_name = "gateway_with_mock_mcp_web_search_cloud"


# Harmony lane (gpt-oss) integration note
# ----------------------------------------
# The harmony builder renders ``web_search_preview`` as a native builtin
# channel (recipient prefix ``browser``), not a function tool — see
# ``model_gateway/src/routers/grpc/harmony/builder.rs::BUILTIN_TOOLS``.
# The harmony parser then routes those messages into ``analysis``
# (reasoning text) rather than emitting a function-call ToolCall, so the
# MCP-dispatch path that powers the cloud lane never engages on the
# harmony lane today. Running the cloud assertions here would always red
# (no ``web_search_call`` item in output). The class is preserved for
# parity with R6.5 so the lane is easy to flip on once gpt-oss harmony
# routes ``browser``-recipient calls through MCP, but each test method
# skips with a clear reason for now.


@pytest.mark.engine("sglang")
@pytest.mark.gpu(2)
@pytest.mark.e2e
@pytest.mark.model("openai/gpt-oss-20b")
class TestWebSearchPreviewGrpcSglang(_WebSearchPreviewAssertions):
    """``web_search_preview`` against local SGLang + gpt-oss via harmony.

    The harmony lane does not yet route gpt-oss's native ``browser``
    builtin through MCP, so the shared assertions cannot pass. Once the
    harmony parser converts ``browser``-recipient calls into MCP
    dispatches, the skip overrides below become unreachable and the
    inherited assertions take effect.
    """

    _fixture_name = "gateway_with_mock_mcp_web_search_grpc_sglang"

    def test_web_search_non_streaming(self, request) -> None:
        pytest.skip(
            "Harmony lane does not route web_search_preview through MCP yet "
            "(gpt-oss emits a 'browser' builtin which the harmony parser "
            "treats as reasoning, not a function call). See "
            "model_gateway/src/routers/grpc/harmony/parser.rs."
        )

    def test_web_search_streaming(self, request) -> None:
        pytest.skip(
            "Harmony lane does not route web_search_preview through MCP yet "
            "(gpt-oss emits a 'browser' builtin which the harmony parser "
            "treats as reasoning, not a function call). See "
            "model_gateway/src/routers/grpc/harmony/parser.rs."
        )


# =============================================================================
# Non-preview ``web_search`` documentation
# =============================================================================


@pytest.mark.vendor("openai")
@pytest.mark.gpu(0)
class TestWebSearchNonPreview:
    """Document the gap: ``{"type": "web_search"}`` is not MCP-routed.

    The gateway's ``extract_builtin_types`` (in
    ``model_gateway/src/routers/common/mcp_utils.rs``) maps only
    ``WebSearchPreview`` / ``CodeInterpreter`` / ``ImageGeneration`` to
    builtin MCP routing. Requests with ``ResponseTool::WebSearch`` fall
    through to the upstream model unchanged via
    ``serde_json::to_value(tool)`` in
    ``model_gateway/src/routers/openai/responses/utils.rs``. There is no
    deterministic-mock equivalent today, so we record the gap as a skip
    rather than a flaky upstream-dependent assertion.
    """

    def test_web_search_non_preview_is_not_mcp_routed(self) -> None:
        pytest.skip(
            "web_search (non-preview) is not routed through MCP-builtin in the "
            "current gateway. extract_builtin_types only handles WebSearchPreview; "
            "see model_gateway/src/routers/common/mcp_utils.rs."
        )
