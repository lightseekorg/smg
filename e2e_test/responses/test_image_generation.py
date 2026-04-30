"""End-to-end tests for the ``image_generation`` built-in tool.

Validates R6.1-R6.4: a request that opts into the ``image_generation`` tool is
routed through an MCP server (configured via ``--mcp-config-path``), the MCP
tool result is transformed into an ``image_generation_call`` output item, the
argument compactor applies size/quality overrides correctly, and multi-turn
replay strips the base64 payload from stored conversation context.

The tests point the gateway at the in-process ``MockMcpServer`` so that:

* responses are deterministic (byte-for-byte assertions on the base64 image),
* tests run in <100 ms per case rather than waiting on OpenAI's real
  image-gen backend,
* no external service (Brave, OpenAI Images API) is a required dependency.

See ``e2e_test/infra/mock_mcp_server.py`` for the mock implementation and
``e2e_test/responses/conftest.py`` for the ``gateway_with_mock_mcp`` and
local-engine fixtures.

Engine matrix
-------------
* **openai** cloud (``TestImageGenerationCloud``) — exercised against the
  real ``gpt-5-nano`` upstream; the mock MCP server replaces the image
  backend. Validates the OpenAI-compat router (R6.2).
* **sglang** + gpt-oss-20b via harmony (``TestImageGenerationGrpc``,
  engine=sglang) — validates the gRPC-harmony path (R6.3).
* **vllm** + Llama-3.1-8B-Instruct via regular (``TestImageGenerationGrpc``,
  engine=vllm) — validates the gRPC-regular path (R6.4).

Real-worker lanes may surface genuine integration gaps in R6.3/R6.4; those
are filed as R6.6/R6.7 follow-ups rather than patched from this PR.
"""

from __future__ import annotations

import logging

import httpx
import pytest

logger = logging.getLogger(__name__)


# =============================================================================
# Shared constants
# =============================================================================

_IMAGE_GEN_PROMPT = "Generate a picture of a cat"

# Force the model to invoke ``image_generation`` rather than auto-planning its
# way to a text-only answer. Without this, a model run that skips the tool
# would make every assertion below either vacuous (the base64-strip test
# would trivially pass) or silently misleading. The wire shape matches
# ``ToolChoice::Hosted`` in ``crates/protocols/src/responses.rs``.
_FORCED_TOOL_CHOICE = {"type": "image_generation"}

# Event types emitted for a ``image_generation_call`` streaming response,
# per ``crates/protocols/src/event_types.rs::ImageGenerationCallEvent`` and
# the envelope from ``ResponseEvent`` / ``OutputItemEvent``. ``partial_image``
# is optional (depends on backend support); every other event must appear.
_IMG_IN_PROGRESS = "response.image_generation_call.in_progress"
_IMG_GENERATING = "response.image_generation_call.generating"
_IMG_PARTIAL_IMAGE = "response.image_generation_call.partial_image"
_IMG_COMPLETED = "response.image_generation_call.completed"

_REQUIRED_STREAM_EVENTS = (
    "response.created",
    "response.output_item.added",
    _IMG_IN_PROGRESS,
    _IMG_GENERATING,
    _IMG_COMPLETED,
    "response.output_item.done",
    "response.completed",
)


# =============================================================================
# Helpers
# =============================================================================


def _find_image_generation_call(output) -> object | None:
    """Return the first ``image_generation_call`` item in a response's output."""
    for item in output:
        if getattr(item, "type", None) == "image_generation_call":
            return item
    return None


def _assert_image_generation_call_item(
    item,
    *,
    expected_base64: str,
) -> None:
    """Assert every field on the emitted ``ImageGenerationCall`` item.

    Spec (``crates/protocols/src/responses.rs``):
    ``{ id, result: base64, revised_prompt?, status, type: "image_generation_call" }``.
    Asserting each field individually (rather than relying on SDK
    deserialization alone) catches silent drift where the gateway drops a
    field after R6.1-R6.4.

    ``revised_prompt`` is checked as "non-empty string" rather than
    byte-equal to the caller's input. The mock MCP server echoes
    whichever prompt the gateway forwards to it, but the forwarded
    prompt is supplied by the upstream model (gpt-5-nano on the cloud
    lane, gpt-oss-20b / Llama-3.1-8B on the gRPC lanes). Every one of
    those models rewrites the caller's string before invoking the
    tool — sometimes safety-revising, sometimes summarising
    ("Generate a picture of a cat" → "cat") — so an exact-echo
    assertion is inherently flaky across models. A dropped /
    non-string ``revised_prompt`` still fails, which is what we
    actually need to guard against.
    """
    assert item is not None, "Expected an image_generation_call output item"
    assert item.type == "image_generation_call", f"wrong item type: {item.type!r}"
    assert item.id.startswith("ig_"), f"id should be prefixed 'ig_'; got {item.id!r}"
    assert item.status == "completed", f"status should be 'completed'; got {item.status!r}"
    assert item.result == expected_base64, (
        "image_generation_call.result did not round-trip the deterministic mock PNG"
    )
    assert isinstance(item.revised_prompt, str) and item.revised_prompt, (
        f"revised_prompt should be a non-empty string; got {item.revised_prompt!r}"
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
    """Assert the full ordered streaming envelope for image_generation.

    Verifies:
    * every non-optional event (``_REQUIRED_STREAM_EVENTS``) appears at
      least once;
    * each required event's first index is strictly less than the next
      required event's first index;
    * ``sequence_number`` (if present on all events) is strictly
      monotonically increasing with no gaps > 1;
    * ``partial_image`` (optional) sits between ``generating`` and
      ``completed`` when present.
    """
    # Coerce to ``str`` so the list is narrowly typed for downstream use —
    # ``getattr(e, "type", None)`` would otherwise type to ``str | None``.
    types_in_order: list[str] = [str(getattr(e, "type", "")) for e in events]

    def first_idx(evt: str) -> int:
        try:
            return types_in_order.index(evt)
        except ValueError:
            return -1

    def first_img_envelope_idx(evt: str) -> int:
        """First ``response.output_item.{added,done}`` whose ``item.type`` is
        ``image_generation_call``. Plain ``first_idx`` would match the
        earlier envelope for a preceding reasoning / message item —
        OpenAI cloud emits a reasoning item with its own
        added/done pair before the image_generation_call item on
        ``gpt-5-nano``, which would make the ordering check compare the
        wrong envelope.
        """
        for i, e in enumerate(events):
            if getattr(e, "type", None) == evt and (
                getattr(getattr(e, "item", None), "type", None) == "image_generation_call"
            ):
                return i
        return -1

    def scoped_idx(evt: str) -> int:
        """Per-event index that filters envelope events to the
        image_generation_call item but passes other event types through
        to ``first_idx``.
        """
        if evt in ("response.output_item.added", "response.output_item.done"):
            return first_img_envelope_idx(evt)
        return first_idx(evt)

    # Presence — envelope events must be present AND scoped to the
    # image_generation_call item, not any other item in the output array.
    missing = [evt for evt in _REQUIRED_STREAM_EVENTS if scoped_idx(evt) < 0]
    assert not missing, (
        f"Missing required streaming events: {missing}. "
        f"Observed event types: {sorted(set(types_in_order))}"
    )

    # Order (each event's first occurrence < the next event's first occurrence)
    idxs = [scoped_idx(evt) for evt in _REQUIRED_STREAM_EVENTS]
    for earlier, later in zip(_REQUIRED_STREAM_EVENTS, _REQUIRED_STREAM_EVENTS[1:]):
        assert scoped_idx(earlier) < scoped_idx(later), (
            f"Events out of order: {earlier!r} (first@{scoped_idx(earlier)}) must precede "
            f"{later!r} (first@{scoped_idx(later)}). Full sequence of required first indices: "
            f"{dict(zip(_REQUIRED_STREAM_EVENTS, idxs))}"
        )

    # Optional partial_image must sit between generating and completed
    partial_idx = first_idx(_IMG_PARTIAL_IMAGE)
    if partial_idx >= 0:
        gen_idx = first_idx(_IMG_GENERATING)
        comp_idx = first_idx(_IMG_COMPLETED)
        assert gen_idx < partial_idx < comp_idx, (
            f"partial_image@{partial_idx} must sit between generating@{gen_idx} "
            f"and completed@{comp_idx}"
        )

    # Count bounds: exactly one response.created / response.completed and
    # one output_item.added / output_item.done per image_gen call.
    assert types_in_order.count("response.created") == 1, (
        "Expected exactly one response.created event"
    )
    assert types_in_order.count("response.completed") == 1, (
        "Expected exactly one response.completed event"
    )
    img_added = sum(
        1
        for e in events
        if getattr(e, "type", None) == "response.output_item.added"
        and getattr(getattr(e, "item", None), "type", None) == "image_generation_call"
    )
    img_done = sum(
        1
        for e in events
        if getattr(e, "type", None) == "response.output_item.done"
        and getattr(getattr(e, "item", None), "type", None) == "image_generation_call"
    )
    assert img_added == img_done == 1, (
        f"Expected exactly one output_item.added + one output_item.done for the "
        f"image_generation_call; got added={img_added}, done={img_done}"
    )

    # sequence_number monotonic-and-no-gaps (when the field is populated on
    # every event; some streams skip it on a few envelope events, so only
    # check when every event has one).
    raw_seqs = [getattr(e, "sequence_number", None) for e in events]
    if all(s is not None for s in raw_seqs):
        seqs: list[int] = [int(s) for s in raw_seqs if s is not None]
        for i in range(1, len(seqs)):
            assert seqs[i] > seqs[i - 1], (
                f"sequence_number not strictly increasing at index {i}: {seqs[i - 1]} -> {seqs[i]}"
            )
            assert seqs[i] - seqs[i - 1] == 1, (
                f"sequence_number gap > 1 at index {i}: {seqs[i - 1]} -> {seqs[i]}"
            )


def _extract_conversation_id(resp) -> str | None:
    """Normalise the conversation id across SDK variations.

    The OpenAI Responses SDK surfaces the conversation id as either
    ``resp.conversation_id`` (a plain string) or ``resp.conversation`` (an
    object with an ``.id`` attribute), and older releases expose neither.
    This helper collapses those cases to a single ``str | None``.
    """
    conv_attr = getattr(resp, "conversation_id", None) or getattr(resp, "conversation", None)
    if conv_attr is None:
        return None
    if isinstance(conv_attr, str):
        return conv_attr
    return getattr(conv_attr, "id", None)


# =============================================================================
# Shared test mix-in body
# =============================================================================


class _ImageGenerationAssertions:
    """Concrete test bodies shared by cloud + local fixture classes.

    Subclasses supply a class-level ``_fixture_name`` pointing at the
    pytest fixture that yields ``(gateway, client, mock_mcp, model)``.
    Keeping the assertions in a mix-in rather than duplicating them keeps
    the cloud + gRPC lanes strictly in lock-step.
    """

    _fixture_name: str = ""  # overridden by subclasses

    def _ctx(self, request):
        """Pull ``(gateway, client, mock_mcp, model)`` from the concrete fixture."""
        return request.getfixturevalue(self._fixture_name)

    def test_image_generation_non_streaming(self, request, image_gen_tool_args) -> None:
        """Non-streaming: assert every documented field on the output item."""
        _, client, mock_mcp, model = self._ctx(request)

        resp = client.responses.create(
            model=model,
            input=_IMAGE_GEN_PROMPT,
            tools=[image_gen_tool_args],
            tool_choice=_FORCED_TOOL_CHOICE,
            stream=False,
        )

        assert resp.error is None, f"Response error: {resp.error}"
        assert resp.output is not None

        item = _find_image_generation_call(resp.output)
        _assert_image_generation_call_item(
            item,
            expected_base64=mock_mcp.image_generation_png_base64,
        )

    def test_image_generation_streaming(self, request, image_gen_tool_args) -> None:
        """Streaming: assert the full envelope sequence and field payload."""
        _, client, mock_mcp, model = self._ctx(request)

        resp = client.responses.create(
            model=model,
            input=_IMAGE_GEN_PROMPT,
            tools=[image_gen_tool_args],
            tool_choice=_FORCED_TOOL_CHOICE,
            stream=True,
        )

        events = _collect_stream_events(resp)
        _assert_streaming_envelope(events)

        # Final payload must still round-trip.
        final_output = _final_output_from_stream(events)
        item = _find_image_generation_call(final_output)
        _assert_image_generation_call_item(
            item,
            expected_base64=mock_mcp.image_generation_png_base64,
        )

    def test_image_generation_tool_overrides_size(self, request) -> None:
        """Argument compactor: a non-default ``size``/``quality`` reach the tool.

        ``mock_mcp_server`` is session-scoped, so ``last_call_args``
        carries over across tests and classes. Snapshot the call-log
        length before issuing the request and assert a new entry was
        appended; otherwise a regression that silently skips the tool
        invocation here could pass vacuously by re-observing a stale
        ``size="512x512"`` from an earlier class in the matrix.
        """
        _, client, mock_mcp, model = self._ctx(request)

        tool_args = {
            "type": "image_generation",
            "size": "512x512",
            "quality": "high",
        }

        baseline_calls = len(mock_mcp.call_log)

        resp = client.responses.create(
            model=model,
            input=_IMAGE_GEN_PROMPT,
            tools=[tool_args],
            tool_choice=_FORCED_TOOL_CHOICE,
            stream=False,
        )

        assert resp.error is None, f"Response error: {resp.error}"

        # Non-vacuity guard: a fresh MCP invocation must have been
        # recorded. Without this the next two asserts could green on
        # leftover state from an earlier test in the same session.
        assert len(mock_mcp.call_log) > baseline_calls, (
            f"Mock MCP server saw no new calls (baseline={baseline_calls}); "
            "overrides assertion would be vacuous on session-scoped mock."
        )

        last_args = mock_mcp.last_call_args
        assert last_args is not None, "Mock MCP server saw no calls"
        received = last_args.get("arguments", {})
        assert received.get("size") == "512x512", (
            f"Compactor did not pin size override; got {received.get('size')!r}"
        )
        assert received.get("quality") == "high", (
            f"Compactor did not pin quality override; got {received.get('quality')!r}"
        )

    def test_image_generation_user_forwarded_to_mcp(self, request, image_gen_tool_args) -> None:
        """Request-level ``user`` reaches the MCP server's dispatch args.

        The OpenAI Responses API takes a top-level ``user`` field for
        end-user attribution. The gateway forwards that into the
        hosted-tool dispatch args (``routers/common/mcp_utils.rs::
        prepare_hosted_dispatch_args``). The mock server records every
        invocation in ``call_log``, so we can assert the value rode
        through end-to-end without relying on side-channel logs.
        """
        _, client, mock_mcp, model = self._ctx(request)

        user_value = "test-user-forward-abc123"
        baseline_calls = len(mock_mcp.call_log)

        resp = client.responses.create(
            model=model,
            input=_IMAGE_GEN_PROMPT,
            tools=[image_gen_tool_args],
            tool_choice=_FORCED_TOOL_CHOICE,
            user=user_value,
            stream=False,
        )

        assert resp.error is None, f"Response error: {resp.error}"

        # Non-vacuity guard: at least one fresh MCP call was recorded.
        # Without this the next assertion could pass on stale state
        # from an earlier test in the same session.
        assert len(mock_mcp.call_log) > baseline_calls, (
            f"Mock MCP server saw no new calls (baseline={baseline_calls}); "
            "user-forwarding assertion would be vacuous."
        )

        last_args = mock_mcp.last_call_args
        assert last_args is not None, "Mock MCP server saw no calls"
        received = last_args.get("arguments", {})
        assert received.get("user") == user_value, (
            f"Gateway did not forward request-level `user` into MCP dispatch args. "
            f"Expected {user_value!r}, got {received.get('user')!r}. "
            f"Full received args: {received!r}"
        )

    def test_image_generation_compactor_strips_base64(self, request, image_gen_tool_args) -> None:
        """Multi-turn replay: base64 payload must not survive into stored context."""
        gateway, client, mock_mcp, model = self._ctx(request)

        resp1 = client.responses.create(
            model=model,
            input=_IMAGE_GEN_PROMPT,
            tools=[image_gen_tool_args],
            tool_choice=_FORCED_TOOL_CHOICE,
            stream=False,
            store=True,
        )
        assert resp1.error is None, f"Turn 1 error: {resp1.error}"

        # Sanity-check that the tool actually ran.
        assert _find_image_generation_call(resp1.output) is not None, (
            "Turn 1 response did not contain an image_generation_call item; "
            "the compactor-replay assertion would be vacuous. "
            f"output types: {[getattr(i, 'type', None) for i in resp1.output or []]}"
        )

        conversation_id = _extract_conversation_id(resp1)
        if not conversation_id:
            pytest.skip(
                "Gateway did not expose a conversation_id on the first response "
                "— compactor-replay assertion depends on stored history."
            )

        api_key = client.api_key
        with httpx.Client(timeout=10.0) as http:
            items_resp = http.get(
                f"{gateway.base_url}/v1/conversations/{conversation_id}/items",
                headers={"Authorization": f"Bearer {api_key}"},
            )
        assert items_resp.status_code == 200, (
            f"Failed to list conversation items: {items_resp.status_code} {items_resp.text}"
        )

        # Positive persistence guard.
        items_data = items_resp.json()
        stored_items = items_data.get("data") or items_data.get("items") or []
        assert stored_items, (
            "Conversation persisted no items — compactor-strip assertion would be "
            f"vacuous. Full response: {items_data!r}"
        )

        payload = items_resp.text
        assert mock_mcp.image_generation_png_base64 not in payload, (
            "Compactor failed to strip base64 payload from stored conversation "
            "history; replay would re-ship the image bytes to the model."
        )


# =============================================================================
# Engine-specific classes
# =============================================================================


@pytest.mark.vendor("openai")
@pytest.mark.gpu(0)
class TestImageGenerationCloud(_ImageGenerationAssertions):
    """``image_generation`` tool against the OpenAI cloud backend (R6.2)."""

    _fixture_name = "gateway_with_mock_mcp_cloud"


# ``gpu(2)`` on the gRPC classes matches the ``e2e-2gpu-responses`` job in
# ``.github/workflows/pr-test-rust.yml`` (``gpu_tier: "2"``). Using
# ``gpu(1)`` would filter these tests out at collection time because
# ``pytest_collection_modifyitems`` in ``e2e_test/fixtures/hooks.py`` does
# a strict ``gpu_count == E2E_GPU_TIER`` comparison, and no 1-GPU CI lane
# currently runs the Responses directory.
#
# The ``e2e-2gpu-responses`` job runs ``engine=sglang`` only — the vllm
# class is a workflow-level gap (no current CI job pairs ``engine=vllm``
# with ``e2e_test/responses`` at tier 2). That's documented in the PR
# body as a follow-up.


@pytest.mark.engine("sglang")
@pytest.mark.gpu(2)
@pytest.mark.e2e
@pytest.mark.model("openai/gpt-oss-20b")
class TestImageGenerationGrpcSglang(_ImageGenerationAssertions):
    """``image_generation`` tool against local SGLang + gpt-oss via harmony (R6.3).

    Failures here indicate a genuine R6.3 integration gap; file as R6.6
    follow-up rather than patching from this PR.
    """

    _fixture_name = "gateway_with_mock_mcp_grpc_sglang"


@pytest.mark.engine("vllm")
@pytest.mark.gpu(2)
@pytest.mark.e2e
@pytest.mark.model("meta-llama/Llama-3.1-8B-Instruct")
class TestImageGenerationGrpcVllm(_ImageGenerationAssertions):
    """``image_generation`` tool against local vLLM + Llama-3.1 via regular (R6.4).

    Failures here indicate a genuine R6.4 integration gap; file as R6.7
    follow-up rather than patching from this PR.
    """

    _fixture_name = "gateway_with_mock_mcp_grpc_vllm"


# =============================================================================
# Negative coverage: server NOT tagged with builtin_type
# =============================================================================


@pytest.mark.vendor("openai")
@pytest.mark.gpu(0)
@pytest.mark.e2e
class TestImageGenerationRegularMcpServer:
    """Locks the design: when the MCP server is registered as a plain MCP
    server (no ``builtin_type`` tag) and the model invokes a tool that
    happens to be named ``image_generation``, the gateway emits an
    ``mcp_call``-shaped output item — NOT an ``image_generation_call``.

    This is the wire shape an external reviewer reported (the ``output``
    field carrying a stringified MCP payload + ``arguments`` field
    carrying the dispatched args) and the configuration that produces
    it. Server-side ``builtin_type`` is the authoritative knob for
    surfacing a hosted-tool shape; absent that tag, the request's
    ``tools: [{"type": "image_generation"}]`` declaration cannot
    auto-promote a plain MCP tool into a hosted-tool output. Any future
    change that auto-resolves response_format from the request tools
    array would break this test on purpose.
    """

    def test_response_is_mcp_call_shape(self, request) -> None:
        gateway, client, _mock, model = request.getfixturevalue(
            "gateway_with_mock_mcp_regular_cloud"
        )
        del gateway

        resp = client.responses.create(
            model=model,
            input=(
                "Use the image_generation tool to produce a 1x1 pixel test image. "
                "Reply with whatever the tool returns."
            ),
            tools=[
                {
                    "type": "mcp",
                    "server_label": "mock-regular",
                    "allowed_tools": ["image_generation"],
                    "require_approval": "never",
                }
            ],
            tool_choice={
                "type": "mcp",
                "server_label": "mock-regular",
                "name": "image_generation",
            },
            stream=False,
        )

        assert resp.error is None, f"Response error: {resp.error}"
        assert resp.output is not None

        mcp_calls = [item for item in resp.output if getattr(item, "type", None) == "mcp_call"]
        assert mcp_calls, (
            f"Expected at least one mcp_call in response.output; got types "
            f"{[getattr(i, 'type', None) for i in resp.output]}"
        )

        # Every dispatched plain-MCP call carries the mcp_call wire shape:
        # `output` holds the stringified MCP payload and `arguments` holds
        # the dispatched args — neither should be the hosted
        # image_generation_call shape.
        for call in mcp_calls:
            assert call.name == "image_generation", f"unexpected mcp_call.name={call.name!r}"
            assert hasattr(call, "output") and call.output is not None, (
                f"mcp_call must carry an `output` field; got dump={call.model_dump()}"
            )
            assert hasattr(call, "arguments") and call.arguments is not None, (
                f"mcp_call must carry an `arguments` field; got dump={call.model_dump()}"
            )

        assert _find_image_generation_call(resp.output) is None, (
            "Plain MCP server (no builtin_type) MUST NOT auto-promote a tool "
            "named `image_generation` into the hosted image_generation_call "
            "output shape. Server-side builtin_type is the authoritative knob."
        )
