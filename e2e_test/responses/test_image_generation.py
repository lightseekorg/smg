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
``e2e_test/responses/conftest.py`` for the ``gateway_with_mock_mcp`` fixture.

Engine matrix
-------------
Only the OpenAI cloud backend is exercised today. Local gRPC lanes
(``sglang``, ``vllm``) are skipped pending CI maturity on R6.3/R6.4; remove
the ``skip_for_runtime`` decorators in a follow-up once those lanes are
stable.
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


# =============================================================================
# Helpers
# =============================================================================


def _find_image_generation_call(output) -> object | None:
    """Return the first ``image_generation_call`` item in a response's output.

    The Responses SDK deserializes each output item into its typed class
    (``ResponseImageGenCallItem`` and friends); here we just dispatch on the
    ``type`` string because the suite only cares about the wire shape.
    """
    for item in output:
        if getattr(item, "type", None) == "image_generation_call":
            return item
    return None


def _assert_image_generation_call_item(item, *, expected_base64: str, expected_prompt: str) -> None:
    """Assert the shape documented at ``ResponseOutputItem::ImageGenerationCall``.

    Spec is defined in
    ``crates/protocols/src/responses.rs``: ``{ id, result: base64, revised_prompt?,
    status, type: "image_generation_call" }``. Asserting the specific fields
    here (rather than relying on SDK deserialization alone) catches any
    silent drift where the gateway drops a field after R6.1-R6.4.
    """
    assert item is not None, "Expected an image_generation_call output item"
    assert item.id.startswith("ig_"), f"id should be prefixed 'ig_'; got {item.id!r}"
    assert item.status == "completed", f"status should be 'completed'; got {item.status!r}"
    assert item.result == expected_base64, (
        "image_generation_call.result did not round-trip the deterministic mock PNG"
    )
    assert item.revised_prompt == expected_prompt, (
        f"revised_prompt should echo the input prompt; got {item.revised_prompt!r}"
    )


def _collect_stream_events(events) -> tuple[list[str], list]:
    """Return (event_types_in_order, events_in_order) from a response iterator."""
    collected = list(events)
    assert collected, "Streaming response produced no events"
    return [e.type for e in collected], collected


def _final_output_from_stream(events: list) -> list:
    """Pull the final ``response.output`` from the ``response.completed`` event."""
    completed = [e for e in events if getattr(e, "type", None) == "response.completed"]
    assert len(completed) == 1, (
        f"Expected exactly one response.completed event; got {len(completed)}"
    )
    return completed[0].response.output


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
# Image generation tests (OpenAI cloud + mock MCP server)
# =============================================================================


@pytest.mark.vendor("openai")
@pytest.mark.gpu(0)
@pytest.mark.skip_for_runtime(
    "sglang", reason="TBD: R6.3 harmony wiring needs e2e coverage on real worker"
)
@pytest.mark.skip_for_runtime(
    "vllm", reason="TBD: R6.4 regular wiring needs e2e coverage on real worker"
)
class TestImageGeneration:
    """``image_generation`` built-in tool round-trip via the mock MCP server."""

    def test_image_generation_non_streaming(
        self, gateway_with_mock_mcp, image_gen_tool_args
    ) -> None:
        """Non-streaming: assert the ``image_generation_call`` output shape."""
        _, client, mock_mcp = gateway_with_mock_mcp

        resp = client.responses.create(
            model="gpt-5-nano",
            input=_IMAGE_GEN_PROMPT,
            tools=[image_gen_tool_args],
            stream=False,
        )

        assert resp.error is None, f"Response error: {resp.error}"
        assert resp.output is not None

        item = _find_image_generation_call(resp.output)
        _assert_image_generation_call_item(
            item,
            expected_base64=mock_mcp.image_generation_png_base64,
            expected_prompt=_IMAGE_GEN_PROMPT,
        )

    def test_image_generation_streaming(self, gateway_with_mock_mcp, image_gen_tool_args) -> None:
        """Streaming: assert the ordered ``image_generation_call.*`` event trio.

        Required order: ``in_progress`` before ``generating`` before ``completed``.
        ``partial_image`` is optional (depends on backend support); when
        present, it must sit between ``generating`` and ``completed``.
        """
        _, client, mock_mcp = gateway_with_mock_mcp

        resp = client.responses.create(
            model="gpt-5-nano",
            input=_IMAGE_GEN_PROMPT,
            tools=[image_gen_tool_args],
            stream=True,
        )

        event_types, events = _collect_stream_events(resp)

        def first(evt: str) -> int:
            """Index of the first event matching ``evt``, or ``-1`` if absent."""
            try:
                return event_types.index(evt)
            except ValueError:
                return -1

        in_progress_idx = first("response.image_generation_call.in_progress")
        generating_idx = first("response.image_generation_call.generating")
        completed_idx = first("response.image_generation_call.completed")
        partial_image_idx = first("response.image_generation_call.partial_image")

        assert in_progress_idx >= 0, (
            "Missing response.image_generation_call.in_progress; "
            f"observed event types: {sorted(set(event_types))}"
        )
        assert generating_idx >= 0, (
            "Missing response.image_generation_call.generating; "
            f"observed event types: {sorted(set(event_types))}"
        )
        assert completed_idx >= 0, (
            "Missing response.image_generation_call.completed; "
            f"observed event types: {sorted(set(event_types))}"
        )
        assert in_progress_idx < generating_idx < completed_idx, (
            "image_generation_call events out of order: "
            f"in_progress@{in_progress_idx}, generating@{generating_idx}, "
            f"completed@{completed_idx}"
        )
        if partial_image_idx >= 0:
            assert generating_idx < partial_image_idx < completed_idx, (
                "partial_image event must sit between generating and completed; "
                f"generating@{generating_idx}, partial_image@{partial_image_idx}, "
                f"completed@{completed_idx}"
            )

        # Round-trip: the ``response.completed`` event carries the final output,
        # and the deterministic payload must match the mock.
        final_output = _final_output_from_stream(events)
        item = _find_image_generation_call(final_output)
        _assert_image_generation_call_item(
            item,
            expected_base64=mock_mcp.image_generation_png_base64,
            expected_prompt=_IMAGE_GEN_PROMPT,
        )

    def test_image_generation_tool_overrides_size(self, gateway_with_mock_mcp) -> None:
        """Argument compactor: a non-default ``size`` flows through to the tool.

        R6.1 lands the size/quality override pipeline; here we drive a custom
        ``size`` through the public API and assert the mock server observed
        the pinned value. If the compactor silently dropped or mutated the
        size this assertion fails.
        """
        _, client, mock_mcp = gateway_with_mock_mcp

        tool_args = {
            "type": "image_generation",
            "size": "512x512",
            "quality": "high",
        }

        resp = client.responses.create(
            model="gpt-5-nano",
            input=_IMAGE_GEN_PROMPT,
            tools=[tool_args],
            stream=False,
        )

        assert resp.error is None, f"Response error: {resp.error}"

        last_args = mock_mcp.last_call_args
        assert last_args is not None, "Mock MCP server saw no calls"
        received = last_args.get("arguments", {})
        assert received.get("size") == "512x512", (
            f"Compactor did not pin size override; got {received.get('size')!r}"
        )
        assert received.get("quality") == "high", (
            f"Compactor did not pin quality override; got {received.get('quality')!r}"
        )

    def test_image_generation_compactor_strips_base64(
        self, gateway_with_mock_mcp, image_gen_tool_args
    ) -> None:
        """Multi-turn replay: base64 payload must not survive into stored context.

        The gateway's compactor is supposed to replace the (potentially huge)
        ``result`` field with a placeholder when persisting conversation
        history, so a follow-up turn built on ``previous_response_id`` does
        not ship the bytes back to the model. We verify the server view via
        ``/v1/conversations/.../items`` — whichever item type the gateway
        chose to store, its payload must not contain the raw base64 string.
        """
        gateway, client, mock_mcp = gateway_with_mock_mcp

        # Turn 1: force the model to invoke ``image_generation``. Leaving
        # tool_choice at the default ``auto`` would let the model answer
        # without ever running the tool, which would make the later
        # "base64 not in stored payload" assertion pass vacuously and
        # miss a real compactor regression. Pinning ``tool_choice`` to
        # the image_generation tool guarantees the stored payload has an
        # ``image_generation_call`` item worth checking.
        resp1 = client.responses.create(
            model="gpt-5-nano",
            input=_IMAGE_GEN_PROMPT,
            tools=[image_gen_tool_args],
            tool_choice={"type": "image_generation"},
            stream=False,
            store=True,
        )
        assert resp1.error is None, f"Turn 1 error: {resp1.error}"

        # Sanity-check that the tool actually ran — otherwise the
        # stripped-base64 assertion below is vacuous.
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

        # Pull stored items for the conversation and look for any persisted
        # form of the image_generation_call payload.
        api_key = client.api_key
        with httpx.Client(timeout=10.0) as http:
            items_resp = http.get(
                f"{gateway.base_url}/v1/conversations/{conversation_id}/items",
                headers={"Authorization": f"Bearer {api_key}"},
            )
        assert items_resp.status_code == 200, (
            f"Failed to list conversation items: {items_resp.status_code} {items_resp.text}"
        )

        payload = items_resp.text
        assert mock_mcp.image_generation_png_base64 not in payload, (
            "Compactor failed to strip base64 payload from stored conversation "
            "history; replay would re-ship the image bytes to the model."
        )
