"""Ground-truth tests for OpenAI's `image_generation` Responses API output.

These tests hit `api.openai.com` directly (no SMG gateway in the loop) and
assert the wire shape that OpenAI actually returns. They exist to pin the
contract our protocol types and transformer claim to implement, so any
drift between our crates/protocols types and OpenAI's production wire
format is caught by CI rather than by downstream consumers.

Runs in the existing `openai-responses` job (matrix entry in
`.github/workflows/pr-test-rust.yml`) which already provisions
``OPENAI_API_KEY``. Skips locally when the key is absent.

Artefacts (written under the pytest tmp dir) are kept for offline
inspection when a test fails — a failing assertion prints the captured
JSON alongside the diff so the fix scope is obvious.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import openai
import pytest

MODEL = os.environ.get("OPENAI_IMAGE_GEN_MODEL", "gpt-5-nano")
PROMPT = "Generate a 1x1 pixel test image."

# Fields the spec documents on an `image_generation_call` output item.
# See `openai-responses-api-spec.md` §ImageGenerationCall and the Responses
# streaming events reference. Every field is either REQUIRED or OPTIONAL;
# the tests assert REQUIRED fields are present and any OPTIONAL field
# that does appear has a spec-compliant type.
_REQUIRED_ITEM_FIELDS: dict[str, type | tuple[type, ...]] = {
    "id": str,
    "type": str,
    "status": str,
    "result": str,
}
_OPTIONAL_ITEM_FIELDS: dict[str, type | tuple[type, ...]] = {
    "revised_prompt": (str, type(None)),
    "action": (str, type(None)),
    "background": (str, type(None)),
    "output_format": (str, type(None)),
    "quality": (str, type(None)),
    "size": (str, type(None)),
}
_VALID_STATUS: set[str] = {"in_progress", "completed", "generating", "failed"}

_REQUIRED_STREAM_EVENTS: tuple[str, ...] = (
    "response.created",
    "response.output_item.added",
    "response.image_generation_call.in_progress",
    "response.image_generation_call.generating",
    "response.image_generation_call.completed",
    "response.output_item.done",
    "response.completed",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def openai_client() -> openai.OpenAI:
    """Real OpenAI client. Skips the module when the key is absent."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set — ground-truth tests require the live API")
    return openai.OpenAI(api_key=key)


def _find_image_gen_call(output):
    """Return the first `image_generation_call` output item, or ``None``."""
    for item in output or []:
        if getattr(item, "type", None) == "image_generation_call":
            return item
    return None


def _assert_item_shape(item_dict: dict, *, context: str) -> None:
    """Assert the item matches the `image_generation_call` spec shape."""
    for field, expected_type in _REQUIRED_ITEM_FIELDS.items():
        assert field in item_dict, (
            f"[{context}] required field {field!r} missing from image_generation_call. "
            f"Observed keys: {sorted(item_dict)}\nFull item: {json.dumps(item_dict, indent=2)}"
        )
        assert isinstance(item_dict[field], expected_type), (
            f"[{context}] field {field!r} has wrong type "
            f"{type(item_dict[field]).__name__}, expected {expected_type}"
        )

    for field, expected_type in _OPTIONAL_ITEM_FIELDS.items():
        if field in item_dict:
            assert isinstance(item_dict[field], expected_type), (
                f"[{context}] optional field {field!r} has wrong type "
                f"{type(item_dict[field]).__name__}, expected {expected_type}"
            )

    assert item_dict["type"] == "image_generation_call", (
        f"[{context}] wrong type: {item_dict['type']!r}"
    )
    assert item_dict["id"].startswith("ig_"), (
        f"[{context}] id should start with 'ig_'; got {item_dict['id']!r}"
    )
    assert item_dict["status"] in _VALID_STATUS, (
        f"[{context}] status {item_dict['status']!r} not in spec set {_VALID_STATUS}"
    )
    # A completed image MUST carry a non-empty base64 `result`.
    if item_dict["status"] == "completed":
        assert item_dict["result"], (
            f"[{context}] completed image_generation_call has empty result field"
        )


def _extract_unknown_fields(item_dict: dict) -> set[str]:
    """Return keys OpenAI sent but we don't know about (excluding `object`)."""
    known = set(_REQUIRED_ITEM_FIELDS) | set(_OPTIONAL_ITEM_FIELDS)
    # `object` is SDK bookkeeping on some items; ignore it.
    return {k for k in item_dict if k not in known and k != "object"}


def _dump_artifact(tmp_path: Path, name: str, data) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data, indent=2, default=str))
    return path


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


@pytest.mark.vendor("openai")
@pytest.mark.gpu(0)
@pytest.mark.e2e
class TestOpenAIImageGenerationGroundTruth:
    """Pin the OpenAI wire format for `image_generation` against our spec.

    These tests own the answer to "does our ImageGenerationCall type match
    what OpenAI actually sends?" — if a test fails, the attached JSON dump
    is the authoritative reference for the fix.
    """

    def test_non_streaming_output_item_shape(
        self,
        openai_client: openai.OpenAI,
        tmp_path: Path,
    ) -> None:
        resp = openai_client.responses.create(
            model=MODEL,
            input=PROMPT,
            tools=[{"type": "image_generation"}],
            tool_choice={"type": "image_generation"},
            stream=False,
        )
        resp_dict = resp.model_dump(exclude_none=False)
        _dump_artifact(tmp_path, "non_streaming_response.json", resp_dict)

        item = _find_image_gen_call(resp.output)
        assert item is not None, (
            f"No image_generation_call found in response.output. "
            f"Full response: {json.dumps(resp_dict, indent=2)}"
        )
        item_dict = item.model_dump(exclude_none=False)

        _assert_item_shape(item_dict, context="non-streaming")

        unknown = _extract_unknown_fields(item_dict)
        assert not unknown, (
            f"OpenAI returned image_generation_call fields we don't model: {sorted(unknown)}. "
            f"Update crates/protocols/src/responses.rs::ResponseOutputItem::ImageGenerationCall "
            f"to declare them. Full item: {json.dumps(item_dict, indent=2)}"
        )

    def test_streaming_event_sequence_and_final_shape(
        self,
        openai_client: openai.OpenAI,
        tmp_path: Path,
    ) -> None:
        events: list[dict] = []
        final_dict: dict = {}

        with openai_client.responses.stream(
            model=MODEL,
            input=PROMPT,
            tools=[{"type": "image_generation"}],
            tool_choice={"type": "image_generation"},
        ) as stream:
            for event in stream:
                events.append(event.model_dump(exclude_none=False))
            final_dict = stream.get_final_response().model_dump(exclude_none=False)

        _dump_artifact(tmp_path, "streaming_events.json", events)
        _dump_artifact(tmp_path, "streaming_final_response.json", final_dict)

        types_in_order = [e.get("type", "") for e in events]

        def first_img_envelope_idx(evt_type: str) -> int:
            for i, e in enumerate(events):
                if (
                    e.get("type") == evt_type
                    and (e.get("item") or {}).get("type") == "image_generation_call"
                ):
                    return i
            return -1

        def first_idx(evt: str) -> int:
            try:
                return types_in_order.index(evt)
            except ValueError:
                return -1

        # Every required event is present (envelope events scoped to the
        # image_generation_call item — OpenAI emits its own output_item.added/done
        # for any preceding reasoning item, which is not the one we care about).
        missing: list[str] = []
        for evt in _REQUIRED_STREAM_EVENTS:
            idx = (
                first_img_envelope_idx(evt)
                if evt in ("response.output_item.added", "response.output_item.done")
                else first_idx(evt)
            )
            if idx < 0:
                missing.append(evt)
        assert not missing, (
            f"Missing required streaming events: {missing}. "
            f"Observed types: {sorted(set(types_in_order))}\n"
            f"Artefacts in {tmp_path}"
        )

        # Ordering: ig.in_progress < ig.generating < ig.completed < output_item.done (image_gen)
        ig_in_progress = first_idx("response.image_generation_call.in_progress")
        ig_generating = first_idx("response.image_generation_call.generating")
        ig_completed = first_idx("response.image_generation_call.completed")
        img_done = first_img_envelope_idx("response.output_item.done")
        assert ig_in_progress < ig_generating < ig_completed < img_done, (
            f"Event ordering invariant violated: "
            f"in_progress@{ig_in_progress} < generating@{ig_generating} < "
            f"completed@{ig_completed} < output_item.done(img)@{img_done}"
        )

        # Final response.output must carry the image_generation_call item with
        # a shape that matches the non-streaming contract.
        final_output = final_dict.get("output") or []
        final_item = next(
            (i for i in final_output if i.get("type") == "image_generation_call"),
            None,
        )
        assert final_item is not None, (
            f"Streaming final response.output has no image_generation_call. "
            f"Output types: {[i.get('type') for i in final_output]}"
        )
        _assert_item_shape(final_item, context="streaming-final")

        unknown = _extract_unknown_fields(final_item)
        assert not unknown, (
            f"Streaming final image_generation_call has fields we don't model: "
            f"{sorted(unknown)}. Full item: {json.dumps(final_item, indent=2)}"
        )
