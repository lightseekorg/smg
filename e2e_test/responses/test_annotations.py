"""Annotation round-trip tests for Response API.

Task E5 (from the Responses API gap audit): verify that
``message.output_text.annotations`` carries typed ``FileCitation``,
``URLCitation``, ``ContainerFileCitation`` and ``FilePath`` variants across
both streaming and non-streaming responses.

Protocol types landed in P1 (``crates/protocols/src/responses.rs``); this
test file only exercises the gateway's surfacing of annotations end-to-end
and makes no assumption about which backend produced them.

The tests run against the OpenAI cloud backend (``@pytest.mark.vendor("openai")``)
because the built-in tools that emit typed annotations (``web_search_preview``,
``file_search``, ``code_interpreter``) are hosted upstream. Each annotation
variant has its own class with both a non-streaming and a streaming case.
Tests that require extra external setup (uploading a file for file_search)
create that setup directly against the OpenAI API and clean it up at
teardown; tests that depend on model-driven output shape (code_interpreter
producing container_file_citation / file_path) validate the variant
structurally only if the model actually produced it, otherwise they skip
with an explanation rather than falsely fail.
"""

from __future__ import annotations

import io
import logging
import os
import time
from collections.abc import Iterable

import openai
import pytest

logger = logging.getLogger(__name__)


# =============================================================================
# Shared helpers
# =============================================================================


_WEB_SEARCH_PROMPT = (
    "Search the web for the latest stable release of the Rust programming language "
    "and cite the official source in one sentence."
)


def _iter_output_text_parts(output: Iterable) -> Iterable:
    """Yield every output_text content part across all message items."""
    for item in output:
        # ``item`` is a ResponseOutputItem; only ``message`` items carry content.
        item_type = getattr(item, "type", None)
        if item_type != "message":
            continue
        content = getattr(item, "content", None) or []
        for part in content:
            if getattr(part, "type", None) == "output_text":
                yield part


def _collect_annotations(output: Iterable) -> list:
    """Flatten annotations from all output_text parts in a response's output."""
    annotations = []
    for part in _iter_output_text_parts(output):
        anns = getattr(part, "annotations", None) or []
        annotations.extend(anns)
    return annotations


def _stream_to_final_output(resp) -> list:
    """Iterate a streaming Responses SDK object and return the final output list.

    The OpenAI SDK emits discrete events; the final output array lives on the
    ``response.completed`` event's nested ``response.output`` field. Iterating
    the full event stream also forces the SDK to deserialize every intermediate
    ``output_item.done`` / ``content_part.done`` / ``content_part.added`` event
    through its discriminated annotation union — that would raise on an unknown
    ``type`` string, so a clean iteration is itself a round-trip assertion.
    """
    events = list(resp)
    assert events, "Streaming response produced no events"

    completed = [e for e in events if getattr(e, "type", None) == "response.completed"]
    assert len(completed) == 1, (
        f"Expected exactly one response.completed event, got {len(completed)}"
    )
    final_output = completed[0].response.output
    assert isinstance(final_output, list)
    return final_output


def _assert_annotation_structurally_valid(ann) -> None:
    """Sanity-check an annotation object against its spec-required fields.

    The OpenAI SDK already discriminates on ``type`` and deserializes into the
    matching typed class; re-checking the required fields here catches both
    regressions in smg's gateway (e.g., dropping a field) and drift in the
    upstream wire contract.
    """
    ann_type = getattr(ann, "type", None)
    if ann_type == "file_citation":
        assert isinstance(ann.file_id, str) and ann.file_id, "file_citation.file_id missing"
        assert isinstance(ann.filename, str) and ann.filename, "file_citation.filename missing"
        assert isinstance(ann.index, int) and ann.index >= 0, "file_citation.index missing"
    elif ann_type == "url_citation":
        assert isinstance(ann.url, str) and ann.url.startswith(("http://", "https://")), (
            f"url_citation.url invalid: {ann.url!r}"
        )
        assert isinstance(ann.title, str), "url_citation.title missing"
        assert isinstance(ann.start_index, int) and ann.start_index >= 0
        assert isinstance(ann.end_index, int) and ann.end_index >= ann.start_index
    elif ann_type == "container_file_citation":
        assert isinstance(ann.container_id, str) and ann.container_id
        assert isinstance(ann.file_id, str) and ann.file_id
        assert isinstance(ann.filename, str) and ann.filename
        assert isinstance(ann.start_index, int) and ann.start_index >= 0
        assert isinstance(ann.end_index, int) and ann.end_index >= ann.start_index
    elif ann_type == "file_path":
        assert isinstance(ann.file_id, str) and ann.file_id, "file_path.file_id missing"
        assert isinstance(ann.index, int) and ann.index >= 0, "file_path.index missing"
    else:
        pytest.fail(
            f"Annotation type {ann_type!r} is not one of the spec variants "
            "(file_citation | url_citation | container_file_citation | file_path)"
        )


# =============================================================================
# URL citation (web_search_preview)
# =============================================================================


@pytest.mark.vendor("openai")
@pytest.mark.gpu(0)
@pytest.mark.parametrize("setup_backend", ["openai"], indirect=True)
class TestUrlCitationAnnotation:
    """Verify ``url_citation`` annotations round-trip from the web_search tool."""

    def _create_web_search_response(self, api_client, model, stream):
        return api_client.responses.create(
            model=model,
            input=_WEB_SEARCH_PROMPT,
            tools=[{"type": "web_search_preview"}],
            stream=stream,
        )

    def test_url_citation_non_streaming(self, model, api_client):
        time.sleep(2)
        resp = self._create_web_search_response(api_client, model, stream=False)
        assert resp.error is None, f"Response error: {resp.error}"
        assert resp.output is not None

        annotations = _collect_annotations(resp.output)
        url_citations = [a for a in annotations if getattr(a, "type", None) == "url_citation"]
        assert url_citations, (
            "Expected at least one url_citation annotation from web_search_preview; "
            f"got annotations: {[getattr(a, 'type', None) for a in annotations]}"
        )
        for ann in url_citations:
            _assert_annotation_structurally_valid(ann)

    def test_url_citation_streaming(self, model, api_client):
        time.sleep(2)
        resp = self._create_web_search_response(api_client, model, stream=True)
        final_output = _stream_to_final_output(resp)

        annotations = _collect_annotations(final_output)
        url_citations = [a for a in annotations if getattr(a, "type", None) == "url_citation"]
        assert url_citations, (
            "Expected at least one url_citation annotation in streaming response; "
            f"got annotations: {[getattr(a, 'type', None) for a in annotations]}"
        )
        for ann in url_citations:
            _assert_annotation_structurally_valid(ann)


# =============================================================================
# File citation (file_search)
# =============================================================================


_FILE_SEARCH_DOC = (
    "Project Astra uses the Pulsar-7 reactor core. "
    "The reactor operates at a nominal temperature of 842 Kelvin. "
    "Maintenance protocol AST-42 requires quarterly coolant flush. "
)


@pytest.fixture(scope="class")
def openai_direct_client():
    """Direct OpenAI client for out-of-band setup that smg does not proxy.

    smg's gateway does not mount ``/v1/files`` or ``/v1/vector_stores``, so we
    create the vector store directly against the OpenAI API and then pass the
    resulting ID through the gateway when calling ``responses.create`` with the
    ``file_search`` tool. This keeps the smg-under-test path exercised for the
    annotation round-trip while sidestepping a hard dependency on endpoint
    coverage that is out of scope for E5.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set — cannot create a vector store for file_search")
    client = openai.OpenAI(api_key=api_key)
    yield client


@pytest.fixture(scope="class")
def file_search_vector_store(openai_direct_client):
    """Create + populate a vector store, then tear it down at class scope."""
    client = openai_direct_client
    file_obj = None
    vs = None
    try:
        buf = io.BytesIO(_FILE_SEARCH_DOC.encode("utf-8"))
        buf.name = "project_astra.txt"
        try:
            file_obj = client.files.create(file=buf, purpose="assistants")
        except openai.OpenAIError as exc:
            pytest.skip(f"Unable to upload file for file_search: {exc}")

        try:
            vs = client.vector_stores.create(name="smg-e5-annotations-test")
        except openai.OpenAIError as exc:
            pytest.skip(f"Unable to create vector store for file_search: {exc}")

        try:
            client.vector_stores.files.create_and_poll(vector_store_id=vs.id, file_id=file_obj.id)
        except openai.OpenAIError as exc:
            pytest.skip(f"Unable to index file into vector store: {exc}")

        yield vs.id
    finally:
        if vs is not None:
            try:
                client.vector_stores.delete(vector_store_id=vs.id)
            except openai.OpenAIError:
                pass
        if file_obj is not None:
            try:
                client.files.delete(file_id=file_obj.id)
            except openai.OpenAIError:
                pass


@pytest.mark.vendor("openai")
@pytest.mark.gpu(0)
@pytest.mark.parametrize("setup_backend", ["openai"], indirect=True)
class TestFileCitationAnnotation:
    """Verify ``file_citation`` annotations round-trip from the file_search tool."""

    _PROMPT = (
        "Using the attached project documentation, what is the nominal operating "
        "temperature of the Pulsar-7 reactor core? Cite the document."
    )

    def _create_file_search_response(self, api_client, model, vector_store_id, stream):
        return api_client.responses.create(
            model=model,
            input=self._PROMPT,
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": [vector_store_id],
                }
            ],
            include=["file_search_call.results"],
            stream=stream,
        )

    def test_file_citation_non_streaming(self, model, api_client, file_search_vector_store):
        time.sleep(2)
        resp = self._create_file_search_response(
            api_client, model, file_search_vector_store, stream=False
        )
        assert resp.error is None, f"Response error: {resp.error}"
        assert resp.output is not None

        annotations = _collect_annotations(resp.output)
        file_citations = [a for a in annotations if getattr(a, "type", None) == "file_citation"]
        if not file_citations:
            pytest.skip(
                "Model did not emit file_citation annotations for this prompt; "
                "annotation structural check is skipped. "
                f"Annotations observed: {[getattr(a, 'type', None) for a in annotations]}"
            )
        for ann in file_citations:
            _assert_annotation_structurally_valid(ann)

    def test_file_citation_streaming(self, model, api_client, file_search_vector_store):
        time.sleep(2)
        resp = self._create_file_search_response(
            api_client, model, file_search_vector_store, stream=True
        )
        final_output = _stream_to_final_output(resp)

        annotations = _collect_annotations(final_output)
        file_citations = [a for a in annotations if getattr(a, "type", None) == "file_citation"]
        if not file_citations:
            pytest.skip(
                "Model did not emit file_citation annotations for this prompt in "
                "streaming mode; annotation structural check is skipped. "
                f"Annotations observed: {[getattr(a, 'type', None) for a in annotations]}"
            )
        for ann in file_citations:
            _assert_annotation_structurally_valid(ann)


# =============================================================================
# Container file citation + File path (code_interpreter)
# =============================================================================


_CODE_INTERPRETER_PROMPT = (
    "Write a tiny CSV file named 'sample.csv' with the columns 'id,value' and "
    "three rows (1,'apple'; 2,'banana'; 3,'cherry'). Then read the file back "
    "and answer: which row has id=2? Reference the file explicitly in your "
    "response so it appears as a citation."
)


def _create_code_interpreter_response(api_client, model, stream):
    return api_client.responses.create(
        model=model,
        input=_CODE_INTERPRETER_PROMPT,
        tools=[
            {
                "type": "code_interpreter",
                "container": {"type": "auto"},
            }
        ],
        stream=stream,
    )


@pytest.mark.vendor("openai")
@pytest.mark.gpu(0)
@pytest.mark.parametrize("setup_backend", ["openai"], indirect=True)
class TestContainerFileCitationAnnotation:
    """Verify ``container_file_citation`` annotations from code_interpreter.

    The container_file_citation variant is produced when the model cites a
    file that lives inside the code_interpreter container. Whether the model
    emits one is model-dependent; the tests validate the wire shape when the
    annotation is present and skip (rather than fail) when it is not, so that
    regressions in the typed round-trip are caught while not tying the suite
    to model-prompt sensitivity.
    """

    def test_container_file_citation_non_streaming(self, model, api_client):
        time.sleep(2)
        try:
            resp = _create_code_interpreter_response(api_client, model, stream=False)
        except openai.BadRequestError as exc:
            pytest.skip(f"code_interpreter unavailable on this model: {exc}")
        assert resp.error is None, f"Response error: {resp.error}"

        annotations = _collect_annotations(resp.output)
        container_citations = [
            a for a in annotations if getattr(a, "type", None) == "container_file_citation"
        ]
        if not container_citations:
            pytest.skip(
                "Model did not emit container_file_citation annotations; "
                f"annotations observed: {[getattr(a, 'type', None) for a in annotations]}"
            )
        for ann in container_citations:
            _assert_annotation_structurally_valid(ann)

    def test_container_file_citation_streaming(self, model, api_client):
        time.sleep(2)
        try:
            resp = _create_code_interpreter_response(api_client, model, stream=True)
        except openai.BadRequestError as exc:
            pytest.skip(f"code_interpreter unavailable on this model: {exc}")
        final_output = _stream_to_final_output(resp)

        annotations = _collect_annotations(final_output)
        container_citations = [
            a for a in annotations if getattr(a, "type", None) == "container_file_citation"
        ]
        if not container_citations:
            pytest.skip(
                "Model did not emit container_file_citation annotations in streaming mode; "
                f"annotations observed: {[getattr(a, 'type', None) for a in annotations]}"
            )
        for ann in container_citations:
            _assert_annotation_structurally_valid(ann)


@pytest.mark.vendor("openai")
@pytest.mark.gpu(0)
@pytest.mark.parametrize("setup_backend", ["openai"], indirect=True)
class TestFilePathAnnotation:
    """Verify ``file_path`` annotations round-trip from generated files.

    ``file_path`` is emitted when the model references a generated file by
    path (e.g., a CSV or image generated during a code_interpreter turn).
    As with ``container_file_citation``, the presence of the annotation is
    model-driven; the test validates the wire shape when present.
    """

    def test_file_path_non_streaming(self, model, api_client):
        time.sleep(2)
        try:
            resp = _create_code_interpreter_response(api_client, model, stream=False)
        except openai.BadRequestError as exc:
            pytest.skip(f"code_interpreter unavailable on this model: {exc}")
        assert resp.error is None, f"Response error: {resp.error}"

        annotations = _collect_annotations(resp.output)
        file_paths = [a for a in annotations if getattr(a, "type", None) == "file_path"]
        if not file_paths:
            pytest.skip(
                "Model did not emit file_path annotations; "
                f"annotations observed: {[getattr(a, 'type', None) for a in annotations]}"
            )
        for ann in file_paths:
            _assert_annotation_structurally_valid(ann)

    def test_file_path_streaming(self, model, api_client):
        time.sleep(2)
        try:
            resp = _create_code_interpreter_response(api_client, model, stream=True)
        except openai.BadRequestError as exc:
            pytest.skip(f"code_interpreter unavailable on this model: {exc}")
        final_output = _stream_to_final_output(resp)

        annotations = _collect_annotations(final_output)
        file_paths = [a for a in annotations if getattr(a, "type", None) == "file_path"]
        if not file_paths:
            pytest.skip(
                "Model did not emit file_path annotations in streaming mode; "
                f"annotations observed: {[getattr(a, 'type', None) for a in annotations]}"
            )
        for ann in file_paths:
            _assert_annotation_structurally_valid(ann)
