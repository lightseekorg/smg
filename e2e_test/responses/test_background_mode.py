"""End-to-end coverage for `POST /v1/responses` with `background=true`.

The gateway accepts `background=true` requests and enqueues them through
`MemoryBackgroundRepository` when `--history-backend=memory` (the default).
The queued skeleton is mirrored into `MemoryResponseStorage` so
`GET /v1/responses/{id}` reads it back unchanged until a worker updates it.

Scope:

* Layer-1 validator rejections (`ValidatedJson` via
  `validate_responses_cross_parameters` in `crates/protocols/src/responses.rs`).
* Shared create handler (`model_gateway/src/routers/common/background/create.rs`)
  happy path, GET read-back, and the two state-dependent 404 branches.
* `history_backend=none` produces `background_not_supported`.

Out of scope:

* `queued → in_progress → completed` transitions. No queue consumer exists
  yet in production code — `MemoryBackgroundRepository::claim_next` has no
  caller outside unit tests. Full completion scenarios stay behind
  `@pytest.mark.skip` in `test_basic_crud.py::test_background_response`
  until the worker loop lands.
"""

from __future__ import annotations

import logging
import os

import httpx
import openai
import pytest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Raw-httpx helpers (validation/error-envelope assertions need exact wire shape)
# ---------------------------------------------------------------------------


def _post_responses(gateway, body: dict, timeout: float = 30.0) -> httpx.Response:
    """POST a raw JSON body to ``{gateway.base_url}/v1/responses``.

    Validation rejections need exact-envelope checks that the OpenAI SDK
    would hide behind a typed exception, so all negative-path tests use
    ``httpx`` directly.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "sk-not-used")
    return httpx.post(
        f"{gateway.base_url}/v1/responses",
        json=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=timeout,
    )


def _assert_validator_400(resp: httpx.Response, expected_code_substring: str) -> None:
    """Assert the gateway returned the Layer-1 validator envelope.

    ``ValidatedJson`` in ``crates/protocols/src/validated.rs:91-103`` maps a
    ``Validate`` failure to ``400`` with
    ``{error: {type: "invalid_request_error", code: 400, message: "..."}}``.
    The validator's ``ValidationError::code`` string
    (e.g. ``background_requires_store``) is rendered into ``message`` via
    ``ValidationErrors::to_string()`` — assert on its presence there rather
    than on the numeric ``code`` field so the test pins the semantic
    invariant instead of the (weird) `code: 400` integer envelope.
    """
    assert resp.status_code == 400, (
        f"expected HTTP 400, got {resp.status_code}: body={resp.text!r}"
    )
    body = resp.json()
    err = body.get("error")
    assert isinstance(err, dict), f"expected error object, got {body!r}"
    assert err.get("type") == "invalid_request_error", (
        f"expected invalid_request_error, got {err!r}"
    )
    message = err.get("message", "")
    assert isinstance(message, str) and expected_code_substring in message, (
        f"expected message containing {expected_code_substring!r}, got {message!r}"
    )


def _assert_handler_error(
    resp: httpx.Response, expected_status: int, expected_code: str
) -> None:
    """Assert a handler-layer (Layer 3) error envelope.

    ``routers::error`` emits
    ``{error: {code: "<kebab>", message: "...", type: "invalid_request_error"}}``
    — asserting on ``code`` (string) distinguishes these from Layer-1
    validator rejections (which use integer ``code: 400``).
    """
    assert resp.status_code == expected_status, (
        f"expected HTTP {expected_status}, got {resp.status_code}: body={resp.text!r}"
    )
    body = resp.json()
    err = body.get("error")
    assert isinstance(err, dict), f"expected error object, got {body!r}"
    assert err.get("code") == expected_code, (
        f"expected code={expected_code!r}, got {err!r}"
    )


# ===========================================================================
# Layer 1 — cross-parameter validator rejections
# ===========================================================================


@pytest.mark.vendor("openai")
@pytest.mark.gpu(0)
@pytest.mark.parametrize("setup_backend", ["openai"], indirect=True)
class TestBackgroundModeValidation:
    """Layer-1 validator (``validate_responses_cross_parameters``) coverage.

    Cloud backend is fine — these requests short-circuit inside
    ``ValidatedJson`` before any upstream call.
    """

    def test_background_plus_stream_rejected_400(self, setup_backend):
        """``background=true`` + ``stream=true`` → ``background_conflicts_with_stream``.

        Pinned at ``crates/protocols/src/responses.rs:3127-3132``.
        """
        _, model, _, gw = setup_backend
        resp = _post_responses(
            gw,
            {
                "model": model,
                "input": "hello",
                "background": True,
                "stream": True,
            },
        )
        _assert_validator_400(resp, "background_conflicts_with_stream")

    def test_background_plus_store_false_rejected_400(self, setup_backend):
        """``background=true`` + explicit ``store=false`` → ``background_requires_store``.

        Pinned at ``crates/protocols/src/responses.rs:3134-3140``.
        """
        _, model, _, gw = setup_backend
        resp = _post_responses(
            gw,
            {
                "model": model,
                "input": "hello",
                "background": True,
                "store": False,
            },
        )
        _assert_validator_400(resp, "background_requires_store")

    def test_background_with_store_unset_is_accepted(self, setup_backend):
        """``store`` unset defaults to ``true`` per the OpenAI spec.

        The validator only rejects ``store=Some(false)`` — ``None`` must pass.
        We verify by asserting the request is NOT rejected at the validator
        layer (status is either 200 with queued body, or a handler-level
        error — anything that is not the validator's 400 envelope).
        """
        _, model, _, gw = setup_backend
        resp = _post_responses(
            gw,
            {
                "model": model,
                "input": "hello",
                "background": True,
            },
        )
        if resp.status_code == 400:
            body = resp.json()
            err = body.get("error", {})
            message = err.get("message", "")
            assert "background_requires_store" not in message, (
                f"store=None must not trigger background_requires_store; "
                f"got message={message!r}"
            )


# ===========================================================================
# Layer 3 — shared handler: enqueue + GET read-back + state-dependent 404s
# ===========================================================================


@pytest.mark.vendor("openai")
@pytest.mark.gpu(0)
@pytest.mark.parametrize("setup_backend", ["openai"], indirect=True)
class TestBackgroundModeEnqueue:
    """Shared-handler happy path and state-dependent error branches.

    Uses the default ``history_backend=memory`` so
    ``AppContext.background_repository`` is ``Some(MemoryBackgroundRepository)``
    (wired in ``crates/data_connector/src/factory.rs:67-83``).
    """

    def test_enqueue_returns_queued_skeleton(self, model, api_client):
        """POST ``background=true`` returns the initial queued response shape.

        Verifies the JSON built in
        ``routers/common/background/create.rs::initial_queued_response``:
        ``status == "queued"``, ``background == true``, id prefix ``resp_``,
        empty ``output``, matching ``model``.
        """
        resp = api_client.responses.create(
            model=model,
            input="Write a short story",
            background=True,
            max_output_tokens=100,
        )
        assert resp.id.startswith("resp_"), f"expected resp_ prefix, got {resp.id!r}"
        assert resp.status == "queued", f"expected queued, got {resp.status!r}"
        assert resp.background is True, f"expected background=True, got {resp.background!r}"
        assert resp.model is not None
        assert resp.output == [], f"expected empty output, got {resp.output!r}"
        assert resp.error is None, f"expected no error, got {resp.error!r}"

    def test_queued_response_readable_via_get(self, model, api_client):
        """``GET /v1/responses/{id}`` returns the mirrored queued skeleton.

        ``MemoryBackgroundRepository::enqueue`` mirrors the response into
        ``MemoryResponseStorage`` under the same lock
        (``memory_background.rs:234``), so the read path sees the queued
        state immediately.
        """
        created = api_client.responses.create(
            model=model,
            input="hello",
            background=True,
            max_output_tokens=100,
        )
        assert created.status == "queued"

        retrieved = api_client.responses.retrieve(response_id=created.id)
        assert retrieved.id == created.id
        assert retrieved.status == "queued"
        assert retrieved.background is True
        assert retrieved.error is None

    def test_background_with_unknown_previous_response_id_returns_404(
        self, model, setup_backend
    ):
        """Chaining to a missing prior response → 404 ``previous_response_not_found``.

        Pinned at
        ``routers/common/background/create.rs::append_prev_chain_items`` →
        ``chain.responses.is_empty()`` branch.
        """
        _, model_path, _, gw = setup_backend
        resp = _post_responses(
            gw,
            {
                "model": model_path,
                "input": "hello",
                "background": True,
                "store": True,
                "previous_response_id": "resp_does_not_exist_zzz",
            },
        )
        _assert_handler_error(resp, 404, "previous_response_not_found")

    def test_background_with_unknown_conversation_returns_404(
        self, model, setup_backend
    ):
        """Resolving a missing conversation → 404 ``conversation_not_found``.

        Pinned at
        ``routers/common/background/create.rs::append_conversation_items`` →
        ``conv_storage.get_conversation`` ``Ok(None)`` branch.
        """
        _, model_path, _, gw = setup_backend
        resp = _post_responses(
            gw,
            {
                "model": model_path,
                "input": "hello",
                "background": True,
                "store": True,
                "conversation": "conv_does_not_exist_zzz",
            },
        )
        _assert_handler_error(resp, 404, "conversation_not_found")


# ===========================================================================
# history_backend=none → background disabled at the handler layer
# ===========================================================================


@pytest.mark.vendor("openai")
@pytest.mark.gpu(0)
@pytest.mark.storage("none")
@pytest.mark.parametrize("setup_backend", ["openai"], indirect=True)
class TestBackgroundModeUnsupportedBackend:
    """When the gateway runs with ``--history-backend none``,
    ``StorageBundle.background_repository == None`` (``factory.rs:84-93``),
    so any ``background=true`` request must fail fast with
    ``background_not_supported`` — the shared handler's repo guard.
    """

    def test_background_returns_400_when_repo_missing(self, setup_backend):
        _, model_path, _, gw = setup_backend
        resp = _post_responses(
            gw,
            {
                "model": model_path,
                "input": "hello",
                "background": True,
                "store": True,
            },
        )
        _assert_handler_error(resp, 400, "background_not_supported")
