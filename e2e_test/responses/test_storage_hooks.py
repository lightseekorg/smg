"""E2E tests for WASM storage hooks with the Responses API.

Validates that WASM storage hooks integrate with the full gateway pipeline:
- Gateway loads a WASM hook component from disk
- Hook runs before/after every storage operation
- Responses API works correctly with hooks active

The passthrough hook adds a marker extra column (HOOK_ACTIVE=true)
on write operations but never rejects, so normal API behavior is preserved.
"""

from __future__ import annotations

import logging

import pytest
from conftest import smg_compare

logger = logging.getLogger(__name__)

# Path relative to the workspace root (where the gateway binary runs)
PASSTHROUGH_HOOK_PATH = "crates/wasm/tests/fixtures/storage_hook_passthrough.wasm"


@pytest.mark.gateway(extra_args=["--storage-hook-wasm-path", PASSTHROUGH_HOOK_PATH])
@pytest.mark.parametrize("setup_backend", ["openai"], indirect=True)
class TestResponsesWithStorageHook:
    """Verify WASM storage hooks work end-to-end with the Responses API."""

    def test_create_and_get_response_with_hook(self, setup_backend, smg):
        """Responses API works normally when a WASM storage hook is active."""
        _, model, client, gateway = setup_backend

        # Create response — hook runs before(StoreResponse) and after(StoreResponse)
        resp = client.responses.create(model=model, input="Hello with hooks!")
        assert resp.id is not None
        assert resp.error is None
        assert resp.status == "completed"
        assert len(resp.output_text) > 0

        # Retrieve — hook runs before(GetResponse) and after(GetResponse)
        get_resp = client.responses.retrieve(response_id=resp.id)
        assert get_resp.id == resp.id
        assert get_resp.error is None
        assert get_resp.status == "completed"

        # SmgClient comparison
        with smg_compare():
            smg_create = smg.responses.create(model=model, input="Hello with hooks!")
            assert smg_create.id is not None
            assert smg_create.error is None
            assert smg_create.status == "completed"
            assert len(smg_create.output_text) > 0
            # Get response
            smg_get = smg.responses.retrieve(response_id=smg_create.id)
            assert smg_get.id == smg_create.id
            assert smg_get.status == "completed"

    def test_conversation_with_previous_response_with_hook(self, setup_backend, smg):
        """Multi-turn conversation works with hooks active."""
        _, model, client, gateway = setup_backend

        # First turn
        resp1 = client.responses.create(model=model, input="What is 2+2?")
        assert resp1.id is not None
        assert resp1.status == "completed"

        # Second turn referencing the first
        resp2 = client.responses.create(
            model=model,
            input="Now add 3 to that",
            previous_response_id=resp1.id,
        )
        assert resp2.id is not None
        assert resp2.status == "completed"

        # Both should be retrievable
        get1 = client.responses.retrieve(response_id=resp1.id)
        assert get1.id == resp1.id
        get2 = client.responses.retrieve(response_id=resp2.id)
        assert get2.id == resp2.id

        # SmgClient comparison
        with smg_compare():
            smg_resp = smg.responses.create(model=model, input="What is 2+2?")
            assert smg_resp.id is not None
            assert smg_resp.status == "completed"

    def test_input_items_list_with_hook(self, setup_backend, smg):
        """Input items listing works with hooks active."""
        _, model, client, gateway = setup_backend

        resp = client.responses.create(model=model, input="Hello!")
        assert resp.id is not None
        assert resp.status == "completed"

        input_items = client.responses.input_items.list(response_id=resp.id)
        assert input_items.data is not None
        assert len(input_items.data) > 0

        # SmgClient comparison
        with smg_compare():
            smg_resp = smg.responses.create(model=model, input="Hello!")
            assert smg_resp.id is not None
            assert smg_resp.status == "completed"
            smg_items = smg.responses.input_items.list(response_id=smg_resp.id)
            assert smg_items.data is not None
            assert len(smg_items.data) > 0
