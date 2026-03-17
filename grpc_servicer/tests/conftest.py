# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared fixtures for smg_grpc_servicer unit tests."""

from unittest.mock import AsyncMock, MagicMock

import grpc
import pytest


@pytest.fixture
def mock_grpc_context():
    """Mock gRPC async servicer context.

    abort() raises AbortError like the real implementation.
    """
    ctx = MagicMock()
    ctx.abort = AsyncMock(side_effect=grpc.aio.AbortError("", ""))
    return ctx


@pytest.fixture
def mock_model_config():
    """Mock vLLM model config with typical attributes."""
    mc = MagicMock()
    mc.model = "/models/llama-3.1-8b"
    mc.runner_type = "generate"
    mc.max_model_len = 4096
    mc.get_vocab_size.return_value = 32000
    mc.is_multimodal_model = False
    return mc


@pytest.fixture
def mock_state(mock_model_config):
    """Mock Starlette State with vllm_config and openai_serving_render."""
    state = MagicMock()
    state.vllm_config.model_config = mock_model_config
    state.openai_serving_render = MagicMock()
    state.openai_serving_render.render_chat_request = AsyncMock()
    state.openai_serving_render.render_completion_request = AsyncMock()
    return state
