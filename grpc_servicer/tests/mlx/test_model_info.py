"""Tests for GetModelInfo and GetServerInfo RPCs."""

import time
from unittest.mock import AsyncMock

import pytest

from smg_grpc_proto import vllm_engine_pb2

from smg_grpc_servicer.mlx.servicer import MlxEngineServicer


@pytest.fixture
def mock_servicer():
    servicer = object.__new__(MlxEngineServicer)
    servicer.model_path = "mlx-community/SmolLM-135M-4bit"
    servicer.model_config = {
        "model_type": "llama",
        "vocab_size": 32000,
        "max_position_embeddings": 4096,
        "architectures": ["LlamaForCausalLM"],
        "eos_token_id": 2,
        "bos_token_id": 1,
        "pad_token_id": 0,
    }
    servicer._eos_token_ids = [2]
    servicer._active_requests = 0
    servicer.start_time = time.time()
    return servicer


@pytest.fixture
def context():
    return AsyncMock()


class TestGetModelInfo:
    @pytest.mark.asyncio
    async def test_basic_fields(self, mock_servicer, context):
        request = vllm_engine_pb2.GetModelInfoRequest()
        resp = await MlxEngineServicer.GetModelInfo(mock_servicer, request, context)
        assert resp.model_path == "mlx-community/SmolLM-135M-4bit"
        assert resp.vocab_size == 32000
        assert resp.max_context_length == 4096
        assert resp.is_generation is True
        assert resp.model_type == "llama"
        assert list(resp.architectures) == ["LlamaForCausalLM"]
        assert list(resp.eos_token_ids) == [2]
        assert resp.bos_token_id == 1
        assert resp.pad_token_id == 0

    @pytest.mark.asyncio
    async def test_eos_as_list(self, mock_servicer, context):
        mock_servicer.model_config["eos_token_id"] = [2, 128001]
        mock_servicer._eos_token_ids = [2, 128001]
        request = vllm_engine_pb2.GetModelInfoRequest()
        resp = await MlxEngineServicer.GetModelInfo(mock_servicer, request, context)
        assert list(resp.eos_token_ids) == [2, 128001]

    @pytest.mark.asyncio
    async def test_missing_optional_fields(self, mock_servicer, context):
        del mock_servicer.model_config["pad_token_id"]
        del mock_servicer.model_config["bos_token_id"]
        request = vllm_engine_pb2.GetModelInfoRequest()
        resp = await MlxEngineServicer.GetModelInfo(mock_servicer, request, context)
        assert resp.pad_token_id == 0
        assert resp.bos_token_id == 0


class TestGetServerInfo:
    @pytest.mark.asyncio
    async def test_returns_mlx_server_type(self, mock_servicer, context):
        request = vllm_engine_pb2.GetServerInfoRequest()
        resp = await MlxEngineServicer.GetServerInfo(mock_servicer, request, context)
        assert resp.server_type == "mlx-grpc"
        assert resp.kv_connector == ""
        assert resp.kv_role == ""

    @pytest.mark.asyncio
    async def test_active_requests(self, mock_servicer, context):
        mock_servicer._active_requests = 5
        request = vllm_engine_pb2.GetServerInfoRequest()
        resp = await MlxEngineServicer.GetServerInfo(mock_servicer, request, context)
        assert resp.active_requests == 5

    @pytest.mark.asyncio
    async def test_uptime(self, mock_servicer, context):
        mock_servicer.start_time = time.time() - 60.0
        request = vllm_engine_pb2.GetServerInfoRequest()
        resp = await MlxEngineServicer.GetServerInfo(mock_servicer, request, context)
        assert resp.uptime_seconds >= 59.0
