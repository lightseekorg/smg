# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for render_servicer.py — the RenderGrpcServicer gRPC service."""

import time
from unittest.mock import MagicMock, patch

import grpc
import pytest
from smg_grpc_servicer.vllm.render_servicer import RenderGrpcServicer

# Module path prefix for patching
_MOD = "smg_grpc_servicer.vllm.render_servicer"


class TestInit:
    def test_stores_state_and_start_time(self, mock_state):
        servicer = RenderGrpcServicer(mock_state, start_time=1000.0)
        assert servicer.state is mock_state
        assert servicer.start_time == 1000.0


class TestHealthCheck:
    async def test_returns_healthy(self, mock_state, mock_grpc_context):
        servicer = RenderGrpcServicer(mock_state, start_time=1000.0)
        response = await servicer.HealthCheck(MagicMock(), mock_grpc_context)
        assert response.healthy is True
        assert response.message == "Healthy"


class TestGetModelInfo:
    async def test_returns_model_attributes(self, mock_state, mock_grpc_context):
        servicer = RenderGrpcServicer(mock_state, start_time=1000.0)
        response = await servicer.GetModelInfo(MagicMock(), mock_grpc_context)
        assert response.model_path == "/models/llama-3.1-8b"
        assert response.is_generation is True
        assert response.max_context_length == 4096
        assert response.vocab_size == 32000
        assert response.supports_vision is False
        assert response.served_model_name == "/models/llama-3.1-8b"

    async def test_served_model_name_override(self, mock_state, mock_grpc_context):
        mock_state.vllm_config.model_config.served_model_name = "my-model"
        servicer = RenderGrpcServicer(mock_state, start_time=1000.0)
        response = await servicer.GetModelInfo(MagicMock(), mock_grpc_context)
        assert response.served_model_name == "my-model"

    async def test_non_generation_model(self, mock_state, mock_grpc_context):
        mock_state.vllm_config.model_config.runner_type = "encode"
        servicer = RenderGrpcServicer(mock_state, start_time=1000.0)
        response = await servicer.GetModelInfo(MagicMock(), mock_grpc_context)
        assert response.is_generation is False

    async def test_multimodal_model(self, mock_state, mock_grpc_context):
        mock_state.vllm_config.model_config.is_multimodal_model = True
        servicer = RenderGrpcServicer(mock_state, start_time=1000.0)
        response = await servicer.GetModelInfo(MagicMock(), mock_grpc_context)
        assert response.supports_vision is True


class TestGetServerInfo:
    async def test_returns_server_info(self, mock_state, mock_grpc_context):
        servicer = RenderGrpcServicer(mock_state, start_time=1000.0)
        before = time.time()
        response = await servicer.GetServerInfo(MagicMock(), mock_grpc_context)
        after = time.time()

        assert response.server_type == "vllm-render-grpc"
        assert response.uptime_seconds > 0
        assert before <= response.last_receive_timestamp <= after
        assert response.active_requests == 0
        assert response.is_paused is False
        assert response.kv_connector == ""
        assert response.kv_role == ""


class TestRenderChat:
    @patch(f"{_MOD}.vllm_render_pb2")
    @patch(f"{_MOD}.pydantic_to_proto")
    @patch(f"{_MOD}.from_proto")
    async def test_success(
        self,
        mock_from_proto,
        mock_to_proto,
        mock_pb2,
        mock_state,
        mock_grpc_context,
    ):
        mock_pydantic_req = MagicMock()
        mock_from_proto.return_value = mock_pydantic_req

        mock_render_result = MagicMock()
        mock_state.openai_serving_render.render_chat_request.return_value = mock_render_result

        mock_proto_response = MagicMock()
        mock_to_proto.return_value = mock_proto_response

        servicer = RenderGrpcServicer(mock_state, start_time=1000.0)
        await servicer.RenderChat(MagicMock(), mock_grpc_context)

        mock_from_proto.assert_called_once()
        mock_state.openai_serving_render.render_chat_request.assert_awaited_once_with(
            mock_pydantic_req
        )
        mock_to_proto.assert_called_once()
        mock_grpc_context.abort.assert_not_awaited()

    async def test_not_configured(self, mock_state, mock_grpc_context):
        mock_state.openai_serving_render = None
        servicer = RenderGrpcServicer(mock_state, start_time=1000.0)

        with pytest.raises(grpc.aio.AbortError):
            await servicer.RenderChat(MagicMock(), mock_grpc_context)

        mock_grpc_context.abort.assert_awaited_once_with(
            grpc.StatusCode.UNIMPLEMENTED,
            "RenderChat is not configured on this server.",
        )

    @patch(f"{_MOD}.from_proto")
    async def test_error_response(self, mock_from_proto, mock_state, mock_grpc_context):
        mock_from_proto.return_value = MagicMock()

        mock_error = MagicMock()
        mock_error.error.message = "Invalid model"
        mock_state.openai_serving_render.render_chat_request.return_value = mock_error

        with patch(f"{_MOD}.ErrorResponse", new=type(mock_error)):
            servicer = RenderGrpcServicer(mock_state, start_time=1000.0)
            with pytest.raises(grpc.aio.AbortError):
                await servicer.RenderChat(MagicMock(), mock_grpc_context)

        mock_grpc_context.abort.assert_awaited_once_with(
            grpc.StatusCode.INVALID_ARGUMENT,
            "Invalid model",
        )

    @patch(f"{_MOD}.from_proto")
    async def test_value_error_returns_invalid_argument(
        self, mock_from_proto, mock_state, mock_grpc_context
    ):
        mock_from_proto.side_effect = ValueError("bad input")
        servicer = RenderGrpcServicer(mock_state, start_time=1000.0)

        with pytest.raises(grpc.aio.AbortError):
            await servicer.RenderChat(MagicMock(), mock_grpc_context)

        mock_grpc_context.abort.assert_awaited_once_with(
            grpc.StatusCode.INVALID_ARGUMENT,
            "bad input",
        )

    @patch(f"{_MOD}.from_proto")
    async def test_unexpected_exception_returns_internal(
        self, mock_from_proto, mock_state, mock_grpc_context
    ):
        mock_from_proto.side_effect = RuntimeError("server crash")
        servicer = RenderGrpcServicer(mock_state, start_time=1000.0)

        with pytest.raises(grpc.aio.AbortError):
            await servicer.RenderChat(MagicMock(), mock_grpc_context)

        mock_grpc_context.abort.assert_awaited_once_with(
            grpc.StatusCode.INTERNAL,
            "server crash",
        )

    @patch(f"{_MOD}.from_proto")
    async def test_abort_error_propagates(self, mock_from_proto, mock_state, mock_grpc_context):
        mock_from_proto.side_effect = grpc.aio.AbortError("", "")
        servicer = RenderGrpcServicer(mock_state, start_time=1000.0)

        with pytest.raises(grpc.aio.AbortError):
            await servicer.RenderChat(MagicMock(), mock_grpc_context)

    @patch(f"{_MOD}.pydantic_to_proto")
    @patch(f"{_MOD}.from_proto")
    async def test_serialization_error_returns_internal(
        self, mock_from_proto, mock_to_proto, mock_state, mock_grpc_context
    ):
        mock_from_proto.return_value = MagicMock()
        mock_state.openai_serving_render.render_chat_request.return_value = MagicMock()
        mock_to_proto.side_effect = TypeError("serialization bug")

        servicer = RenderGrpcServicer(mock_state, start_time=1000.0)
        with pytest.raises(grpc.aio.AbortError):
            await servicer.RenderChat(MagicMock(), mock_grpc_context)

        mock_grpc_context.abort.assert_awaited_once_with(
            grpc.StatusCode.INTERNAL,
            "serialization bug",
        )


class TestRenderCompletion:
    @patch(f"{_MOD}.vllm_render_pb2")
    @patch(f"{_MOD}.pydantic_to_proto")
    @patch(f"{_MOD}.from_proto")
    async def test_success(
        self,
        mock_from_proto,
        mock_to_proto,
        mock_pb2,
        mock_state,
        mock_grpc_context,
    ):
        mock_pydantic_req = MagicMock()
        mock_from_proto.return_value = mock_pydantic_req

        mock_result_1 = MagicMock()
        mock_result_2 = MagicMock()
        mock_state.openai_serving_render.render_completion_request.return_value = [
            mock_result_1,
            mock_result_2,
        ]

        mock_to_proto.side_effect = [MagicMock(), MagicMock()]

        servicer = RenderGrpcServicer(mock_state, start_time=1000.0)
        await servicer.RenderCompletion(MagicMock(), mock_grpc_context)

        mock_from_proto.assert_called_once()
        assert mock_to_proto.call_count == 2
        mock_grpc_context.abort.assert_not_awaited()

    async def test_not_configured(self, mock_state, mock_grpc_context):
        mock_state.openai_serving_render = None
        servicer = RenderGrpcServicer(mock_state, start_time=1000.0)

        with pytest.raises(grpc.aio.AbortError):
            await servicer.RenderCompletion(MagicMock(), mock_grpc_context)

        mock_grpc_context.abort.assert_awaited_once_with(
            grpc.StatusCode.UNIMPLEMENTED,
            "RenderCompletion is not configured on this server.",
        )

    @patch(f"{_MOD}.from_proto")
    async def test_error_response(self, mock_from_proto, mock_state, mock_grpc_context):
        mock_from_proto.return_value = MagicMock()

        mock_error = MagicMock()
        mock_error.error.message = "Bad prompt"
        mock_state.openai_serving_render.render_completion_request.return_value = mock_error

        with patch(f"{_MOD}.ErrorResponse", new=type(mock_error)):
            servicer = RenderGrpcServicer(mock_state, start_time=1000.0)
            with pytest.raises(grpc.aio.AbortError):
                await servicer.RenderCompletion(MagicMock(), mock_grpc_context)

        mock_grpc_context.abort.assert_awaited_once_with(
            grpc.StatusCode.INVALID_ARGUMENT,
            "Bad prompt",
        )

    @patch(f"{_MOD}.from_proto")
    async def test_value_error_returns_invalid_argument(
        self, mock_from_proto, mock_state, mock_grpc_context
    ):
        mock_from_proto.side_effect = ValueError("bad input")
        servicer = RenderGrpcServicer(mock_state, start_time=1000.0)

        with pytest.raises(grpc.aio.AbortError):
            await servicer.RenderCompletion(MagicMock(), mock_grpc_context)

        mock_grpc_context.abort.assert_awaited_once_with(
            grpc.StatusCode.INVALID_ARGUMENT,
            "bad input",
        )

    @patch(f"{_MOD}.from_proto")
    async def test_unexpected_exception_returns_internal(
        self, mock_from_proto, mock_state, mock_grpc_context
    ):
        mock_from_proto.side_effect = RuntimeError("server crash")
        servicer = RenderGrpcServicer(mock_state, start_time=1000.0)

        with pytest.raises(grpc.aio.AbortError):
            await servicer.RenderCompletion(MagicMock(), mock_grpc_context)

        mock_grpc_context.abort.assert_awaited_once_with(
            grpc.StatusCode.INTERNAL,
            "server crash",
        )

    @patch(f"{_MOD}.from_proto")
    async def test_abort_error_propagates(self, mock_from_proto, mock_state, mock_grpc_context):
        mock_from_proto.side_effect = grpc.aio.AbortError("", "")
        servicer = RenderGrpcServicer(mock_state, start_time=1000.0)

        with pytest.raises(grpc.aio.AbortError):
            await servicer.RenderCompletion(MagicMock(), mock_grpc_context)

    @patch(f"{_MOD}.pydantic_to_proto")
    @patch(f"{_MOD}.from_proto")
    async def test_serialization_error_returns_internal(
        self, mock_from_proto, mock_to_proto, mock_state, mock_grpc_context
    ):
        mock_from_proto.return_value = MagicMock()
        mock_result = MagicMock()
        mock_state.openai_serving_render.render_completion_request.return_value = [mock_result]
        mock_to_proto.side_effect = TypeError("serialization bug")

        servicer = RenderGrpcServicer(mock_state, start_time=1000.0)
        with pytest.raises(grpc.aio.AbortError):
            await servicer.RenderCompletion(MagicMock(), mock_grpc_context)

        mock_grpc_context.abort.assert_awaited_once_with(
            grpc.StatusCode.INTERNAL,
            "serialization bug",
        )
