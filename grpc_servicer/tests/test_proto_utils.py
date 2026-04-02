# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for proto_utils.py — protobuf conversion utilities."""

import json
from unittest.mock import MagicMock, patch

from smg_grpc_servicer.vllm.proto_utils import (
    _apply_transforms,
    from_proto,
    proto_to_dict,
    pydantic_to_proto,
)


class TestApplyTransforms:
    """Tests for _apply_transforms() — recursive field rename/transform."""

    def test_simple_rename(self):
        transforms = {"old_name": ("new_name", None)}
        assert _apply_transforms({"old_name": "value"}, transforms) == {"new_name": "value"}

    def test_rename_with_transform(self):
        transforms = {"count_str": ("count", int)}
        assert _apply_transforms({"count_str": "42"}, transforms) == {"count": 42}

    def test_non_matching_keys_preserved(self):
        transforms = {"x": ("y", None)}
        assert _apply_transforms({"x": 1, "z": 2}, transforms) == {"y": 1, "z": 2}

    def test_nested_dict(self):
        transforms = {"old": ("new", None)}
        assert _apply_transforms({"outer": {"old": "val"}}, transforms) == {"outer": {"new": "val"}}

    def test_list_of_dicts(self):
        transforms = {"old": ("new", None)}
        assert _apply_transforms([{"old": 1}, {"old": 2}], transforms) == [
            {"new": 1},
            {"new": 2},
        ]

    def test_nested_list_in_dict(self):
        transforms = {"k": ("renamed", None)}
        result = _apply_transforms({"items": [{"k": "a"}, {"k": "b"}]}, transforms)
        assert result == {"items": [{"renamed": "a"}, {"renamed": "b"}]}

    def test_scalar_passthrough(self):
        transforms = {"x": ("y", None)}
        assert _apply_transforms("hello", transforms) == "hello"
        assert _apply_transforms(42, transforms) == 42
        assert _apply_transforms(None, transforms) is None

    def test_empty_dict(self):
        assert _apply_transforms({}, {"x": ("y", None)}) == {}

    def test_empty_list(self):
        assert _apply_transforms([], {"x": ("y", None)}) == []

    def test_transform_receives_recursed_value(self):
        """Transform fn receives the already-recursed value."""
        transforms = {"nested": ("flat", lambda d: sorted(d.keys()) if isinstance(d, dict) else d)}
        result = _apply_transforms({"nested": {"b": 1, "a": 2}}, transforms)
        assert result == {"flat": ["a", "b"]}


class TestProtoToDict:
    """Tests for proto_to_dict()."""

    @patch("smg_grpc_servicer.vllm.proto_utils.MessageToDict")
    def test_calls_message_to_dict(self, mock_msg_to_dict):
        mock_message = MagicMock()
        mock_msg_to_dict.return_value = {"field": "value"}

        result = proto_to_dict(mock_message)

        mock_msg_to_dict.assert_called_once_with(mock_message, preserving_proto_field_name=True)
        assert result == {"field": "value"}

    @patch("smg_grpc_servicer.vllm.proto_utils.MessageToDict")
    def test_without_transforms(self, mock_msg_to_dict):
        mock_msg_to_dict.return_value = {"a": 1, "b": 2}
        assert proto_to_dict(MagicMock(), transforms=None) == {"a": 1, "b": 2}

    @patch("smg_grpc_servicer.vllm.proto_utils.MessageToDict")
    def test_with_transforms(self, mock_msg_to_dict):
        mock_msg_to_dict.return_value = {"parameters_json": '{"key": "val"}'}
        transforms = {"parameters_json": ("parameters", json.loads)}
        result = proto_to_dict(MagicMock(), transforms=transforms)
        assert result == {"parameters": {"key": "val"}}


class TestFromProto:
    """Tests for from_proto()."""

    @patch("smg_grpc_servicer.vllm.proto_utils.MessageToDict")
    def test_creates_instance_from_dict(self, mock_msg_to_dict):
        mock_msg_to_dict.return_value = {"model": "llama", "temperature": 0.7}
        mock_class = MagicMock()
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance

        result = from_proto(MagicMock(), mock_class)

        mock_class.assert_called_once_with(model="llama", temperature=0.7)
        assert result is mock_instance

    @patch("smg_grpc_servicer.vllm.proto_utils.MessageToDict")
    def test_applies_transforms_before_construction(self, mock_msg_to_dict):
        mock_msg_to_dict.return_value = {"parameters_json": '{"top_k": 10}'}
        transforms = {"parameters_json": ("parameters", json.loads)}
        mock_class = MagicMock()

        from_proto(MagicMock(), mock_class, transforms=transforms)

        mock_class.assert_called_once_with(parameters={"top_k": 10})


class TestPydanticToProto:
    """Tests for pydantic_to_proto()."""

    @patch("smg_grpc_servicer.vllm.proto_utils.ParseDict")
    def test_converts_model_to_proto(self, mock_parse_dict):
        mock_model = MagicMock()
        mock_model.model_dump.return_value = {"request_id": "abc", "token_ids": [1, 2]}
        mock_message_class = MagicMock()
        mock_proto_instance = MagicMock()
        mock_message_class.return_value = mock_proto_instance
        mock_parse_dict.return_value = mock_proto_instance

        result = pydantic_to_proto(mock_model, mock_message_class)

        mock_model.model_dump.assert_called_once_with(mode="json", exclude_none=True)
        mock_parse_dict.assert_called_once_with(
            {"request_id": "abc", "token_ids": [1, 2]},
            mock_proto_instance,
        )
        assert result is mock_proto_instance
