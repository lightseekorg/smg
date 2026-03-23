# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for field_transforms.py — pure dict transformations."""

import pytest

from smg_grpc_servicer.vllm.field_transforms import (
    FIELD_TRANSFORMS,
    _ensure_message_content,
    _parse_tool_choice,
    flatten_completion_prompt,
)


class TestFlattenCompletionPrompt:
    """Tests for flatten_completion_prompt()."""

    def test_text_prompt(self):
        assert flatten_completion_prompt({"text": "hello world"}) == "hello world"

    def test_texts_prompt(self):
        result = flatten_completion_prompt({"texts": {"texts": ["a", "b", "c"]}})
        assert result == ["a", "b", "c"]

    def test_token_ids_prompt(self):
        result = flatten_completion_prompt({"token_ids": {"token_ids": [1, 2, 3]}})
        assert result == [1, 2, 3]
        assert all(isinstance(x, int) for x in result)

    def test_token_ids_coerces_strings(self):
        """Proto uint32 may arrive as strings via MessageToDict."""
        result = flatten_completion_prompt({"token_ids": {"token_ids": ["1", "2"]}})
        assert result == [1, 2]

    def test_token_id_batches_prompt(self):
        result = flatten_completion_prompt(
            {
                "token_id_batches": {
                    "batches": [
                        {"token_ids": [1, 2]},
                        {"token_ids": [3, 4, 5]},
                    ]
                }
            }
        )
        assert result == [[1, 2], [3, 4, 5]]

    def test_non_dict_passthrough(self):
        assert flatten_completion_prompt("hello") == "hello"
        assert flatten_completion_prompt(42) == 42
        assert flatten_completion_prompt(None) is None

    def test_empty_dict_raises_value_error(self):
        with pytest.raises(ValueError, match="no supported oneof field set"):
            flatten_completion_prompt({})

    def test_unknown_keys_raises_value_error(self):
        with pytest.raises(ValueError, match="no supported oneof field set"):
            flatten_completion_prompt({"unknown": "value"})

    def test_empty_texts_returns_empty_list(self):
        """MessageToDict drops empty repeated fields, producing {"texts": {}}."""
        assert flatten_completion_prompt({"texts": {}}) == []

    def test_empty_token_ids_returns_empty_list(self):
        assert flatten_completion_prompt({"token_ids": {}}) == []

    def test_empty_token_id_batches_returns_empty_list(self):
        assert flatten_completion_prompt({"token_id_batches": {}}) == []

    def test_batch_with_empty_token_ids(self):
        result = flatten_completion_prompt({"token_id_batches": {"batches": [{}]}})
        assert result == [[]]


class TestEnsureMessageContent:
    """Tests for _ensure_message_content()."""

    def test_adds_content_when_missing(self):
        messages = [{"role": "assistant", "tool_calls": []}]
        result = _ensure_message_content(messages)
        assert result[0]["content"] is None

    def test_preserves_existing_content(self):
        messages = [{"role": "user", "content": "hello"}]
        result = _ensure_message_content(messages)
        assert result[0]["content"] == "hello"

    def test_multiple_messages(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant"},
            {"role": "tool", "content": "result"},
        ]
        result = _ensure_message_content(messages)
        assert result[0]["content"] == "hi"
        assert result[1]["content"] is None
        assert result[2]["content"] == "result"

    def test_mutates_in_place(self):
        messages = [{"role": "user"}]
        result = _ensure_message_content(messages)
        assert result is messages

    def test_empty_list(self):
        assert _ensure_message_content([]) == []

    def test_skips_non_dict_items(self):
        messages = [{"role": "user"}, "not_a_dict", 42]
        result = _ensure_message_content(messages)
        assert result[0]["content"] is None
        assert result[1] == "not_a_dict"
        assert result[2] == 42


class TestParseToolChoice:
    """Tests for _parse_tool_choice()."""

    def test_simple_string_passthrough(self):
        assert _parse_tool_choice("none") == "none"
        assert _parse_tool_choice("auto") == "auto"
        assert _parse_tool_choice("required") == "required"

    def test_json_object_parsed(self):
        result = _parse_tool_choice('{"type": "function", "function": {"name": "get_weather"}}')
        assert result == {"type": "function", "function": {"name": "get_weather"}}

    def test_invalid_json_passthrough(self):
        assert _parse_tool_choice("not{json") == "not{json"

    def test_json_non_dict_passthrough(self):
        """JSON arrays or primitives stay as the original string."""
        assert _parse_tool_choice("[1, 2]") == "[1, 2]"
        assert _parse_tool_choice("123") == "123"

    def test_non_string_passthrough(self):
        assert _parse_tool_choice(42) == 42
        assert _parse_tool_choice(None) is None


class TestFieldTransformsDict:
    """Tests for the FIELD_TRANSFORMS constant."""

    def test_has_expected_keys(self):
        assert set(FIELD_TRANSFORMS.keys()) == {
            "parameters_json",
            "content_parts",
            "prompt",
            "messages",
            "tool_choice",
        }

    def test_parameters_json_transform(self):
        name, fn = FIELD_TRANSFORMS["parameters_json"]
        assert name == "parameters"
        assert fn('{"key": "value"}') == {"key": "value"}

    def test_content_parts_no_transform(self):
        name, fn = FIELD_TRANSFORMS["content_parts"]
        assert name == "content"
        assert fn is None

    def test_prompt_uses_flatten(self):
        name, fn = FIELD_TRANSFORMS["prompt"]
        assert name == "prompt"
        assert fn is flatten_completion_prompt

    def test_messages_uses_ensure_content(self):
        name, fn = FIELD_TRANSFORMS["messages"]
        assert name == "messages"
        assert fn is _ensure_message_content

    def test_tool_choice_uses_parse(self):
        name, fn = FIELD_TRANSFORMS["tool_choice"]
        assert name == "tool_choice"
        assert fn is _parse_tool_choice
