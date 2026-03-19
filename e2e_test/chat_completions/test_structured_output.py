"""Structured Output E2E Tests for Chat Completions.

Tests for response_format json_schema/json_object constrained output.
"""

from __future__ import annotations

import json

import pytest

# =============================================================================
# Base class (not collected by pytest due to underscore prefix)
# =============================================================================


class _TestStructuredOutputBase:
    """Base class for structured output tests. Not collected by pytest."""

    def test_response_format_json_schema(self, setup_backend, smg):
        """Test response_format with json_schema produces valid constrained JSON."""
        _, model, client, gateway = setup_backend

        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "maxItems": 2,
                },
            },
            "required": ["items"],
            "additionalProperties": False,
        }

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "List exactly 2 fruits."},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "fruits",
                    "strict": True,
                    "schema": schema,
                },
            },
            temperature=0,
        )

        content = response.choices[0].message.content
        assert content is not None, "Expected non-empty content"

        parsed = json.loads(content)
        assert set(parsed) == {"items"}
        assert isinstance(parsed["items"], list)
        assert len(parsed["items"]) == 2
        assert all(isinstance(item, str) for item in parsed["items"])

        # Should not contain leaked special tokens
        assert "<|channel|>" not in content
        assert "<|constrain|>" not in content
        assert "<|message|>" not in content

    def test_response_format_json_schema_stream(self, setup_backend, smg):
        """Test response_format with json_schema in streaming mode."""
        _, model, client, gateway = setup_backend

        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "maxItems": 2,
                },
            },
            "required": ["items"],
            "additionalProperties": False,
        }

        chunks = list(
            client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": "List exactly 2 fruits."},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "fruits",
                        "strict": True,
                        "schema": schema,
                    },
                },
                temperature=0,
                stream=True,
            )
        )

        content = ""
        for chunk in chunks:
            if chunk.choices and chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content

        assert content, "Expected non-empty streamed content"

        parsed = json.loads(content)
        assert set(parsed) == {"items"}
        assert isinstance(parsed["items"], list)
        assert len(parsed["items"]) == 2
        assert all(isinstance(item, str) for item in parsed["items"])

        # Should not contain leaked special tokens
        assert "<|channel|>" not in content
        assert "<|constrain|>" not in content
        assert "<|message|>" not in content


# =============================================================================
# Regular model (Llama 3.1 8B)
# =============================================================================


@pytest.mark.engine("sglang", "vllm", "trtllm")
@pytest.mark.gpu(1)
@pytest.mark.model("meta-llama/Llama-3.1-8B-Instruct")
@pytest.mark.gateway(extra_args=["--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestStructuredOutputRegular(_TestStructuredOutputBase):
    """Structured output tests for regular models (Llama)."""


# =============================================================================
# Harmony model (GPT-OSS)
# =============================================================================


@pytest.mark.engine("sglang")
@pytest.mark.gpu(2)
@pytest.mark.model("openai/gpt-oss-20b")
@pytest.mark.gateway(extra_args=["--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestStructuredOutputHarmony(_TestStructuredOutputBase):
    """Structured output tests for Harmony models (GPT-OSS).

    Inherits all tests from _TestStructuredOutputBase. The Harmony path uses
    structural tags for json_schema enforcement on the final channel.
    """
