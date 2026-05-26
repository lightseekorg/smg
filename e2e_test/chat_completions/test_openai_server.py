"""Chat Completions API E2E Tests - OpenAI Server Compatibility.

Tests for OpenAI-compatible chat completions API through the gateway.

Source: Migrated from e2e_grpc/basic/test_openai_server.py
"""

from __future__ import annotations

import json
import logging
import re
import textwrap

import pytest

logger = logging.getLogger(__name__)


# System prompt and tool surface used by the gpt-oss post-tool integrity check.
# Style-constrained guidance plus a multi-tool nested-args schema gives the
# model enough structure to produce a non-trivial post-tool continuation.
_GPT_OSS_POST_TOOL_SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are an engineering project assistant. Help engineers triage tasks
    and route work to teammates only when the engineer asks for it.

    ## Style rules
    - Reply in one short paragraph, at most three sentences.
    - Do not use markdown. No asterisks, bold, italic, headings, or code blocks.
    - Do not use bullet lists or tables.
    - Spell out small integers in words rather than digits.

    ## Action rules
    - Before calling any action tool, restate the request and ask the engineer
      to confirm. A clear yes counts as confirmation; a vague restatement does not.
    - After an action tool returns, acknowledge the result in one short
      paragraph. Do not produce additional analysis text or narrate reasoning.
    - Informational questions use search_engineer or lookup_runbook; do not
      call action tools for informational questions.
    """
)

_GPT_OSS_POST_TOOL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "assign_task",
            "description": (
                "Assign one or more open engineering tasks to teammates. "
                "Each assignment names the task id and the assignee."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "assignments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "task_id": {
                                    "type": "string",
                                    "description": "Identifier of the task.",
                                },
                                "assignee": {
                                    "type": "string",
                                    "description": "Username of the assignee.",
                                },
                            },
                            "required": ["task_id", "assignee"],
                        },
                    },
                    "priority": {
                        "type": "string",
                        "description": "Priority label, spelled out.",
                    },
                    "note": {
                        "type": "string",
                        "description": "Short note for the assignees.",
                    },
                },
                "required": ["assignments"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_status",
            "description": "Update the status of an engineering task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "status": {"type": "string"},
                    "comment": {"type": "string"},
                },
                "required": ["task_id", "status"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_task",
            "description": "Escalate a task to a higher priority queue.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "target_queue": {"type": "string"},
                },
                "required": ["task_id", "target_queue"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "close_task",
            "description": "Close an engineering task with a resolution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "resolution": {"type": "string"},
                },
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_engineer",
            "description": "Find engineers matching a skill or availability.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_runbook",
            "description": "Look up an engineering runbook by topic.",
            "parameters": {
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": [],
            },
        },
    },
]

# Detection patterns for malformed text in the post-tool assistant message:
# runs of invisible characters, ellipsis loops, long whitespace runs, and
# leaked English self-analysis. Tuned to flag runs (3+) of invisibles so
# isolated occurrences in normal text do not produce false positives.
_INVISIBLE_RUN = re.compile("[​‌‍⁠﻿   ]{3,}")
_ENGLISH_SELF_ANALYSIS = re.compile(
    r"\b(The assistant|We need to|We should|We must|"
    r"According to (?:rules|guidelines|the spec)|Sorry,? (?:we|the))\b"
)
_WHITESPACE_RUN = re.compile(r"\s{6,}")
_ELLIPSIS_LOOP = re.compile(r"(?:\.\.\.\s*){3,}|(?:…\s*){3,}")


# =============================================================================
# Chat Completion Tests (Llama 8B)
# =============================================================================


@pytest.mark.engine("sglang", "vllm", "trtllm", "tokenspeed")
@pytest.mark.gpu(1)
@pytest.mark.model("meta-llama/Llama-3.1-8B-Instruct")
@pytest.mark.gateway(extra_args=["--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
@pytest.mark.parametrize("api_client", ["openai", "smg"], indirect=True)
class TestChatCompletion:
    """Tests for OpenAI-compatible chat completions API."""

    # Whether the backend trims stop sequences from output.
    # Harmony (gpt-oss) does not trim because its detokenization is not channel-aware.
    STOP_SEQUENCE_TRIMMED = True

    @pytest.mark.parametrize(
        "logprobs",
        [
            None,
            pytest.param(
                5,
                marks=pytest.mark.skip_for_runtime(
                    "tokenspeed",
                    reason=(
                        "tokenspeed's --enable-top-logprobs is not yet implemented "
                        "(raises at startup); base output logprobs work via "
                        "--enable-output-logprobs but the test requires top_logprobs=5"
                    ),
                ),
            ),
        ],
    )
    @pytest.mark.parametrize("parallel_sample_num", [1, 2])
    def test_chat_completion(self, model, api_client, logprobs, parallel_sample_num):
        """Test non-streaming chat completion with logprobs and parallel sampling."""
        # Use temperature > 0 for n > 1 (greedy sampling rejects n > 1)
        temperature = 0.7 if parallel_sample_num > 1 else 0
        response = api_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant"},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What is the capital of France? Answer in a few words.",
                        }
                    ],
                },
            ],
            temperature=temperature,
            logprobs=logprobs is not None and logprobs > 0,
            top_logprobs=logprobs,
            n=parallel_sample_num,
        )

        if logprobs:
            assert isinstance(response.choices[0].logprobs.content[0].top_logprobs[0].token, str)
            ret_num_top_logprobs = len(response.choices[0].logprobs.content[0].top_logprobs)
            assert ret_num_top_logprobs == logprobs, f"{ret_num_top_logprobs} vs {logprobs}"

        assert len(response.choices) == parallel_sample_num
        assert response.choices[0].message.role == "assistant"
        assert isinstance(response.choices[0].message.content, str)
        assert response.id
        assert response.created
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

    @pytest.mark.parametrize(
        "logprobs",
        [
            None,
            pytest.param(
                5,
                marks=pytest.mark.skip_for_runtime(
                    "tokenspeed",
                    reason=(
                        "tokenspeed's --enable-top-logprobs is not yet implemented "
                        "(raises at startup); base output logprobs work via "
                        "--enable-output-logprobs but the test requires top_logprobs=5"
                    ),
                ),
            ),
        ],
    )
    @pytest.mark.parametrize("parallel_sample_num", [1, 2])
    def test_chat_completion_stream(self, model, api_client, logprobs, parallel_sample_num):
        """Test streaming chat completion with logprobs and parallel sampling."""
        temperature = 0.7 if parallel_sample_num > 1 else 0
        generator = api_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant"},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "What is the capital of France?"}],
                },
            ],
            temperature=temperature,
            logprobs=logprobs is not None and logprobs > 0,
            top_logprobs=logprobs,
            stream=True,
            stream_options={"include_usage": True},
            n=parallel_sample_num,
        )

        is_firsts = {}
        is_finished = {}
        finish_reason_counts = {}
        for response in generator:
            # Capture usage from the final chunk
            usage = response.usage
            if usage is not None:
                assert usage.prompt_tokens > 0, "usage.prompt_tokens was zero"
                assert usage.completion_tokens > 0, "usage.completion_tokens was zero"
                assert usage.total_tokens > 0, "usage.total_tokens was zero"
                continue

            # Skip if no choices
            if not response.choices:
                continue

            index = response.choices[0].index
            delta = response.choices[0].delta

            if index not in is_firsts:
                is_firsts[index] = True
                assert delta.role == "assistant"
                continue

            if response.choices[0].finish_reason:
                is_finished[index] = True
                finish_reason_counts[index] = finish_reason_counts.get(index, 0) + 1

            if logprobs and not is_finished.get(index, False):
                assert response.choices[0].logprobs is not None, "logprobs was not returned"
                assert len(response.choices[0].logprobs.content[0].top_logprobs) == logprobs, (
                    "top_logprobs count mismatch"
                )

        for index in range(parallel_sample_num):
            assert index in finish_reason_counts, f"No finish_reason found for index {index}"
            assert finish_reason_counts[index] == 1, (
                f"Expected 1 finish_reason chunk for index {index}, "
                f"got {finish_reason_counts[index]}"
            )

    @pytest.mark.skip_for_runtime(
        "trtllm",
        reason="TRT-LLM gRPC bug: uses 'guided_decoding_params' instead of 'guided_decoding'",
    )
    def test_regex(self, model, api_client):
        """Test structured output with regex constraint."""

        regex = (
            r"""\{\n""" + r"""   "name": "[\w]+",\n""" + r"""   "population": [\d]+\n""" + r"""\}"""
        )

        response = api_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant"},
                {"role": "user", "content": "Introduce the capital of France."},
            ],
            temperature=0,
            max_tokens=128,
            extra_body={"regex": regex},
        )
        text = response.choices[0].message.content

        try:
            js_obj = json.loads(text)
        except (TypeError, json.decoder.JSONDecodeError):
            raise
        assert isinstance(js_obj["name"], str)
        assert isinstance(js_obj["population"], int)

    def test_penalty(self, model, api_client):
        """Test that frequency_penalty parameter is accepted and produces output."""

        response = api_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "What is the capital of France?"},
            ],
            max_tokens=100,
            frequency_penalty=1.0,
            reasoning_effort="none",
        )
        assert isinstance(response.choices[0].message.content, str)
        assert response.usage.completion_tokens > 0

    def test_multi_content_parts(self, model, api_client):
        """Test that multiple content parts in a single message are all processed."""

        response = api_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is the capital of France?"},
                        {"type": "text", "text": "What is the capital of China?"},
                    ],
                },
            ],
            temperature=0,
            max_tokens=200,
        )
        content = response.choices[0].message.content
        assert isinstance(content, str)
        assert "paris" in content.lower(), f"Expected 'Paris' in response: {content}"
        assert "beijing" in content.lower(), f"Expected 'Beijing' in response: {content}"

    def test_response_prefill(self, model, api_client):
        """Test assistant message prefill with continue_final_message."""

        response = api_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant"},
                {
                    "role": "user",
                    "content": """
Extract the name, size, price, and color from this product description as a JSON object:

<description>
The SmartHome Mini is a compact smart home assistant available in black or white for only $49.99.
At just 5 inches wide, it lets you control lights, thermostats, and other connected devices via
voice or app—no matter where you place it in your home. This affordable little hub brings
convenient hands-free control to your smart devices.
</description>
""",
                },
                {
                    "role": "assistant",
                    "content": "{\n",
                },
            ],
            temperature=0,
            extra_body={"continue_final_message": True},
        )

        assert response.choices[0].message.content.strip().startswith('"name": "SmartHome Mini",')

    def test_streaming_token_count_matches_chunks(self, model, api_client):
        """Test that streaming completion_tokens matches the number of content chunks."""

        generator = api_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant"},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "What is the capital of France?"}],
                },
            ],
            temperature=0,
            max_tokens=50,
            stream=True,
            stream_options={"include_usage": True},
        )

        content_chunk_count = 0
        usage_completion_tokens = None

        for response in generator:
            if response.usage is not None:
                usage_completion_tokens = response.usage.completion_tokens
                continue
            if not response.choices:
                continue
            delta = response.choices[0].delta
            # Count chunks that have actual content (not just role or finish_reason)
            # Each chunk with content or reasoning_content represents one token
            if delta.content or getattr(delta, "reasoning_content", None):
                content_chunk_count += 1

        assert usage_completion_tokens is not None, "No usage chunk received"
        assert content_chunk_count > 0, "No content chunks received"
        # completion_tokens should be >= content chunks because some tokens
        # (like EOS) don't produce visible content
        assert usage_completion_tokens >= content_chunk_count, (
            f"completion_tokens ({usage_completion_tokens}) should be >= "
            f"content chunk count ({content_chunk_count})"
        )
        # But they should be close - allow small difference for special tokens
        token_tolerance = getattr(self, "STREAMING_TOKEN_TOLERANCE", 2)
        assert usage_completion_tokens - content_chunk_count <= token_tolerance, (
            f"completion_tokens ({usage_completion_tokens}) differs too much from "
            f"content chunk count ({content_chunk_count})"
        )

    def test_model_list(self, model, api_client):
        """Test listing available models."""

        models = list(api_client.models.list().data)
        assert len(models) == 1

    def test_stop_sequences(self, model, api_client):
        """Test that stop sequences cause the model to stop generating."""

        response = api_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Count from 1 to 10: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10"},
            ],
            temperature=0,
            max_tokens=200,
            stop=[","],
        )

        assert response.choices[0].finish_reason == "stop"
        msg = response.choices[0].message
        content = msg.content or getattr(msg, "reasoning_content", "") or ""
        if self.STOP_SEQUENCE_TRIMMED:
            assert "," not in content, f"Stop sequence ',' should not appear in output: {content}"
        else:
            assert content.endswith(","), (
                f"Stop sequence ',' should be the suffix of output: {content}"
            )

    def test_stop_sequences_stream(self, model, api_client):
        """Test that stop sequences work in streaming mode."""

        chunks = list(
            api_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": "Count from 1 to 10: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10",
                    },
                ],
                temperature=0,
                max_tokens=1024,
                stop=[","],
                stream=True,
            )
        )

        # Find the chunk with finish_reason
        finish_reasons = [
            c.choices[0].finish_reason for c in chunks if c.choices and c.choices[0].finish_reason
        ]
        assert "stop" in finish_reasons

        # Collect all content (fall back to reasoning_content for models like Harmony)
        content = "".join(
            self._delta_text(c.choices[0].delta)
            for c in chunks
            if c.choices and self._delta_text(c.choices[0].delta)
        )
        if self.STOP_SEQUENCE_TRIMMED:
            assert "," not in content, f"Stop sequence ',' should not appear in output: {content}"
        else:
            assert content.endswith(","), (
                f"Stop sequence ',' should be the suffix of output: {content}"
            )

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    @staticmethod
    def _delta_text(delta):
        """Extract text from delta, falling back to reasoning_content for Harmony."""
        return delta.content or getattr(delta, "reasoning_content", "") or ""


@pytest.mark.engine("sglang", "vllm", "trtllm", "tokenspeed")
@pytest.mark.gpu(2)
@pytest.mark.model("openai/gpt-oss-20b")
@pytest.mark.gateway(extra_args=["--history-backend", "memory"])
class TestChatCompletionGptOss(TestChatCompletion):
    """Tests for chat completions API with Harmony model (GPT-OSS).

    Inherits from TestChatCompletion and overrides tests that don't work
    with OSS models. Logprobs are supported via Harmony's built-in tokenizer.
    """

    # Harmony channel markers add ~10 special tokens
    STREAMING_TOKEN_TOLERANCE = 10

    STOP_SEQUENCE_TRIMMED = False

    @pytest.mark.parametrize(
        "logprobs",
        [
            None,
            pytest.param(
                5,
                marks=pytest.mark.skip_for_runtime(
                    "tokenspeed",
                    reason=(
                        "tokenspeed's --enable-top-logprobs is not yet implemented "
                        "(raises at startup); base output logprobs work via "
                        "--enable-output-logprobs but the test requires top_logprobs=5"
                    ),
                ),
            ),
        ],
    )
    @pytest.mark.parametrize("parallel_sample_num", [1, 2])
    def test_chat_completion(self, model, api_client, logprobs, parallel_sample_num):
        """Test non-streaming chat completion with logprobs and parallel sampling."""
        super().test_chat_completion(model, api_client, logprobs, parallel_sample_num)

    @pytest.mark.parametrize(
        "logprobs",
        [
            None,
            pytest.param(
                5,
                marks=pytest.mark.skip_for_runtime(
                    "tokenspeed",
                    reason=(
                        "tokenspeed's --enable-top-logprobs is not yet implemented "
                        "(raises at startup); base output logprobs work via "
                        "--enable-output-logprobs but the test requires top_logprobs=5"
                    ),
                ),
            ),
        ],
    )
    @pytest.mark.parametrize("parallel_sample_num", [1, 2])
    @pytest.mark.skip_for_runtime(
        "trtllm", reason="trtllm may return more top_logprobs than requested in streaming"
    )
    def test_chat_completion_stream(self, model, api_client, logprobs, parallel_sample_num):
        """Test streaming chat completion with logprobs and parallel sampling."""
        super().test_chat_completion_stream(model, api_client, logprobs, parallel_sample_num)

    def test_stop_sequences_stream(self, model, api_client):
        super().test_stop_sequences_stream(model, api_client)

    @pytest.mark.skip(reason="gpt-oss models don't support regex constraints")
    def test_regex(self, model, api_client):
        pass

    @pytest.mark.skip(reason="gpt-oss Harmony pipeline doesn't implement continue_final_message")
    def test_response_prefill(self, model, api_client):
        pass

    def test_post_tool_response_has_no_malformed_residue(self, model, api_client):
        """Post-tool-result final assistant message must not contain malformed text.

        Drives a multi-turn tool-call flow through the gateway and asserts the
        post-tool ``choices[].message.content`` is non-empty and contains no
        runs of invisible characters, long whitespace runs, ellipsis loops, or
        leaked English self-analysis. These properties are what the gpt-oss
        Harmony round-trip fix in PR #1547 is meant to guarantee on a
        successful HTTP 200 response after a tool round trip.
        """
        messages: list[dict] = [
            {"role": "system", "content": _GPT_OSS_POST_TOOL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Please assign task ENG dash four two zero one to "
                    "username alice with priority high and leave a note "
                    "saying needs review by end of day."
                ),
            },
        ]

        # Turn 1: model should ask for confirmation per the style rules.
        response1 = api_client.chat.completions.create(
            model=model,
            messages=messages,
            tools=_GPT_OSS_POST_TOOL_TOOLS,
            tool_choice="auto",
            reasoning_effort="high",
            temperature=0,
            max_tokens=512,
        )
        msg1 = response1.choices[0].message
        assistant_turn: dict = {"role": "assistant", "content": msg1.content or ""}
        if msg1.tool_calls:
            assistant_turn["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg1.tool_calls
            ]
        messages.append(assistant_turn)

        # If the model went straight to a tool call, skip the confirmation hop.
        tool_call_msg = msg1
        if not msg1.tool_calls:
            messages.append({"role": "user", "content": "yes, please confirm"})
            response2 = api_client.chat.completions.create(
                model=model,
                messages=messages,
                tools=_GPT_OSS_POST_TOOL_TOOLS,
                tool_choice="auto",
                reasoning_effort="high",
                temperature=0,
                max_tokens=512,
            )
            msg2 = response2.choices[0].message
            assistant_turn2: dict = {"role": "assistant", "content": msg2.content or ""}
            if msg2.tool_calls:
                assistant_turn2["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg2.tool_calls
                ]
            messages.append(assistant_turn2)
            tool_call_msg = msg2

        assert tool_call_msg.tool_calls, "expected an action tool call after confirmation"

        # Minimal tool result keeps the post-tool generation honest about
        # producing its own final paragraph rather than parroting tool output.
        for tc in tool_call_msg.tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps({"ok": True}),
                }
            )

        # Turn 3: final user-facing response - the assertions below run here.
        response3 = api_client.chat.completions.create(
            model=model,
            messages=messages,
            tools=_GPT_OSS_POST_TOOL_TOOLS,
            tool_choice="auto",
            reasoning_effort="high",
            temperature=0,
            max_tokens=512,
        )
        final_content = response3.choices[0].message.content or ""

        assert final_content.strip(), (
            "post-tool final response must not be empty; "
            f"finish_reason={response3.choices[0].finish_reason!r}"
        )

        invisible_match = _INVISIBLE_RUN.search(final_content)
        assert not invisible_match, (
            "post-tool final content must not contain zero-width or "
            "non-breaking-space characters; first hit at offset "
            f"{invisible_match.start()}: {final_content!r}"
        )

        self_analysis = _ENGLISH_SELF_ANALYSIS.search(final_content)
        assert not self_analysis, (
            "post-tool final content must not leak English self-analysis "
            f"(matched {self_analysis.group(0)!r}): {final_content!r}"
        )

        whitespace_run = _WHITESPACE_RUN.search(final_content)
        assert not whitespace_run, (
            "post-tool final content must not contain long whitespace runs; "
            f"first hit at offset {whitespace_run.start()}: {final_content!r}"
        )

        ellipsis_loop = _ELLIPSIS_LOOP.search(final_content)
        assert not ellipsis_loop, (
            "post-tool final content must not contain ellipsis loops "
            f"(matched {ellipsis_loop.group(0)!r}): {final_content!r}"
        )


@pytest.mark.engine("sglang", "vllm", "trtllm", "tokenspeed")
@pytest.mark.gpu(4)
@pytest.mark.model("openai/gpt-oss-120b")
@pytest.mark.gateway(extra_args=["--history-backend", "memory"])
class TestChatCompletionGptOss120B(TestChatCompletionGptOss):
    """Tests for chat completions API with Harmony model (GPT-OSS 120B, 4 GPU)."""
