# Adapted from https://github.com/openai/simple-evals/ and
# https://github.com/MMMU-Benchmark/MMMU
"""
MMMU Evaluation - A Massive Multi-discipline Multimodal Understanding Benchmark
Yue et al. https://arxiv.org/abs/2311.16502

Evaluates vision-language models on college-level multimodal questions
spanning Art, Business, Science, Health, Humanities, and Engineering.
"""

from __future__ import annotations

import base64
import io
import re
from typing import TYPE_CHECKING

from datasets import load_dataset

from . import simple_eval_common as common
from .simple_eval_common import (
    HTML_JINJA,
    Eval,
    EvalResult,
    SingleEvalResult,
)

if TYPE_CHECKING:
    from .simple_eval_common import SamplerBase

# MMMU subjects grouped by category
MMMU_ART_AND_DESIGN = ["Art", "Art_Theory", "Design", "Music"]

QUERY_TEMPLATE_MMMU = """
Answer the following multiple choice question. The last line of your response should be of the following
format: 'Answer: $LETTER' (without quotes) where LETTER is one of {option_letters}. Think step by step before answering.

{question}

{options}
""".strip()


def _image_to_base64(image) -> str:
    """Convert a PIL image to a base64-encoded PNG string."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _parse_options(options_str: str) -> list[str]:
    """Parse the options string from MMMU dataset (Python list literal)."""
    import ast

    return ast.literal_eval(options_str)


def _build_content_parts(
    sampler: SamplerBase,
    question: str,
    row: dict,
) -> list[dict]:
    """Build multimodal content parts from a MMMU question.

    Replaces <image N> placeholders with actual image content parts.
    """
    parts = []
    # Split question text on image placeholders
    segments = re.split(r"(<image \d+>)", question)
    for segment in segments:
        m = re.match(r"<image (\d+)>", segment)
        if m:
            idx = int(m.group(1))
            image = row.get(f"image_{idx}")
            if image is not None:
                b64 = _image_to_base64(image)
                parts.append(sampler._handle_image(b64, encoding="base64", format="png"))  # type: ignore[attr-defined]
            else:
                parts.append(sampler._handle_text(segment))  # type: ignore[attr-defined]
        elif segment.strip():
            parts.append(sampler._handle_text(segment))  # type: ignore[attr-defined]
    return parts


class MMMUEval(Eval):
    """MMMU benchmark evaluation for a specific category."""

    def __init__(
        self,
        subjects: list[str],
        num_threads: int,
    ):
        self.examples = []
        for subject in subjects:
            ds = load_dataset("MMMU/MMMU", subject, split="validation")
            for row in ds:
                row["_subject"] = subject
                self.examples.append(row)
        self.num_threads = num_threads

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict) -> SingleEvalResult:
            options = _parse_options(row["options"])
            letters = [chr(ord("A") + i) for i in range(len(options))]
            option_letters = "".join(letters)
            options_text = "\n".join(f"{letter}) {opt}" for letter, opt in zip(letters, options))

            prompt_text = QUERY_TEMPLATE_MMMU.format(
                question=row["question"],
                options=options_text,
                option_letters=option_letters,
            )

            # Build multimodal content parts (text + images)
            content_parts = _build_content_parts(sampler, prompt_text, row)

            prompt_messages = [sampler._pack_message(role="user", content=content_parts)]  # type: ignore[attr-defined]
            response_text = sampler(prompt_messages)
            response_text = response_text or ""

            # Use dynamic pattern to handle variable number of options (A-D, A-E, etc.)
            answer_pattern = rf"(?i)Answer\s*:\s*([{''.join(letters)}])"
            match = re.search(answer_pattern, response_text)
            extracted_answer = match.group(1).upper() if match else None
            score = 1.0 if extracted_answer == row["answer"] else 0.0

            subject = row["_subject"]
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=[{"role": "user", "content": prompt_text}],
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["answer"],
                extracted_answer=extracted_answer,
            )
            convo = [{"role": "user", "content": prompt_text}] + [
                dict(content=response_text, role="assistant")
            ]
            return SingleEvalResult(
                html=html,
                score=score,
                metrics={subject: score},
                convo=convo,
            )

        results = common.map_with_progress(fn, self.examples, self.num_threads)
        return common.aggregate_results(results)
