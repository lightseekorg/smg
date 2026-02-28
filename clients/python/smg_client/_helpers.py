"""Shared helpers for SmgClient API modules."""

from __future__ import annotations

from typing import Any


def _serialize_value(value: Any) -> Any:
    """Recursively convert Pydantic BaseModel instances to dicts.

    This enables passing response objects directly in request bodies
    (e.g. ``response1.content`` as message content in a round-trip)
    without requiring manual ``.model_dump()`` calls — matching
    OpenAI and Anthropic SDK behavior.
    """
    try:
        from pydantic import BaseModel
    except ImportError:
        return value

    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    return value


def prepare_body(kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str] | None]:
    """Extract extra_body and extra_headers from kwargs, merge extra_body into body.

    Pydantic BaseModel instances anywhere in the body are automatically
    converted to dicts so that SDK response objects can be passed directly.
    """
    extra_body = kwargs.pop("extra_body", None)
    extra_headers = kwargs.pop("extra_headers", None)
    if extra_body:
        kwargs.update(extra_body)
    return _serialize_value(kwargs), extra_headers
