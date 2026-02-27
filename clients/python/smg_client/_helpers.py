"""Shared helpers for SmgClient API modules."""

from __future__ import annotations

from typing import Any


def prepare_body(kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str] | None]:
    """Extract extra_body and extra_headers from kwargs, merge extra_body into body."""
    extra_body = kwargs.pop("extra_body", None)
    extra_headers = kwargs.pop("extra_headers", None)
    if extra_body:
        kwargs.update(extra_body)
    return kwargs, extra_headers
