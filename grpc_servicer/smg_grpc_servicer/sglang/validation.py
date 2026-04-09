"""Validation helpers for SGLang gRPC request payloads."""

from typing import Any


def validate_tokenized_input(grpc_req: Any, request_kind: str) -> tuple[str, list[int]]:
    """Validate tokenized field and return (original_text, input_ids)."""
    if not grpc_req.HasField("tokenized"):
        raise ValueError("Tokenized input must be provided")

    input_text = grpc_req.tokenized.original_text
    input_ids = list(grpc_req.tokenized.input_ids)
    if not input_ids:
        raise ValueError(f"input_ids cannot be empty for {request_kind} requests")

    return input_text, input_ids
