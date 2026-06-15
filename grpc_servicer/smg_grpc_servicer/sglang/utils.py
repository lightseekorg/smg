"""gRPC utility functions."""

import os
from array import array
from collections.abc import Iterable
from http import HTTPStatus

import grpc

# Toggle between the legacy list contract and the new array("q") contract for
# token IDs handed to the SGLang scheduler. Defaults to the legacy list contract
# for compatibility with older SGLang builds whose scheduler expects
# ``list[int]``. Set ``SGLANG_GRPC_TOKEN_ID_ARRAY=1`` (or true/yes) to emit
# ``array("q")``, which current SGLang requires (see ``to_token_id_array``).
_TOKEN_ID_ARRAY_ENV = "SGLANG_GRPC_TOKEN_ID_ARRAY"
_TRUTHY = ("1", "true", "yes")


def _use_token_id_array() -> bool:
    """Whether to emit ``array("q")`` (new contract) vs ``list`` (old contract)."""
    return os.getenv(_TOKEN_ID_ARRAY_ENV, "false").strip().lower() in _TRUTHY


def to_token_id_array(token_ids: Iterable[int] | None) -> array | list | None:
    """Coerce a token-id sequence to the container the SGLang scheduler expects.

    By default this returns a plain ``list`` (the legacy contract). Setting
    ``SGLANG_GRPC_TOKEN_ID_ARRAY=1`` (or true/yes) switches to ``array("q")``
    (signed 64-bit ints), which current SGLang requires.

    Current SGLang declares ``TokenizedGenerateReqInput.input_ids`` /
    ``TokenizedEmbeddingReqInput.input_ids`` as ``Optional[array[int]]`` and its
    ``Req`` concatenates ``origin_input_ids + output_ids`` where ``output_ids`` is
    ``array("q")``. Passing a plain ``list`` (as gRPC repeated fields decode to)
    makes that concatenation raise ``TypeError: can only concatenate list (not
    "array.array") to list`` on every request. Enabling the array contract mirrors
    what SGLang's own HTTP ``TokenizerManager`` does before handing IDs to the
    scheduler.

    ``array("q", x)`` accepts any iterable of ints (list, protobuf
    ``RepeatedScalarContainer``, or an existing ``array``), so this is safe to
    apply at every call site. Returns ``None`` for ``None`` input.
    """
    if token_ids is None:
        return None
    if _use_token_id_array():
        return array("q", token_ids)
    return list(token_ids)


_HTTP_TO_GRPC_CODE = {
    HTTPStatus.BAD_REQUEST: grpc.StatusCode.INVALID_ARGUMENT,
    HTTPStatus.SERVICE_UNAVAILABLE: grpc.StatusCode.UNAVAILABLE,
    HTTPStatus.INTERNAL_SERVER_ERROR: grpc.StatusCode.INTERNAL,
}


def abort_code_from_output(output: dict) -> grpc.StatusCode:
    """Map a scheduler error output to the appropriate gRPC status code."""
    finish_reason = output.get("meta_info", {}).get("finish_reason")
    if isinstance(finish_reason, dict):
        status_code = finish_reason.get("status_code")
        if status_code is not None:
            return _HTTP_TO_GRPC_CODE.get(status_code, grpc.StatusCode.INTERNAL)
    return grpc.StatusCode.INTERNAL
