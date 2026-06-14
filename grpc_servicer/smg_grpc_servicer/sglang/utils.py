"""gRPC utility functions."""

from array import array
from collections.abc import Iterable
from http import HTTPStatus

import grpc


def to_token_id_array(token_ids: Iterable[int] | None) -> array | None:
    """Coerce a token-id sequence to ``array("q")`` (signed 64-bit ints).

    SGLang declares ``TokenizedGenerateReqInput.input_ids`` /
    ``TokenizedEmbeddingReqInput.input_ids`` as ``Optional[array[int]]`` and its
    ``Req`` concatenates ``origin_input_ids + output_ids`` where ``output_ids`` is
    ``array("q")``. Passing a plain ``list`` (as gRPC repeated fields decode to)
    makes that concatenation raise ``TypeError: can only concatenate list (not
    "array.array") to list`` on every request. This mirrors what SGLang's own
    HTTP ``TokenizerManager`` does before handing IDs to the scheduler.

    ``array("q", x)`` accepts any iterable of ints (list, protobuf
    ``RepeatedScalarContainer``, or an existing ``array``), so this is safe to
    apply at every call site. Returns ``None`` for ``None`` input.
    """
    if token_ids is None:
        return None
    return array("q", token_ids)


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
