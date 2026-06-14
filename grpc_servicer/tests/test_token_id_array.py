"""Unit tests for ``to_token_id_array`` — the token-id type fix.

Regression coverage for the SMG+SGLang gRPC crash:

    schedule_batch.py  self.fill_ids = self.origin_input_ids + self.output_ids
    TypeError: can only concatenate list (not "array.array") to list

SGLang declares ``input_ids: Optional[array[int]]`` and concatenates
``origin_input_ids + output_ids`` where ``output_ids`` is ``array("q")``. The
gRPC servicer used to pass a plain ``list`` (gRPC repeated fields decode to a
list), which made that concatenation raise on every request. ``to_token_id_array``
coerces to ``array("q")`` to match SGLang's contract — exactly what SGLang's own
HTTP ``TokenizerManager`` does before reaching the scheduler.

The helper lives in ``smg_grpc_servicer.sglang.utils`` (stdlib + grpc only), so
these tests need no sglang stack.
"""

from __future__ import annotations

import importlib.util
from array import array
from pathlib import Path

import pytest

# Load utils.py directly by path. Importing it via the package
# (``smg_grpc_servicer.sglang.utils``) would run ``sglang/__init__.py``, which
# eagerly imports the servicer and its heavy deps (msgspec, torch, sglang). The
# helper itself is stdlib + grpc only, so a direct file load keeps this test
# runnable without the full inference stack.
_UTILS_PATH = Path(__file__).resolve().parent.parent / "smg_grpc_servicer" / "sglang" / "utils.py"
_spec = importlib.util.spec_from_file_location("_smg_sglang_utils_under_test", _UTILS_PATH)
_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_utils)
to_token_id_array = _utils.to_token_id_array


def test_list_becomes_array_q():
    out = to_token_id_array([1, 2, 3])
    assert isinstance(out, array)
    assert out.typecode == "q"  # signed 64-bit, matches SGLang output_ids
    assert list(out) == [1, 2, 3]


def test_existing_array_passthrough_is_array_q():
    out = to_token_id_array(array("q", [5, 6]))
    assert isinstance(out, array)
    assert out.typecode == "q"
    assert list(out) == [5, 6]


def test_none_returns_none():
    assert to_token_id_array(None) is None


def test_empty_is_empty_array_q():
    out = to_token_id_array([])
    assert isinstance(out, array)
    assert out.typecode == "q"
    assert len(out) == 0


def test_accepts_arbitrary_int_iterable():
    # gRPC repeated fields behave like any int iterable; a generator must work too.
    out = to_token_id_array(range(4))
    assert list(out) == [0, 1, 2, 3]


def test_documents_the_bug_being_fixed():
    """The concatenation SGLang performs: array+array works, list+array raises."""
    output_ids = array("q")  # how SGLang initializes Req.output_ids

    # Fixed path: helper output concatenates cleanly with SGLang's output_ids.
    origin_fixed = to_token_id_array([1, 2, 3])
    assert list(origin_fixed + output_ids) == [1, 2, 3]

    # Pre-fix path: a raw list + array("q") is exactly the crash we eliminated.
    with pytest.raises(TypeError):
        _ = [1, 2, 3] + output_ids
