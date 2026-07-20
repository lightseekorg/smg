"""Engine-free compatibility test for the TokenSpeed encoder servicer."""

import ast
import sys
import types
from pathlib import Path

_ENCODER_SERVICER = (
    Path(__file__).parents[1] / "smg_grpc_servicer" / "tokenspeed" / "encoder_servicer.py"
)


def _load_lazy_encode_request():
    tree = ast.parse(_ENCODER_SERVICER.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_lazy_encode_request"
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, _ENCODER_SERVICER, "exec"), namespace)
    return namespace["_lazy_encode_request"]


def test_lazy_encode_request_uses_promoted_epd_package(monkeypatch):
    sentinel = object()
    encode_worker = types.ModuleType("tokenspeed.runtime.epd.encode_worker")
    encode_worker.EncodeRequest = sentinel
    monkeypatch.setitem(sys.modules, "tokenspeed.runtime.epd.encode_worker", encode_worker)

    assert _load_lazy_encode_request()() is sentinel
