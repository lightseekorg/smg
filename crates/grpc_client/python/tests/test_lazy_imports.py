from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _import_proto_package(monkeypatch, tmp_path):
    dist_info = tmp_path / "smg_grpc_proto-0.0.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text("Name: smg-grpc-proto\nVersion: 0.0.0\n")

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.syspath_prepend(str(PACKAGE_ROOT))
    sys.modules.pop("smg_grpc_proto", None)
    return importlib.import_module("smg_grpc_proto")


def test_from_import_loads_only_requested_generated_module(monkeypatch, tmp_path):
    smg_grpc_proto = _import_proto_package(monkeypatch, tmp_path)
    requested = types.ModuleType("tokenspeed_scheduler_pb2_grpc")
    calls: list[tuple[str, str | None]] = []

    def fake_import_module(name: str, package: str | None = None):
        calls.append((name, package))
        if name == ".generated.tokenspeed_scheduler_pb2_grpc" and package == "smg_grpc_proto":
            return requested
        raise AssertionError(f"unexpected import: {name!r}, package={package!r}")

    monkeypatch.setattr(smg_grpc_proto, "import_module", fake_import_module)

    namespace: dict[str, object] = {}
    exec("from smg_grpc_proto import tokenspeed_scheduler_pb2_grpc", namespace)

    assert namespace["tokenspeed_scheduler_pb2_grpc"] is requested
    assert calls == [(".generated.tokenspeed_scheduler_pb2_grpc", "smg_grpc_proto")]


def test_unknown_attribute_raises_attribute_error(monkeypatch, tmp_path):
    smg_grpc_proto = _import_proto_package(monkeypatch, tmp_path)

    try:
        smg_grpc_proto.not_a_generated_module
    except AttributeError as exc:
        assert "not_a_generated_module" in str(exc)
    else:
        raise AssertionError("expected AttributeError")
