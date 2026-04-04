from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_PROTO_SRC = _ROOT / "crates" / "grpc_client" / "python"
_SERVICER_SRC = _ROOT / "grpc_servicer"

if str(_PROTO_SRC) in sys.path:
    sys.path.remove(str(_PROTO_SRC))
sys.path.insert(0, str(_PROTO_SRC))

if str(_SERVICER_SRC) in sys.path:
    sys.path.remove(str(_SERVICER_SRC))
sys.path.insert(0, str(_SERVICER_SRC))

_PROTO_SYMBOL_SKIP_REASON: str | None = None


def _ensure_local_proto_stubs() -> None:
    """Generate local proto stubs for source-tree test runs when possible."""
    global _PROTO_SYMBOL_SKIP_REASON

    helper_path = _PROTO_SRC / "smg_grpc_proto" / "_proto_build.py"
    if not helper_path.exists():
        _PROTO_SYMBOL_SKIP_REASON = "local smg-grpc-proto build helper not found"
        return

    spec = importlib.util.spec_from_file_location("_smg_grpc_proto_build", helper_path)
    if spec is None or spec.loader is None:
        _PROTO_SYMBOL_SKIP_REASON = "unable to load local smg-grpc-proto build helper"
        return

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        module.ensure_generated_stubs(_PROTO_SRC)
        importlib.import_module("smg_grpc_proto.generated.common_pb2")
    except Exception as exc:
        _PROTO_SYMBOL_SKIP_REASON = f"smg-grpc-proto stubs unavailable: {exc}"


_ensure_local_proto_stubs()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: mark test as requiring a CUDA-capable GPU")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _PROTO_SYMBOL_SKIP_REASON is None:
        return

    if any("test_proto_symbols.py" in item.nodeid for item in items):
        pytest.fail(_PROTO_SYMBOL_SKIP_REASON, pytrace=False)
