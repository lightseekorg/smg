from __future__ import annotations

import os
import sys
import types
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
while str(_PACKAGE_ROOT) in sys.path:
    sys.path.remove(str(_PACKAGE_ROOT))
sys.path.insert(0, str(_PACKAGE_ROOT))


def _load_proto_build():
    from smg_grpc_proto import _proto_build

    return _proto_build


_proto_build = _load_proto_build()


def _write_proto(proto_dir: Path, name: str = "common.proto") -> Path:
    proto_dir.mkdir(parents=True, exist_ok=True)
    proto_path = proto_dir / name
    proto_path.write_text('syntax = "proto3";\npackage smg.grpc.common;\nmessage Example {}\n')
    return proto_path


def _write_generated_stubs(output_dir: Path, proto_files: list[Path]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in _proto_build.expected_generated_stub_paths(output_dir, proto_files):
        path.write_text("# generated\n")


def test_resolve_proto_sources_falls_back_to_packaged_proto_dir(tmp_path: Path) -> None:
    package_dir = tmp_path / "python"
    packaged_proto = _write_proto(package_dir / "smg_grpc_proto" / "proto")

    source_dir, proto_files = _proto_build.resolve_proto_sources(package_dir)

    assert source_dir == package_dir / "smg_grpc_proto" / "proto"
    assert proto_files == [packaged_proto]


def test_sync_proto_sources_returns_package_paths_for_symlinked_proto_dir(tmp_path: Path) -> None:
    package_dir = tmp_path / "python"
    source_proto = _write_proto(package_dir.parent / "proto")
    proto_dir = package_dir / "smg_grpc_proto" / "proto"
    proto_dir.parent.mkdir(parents=True, exist_ok=True)
    proto_dir.symlink_to(Path("../../proto"))

    synced = _proto_build.sync_proto_sources(package_dir)

    assert synced == [proto_dir / source_proto.name]
    assert synced[0] != source_proto


def test_sync_proto_sources_does_not_unlink_symlink_target_on_override(tmp_path: Path) -> None:
    package_dir = tmp_path / "python"
    repo_proto = _write_proto(package_dir.parent / "proto")
    override_dir = tmp_path / "override"
    override_proto = _write_proto(override_dir, name="override.proto")
    proto_dir = package_dir / "smg_grpc_proto" / "proto"
    proto_dir.parent.mkdir(parents=True, exist_ok=True)
    proto_dir.symlink_to(Path("../../proto"))

    synced = _proto_build.sync_proto_sources(package_dir, source_proto_dir=override_dir)

    assert repo_proto.exists()
    assert repo_proto.read_text() == 'syntax = "proto3";\npackage smg.grpc.common;\nmessage Example {}\n'
    assert synced == [proto_dir / override_proto.name]
    assert (package_dir.parent / "proto" / override_proto.name).exists()


def test_sync_proto_sources_honors_source_proto_dir_override(tmp_path: Path) -> None:
    package_dir = tmp_path / "python"
    override_dir = tmp_path / "override"
    override_proto = _write_proto(override_dir, name="override.proto")

    synced = _proto_build.sync_proto_sources(package_dir, source_proto_dir=override_dir)

    assert synced == [package_dir / "smg_grpc_proto" / "proto" / override_proto.name]


def test_generated_stubs_are_current_requires_fresh_outputs(tmp_path: Path) -> None:
    package_dir = tmp_path / "python"
    source_proto = _write_proto(package_dir.parent / "proto")
    output_dir = package_dir / "smg_grpc_proto" / "generated"
    expected_paths = _proto_build.expected_generated_stub_paths(output_dir, [source_proto])

    _write_generated_stubs(output_dir, [source_proto])
    generated_time = source_proto.stat().st_mtime_ns + 5_000_000
    for path in expected_paths:
        os.utime(path, ns=(generated_time, generated_time))

    assert _proto_build.generated_stubs_are_current(package_dir, source_proto.parent)

    newer_source_time = generated_time + 5_000_000
    os.utime(source_proto, ns=(newer_source_time, newer_source_time))
    assert not _proto_build.generated_stubs_are_current(package_dir, source_proto.parent)


def test_generated_stubs_are_current_rejects_extra_generated_files(tmp_path: Path) -> None:
    package_dir = tmp_path / "python"
    source_proto = _write_proto(package_dir.parent / "proto")
    output_dir = package_dir / "smg_grpc_proto" / "generated"
    expected_paths = _proto_build.expected_generated_stub_paths(output_dir, [source_proto])

    _write_generated_stubs(output_dir, [source_proto])
    generated_time = source_proto.stat().st_mtime_ns + 5_000_000
    for path in expected_paths:
        os.utime(path, ns=(generated_time, generated_time))

    extra_path = output_dir / "stale_pb2.py"
    extra_path.write_text("# stale generated stub\n")
    os.utime(extra_path, ns=(generated_time, generated_time))

    assert not _proto_build.generated_stubs_are_current(package_dir, source_proto.parent)


def test_rewrite_generated_stubs_normalizes_pyi_imports(tmp_path: Path) -> None:
    output_dir = tmp_path / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    common_proto = _write_proto(tmp_path / "proto", name="common.proto")
    engine_proto = _write_proto(tmp_path / "proto", name="vllm_engine.proto")
    pyi_file = output_dir / "vllm_engine_pb2.pyi"
    pyi_file.write_text("import common_pb2 as _common_pb2\n")

    _proto_build._rewrite_generated_stubs(output_dir, [common_proto, engine_proto])

    assert pyi_file.read_text() == (
        "# mypy: ignore-errors\nfrom . import common_pb2 as _common_pb2\n"
    )


def test_ensure_generated_stubs_rebuilds_when_sources_change(tmp_path: Path, monkeypatch) -> None:
    package_dir = tmp_path / "python"
    source_proto = _write_proto(package_dir.parent / "proto")
    output_dir = package_dir / "smg_grpc_proto" / "generated"
    compile_calls: list[int] = []

    def fake_compile(
        package_dir_arg: Path | None = None,
        source_proto_dir: Path | None = None,
    ) -> None:
        compile_calls.append(1)
        assert package_dir_arg is not None
        assert source_proto_dir is not None
        source_proto_files = _proto_build.sync_proto_sources(
            package_dir_arg, source_proto_dir=source_proto_dir
        )
        _write_generated_stubs(output_dir, source_proto_files)
        generated_time = max(path.stat().st_mtime_ns for path in source_proto_files) + 5_000_000

        for path in _proto_build.expected_generated_stub_paths(output_dir, source_proto_files):
            os.utime(path, ns=(generated_time, generated_time))

    monkeypatch.setattr(_proto_build, "compile_grpc_protos", fake_compile)

    _proto_build.ensure_generated_stubs(package_dir)
    _proto_build.ensure_generated_stubs(package_dir)
    assert len(compile_calls) == 1

    source_proto.write_text(
        'syntax = "proto3";\npackage smg.grpc.common;\nmessage Example { string name = 1; }\n'
    )
    newer_source_time = max(path.stat().st_mtime_ns for path in output_dir.iterdir()) + 5_000_000
    os.utime(source_proto, ns=(newer_source_time, newer_source_time))
    _proto_build.ensure_generated_stubs(package_dir)
    assert len(compile_calls) == 2


def test_compile_grpc_protos_preserves_existing_stubs_on_protoc_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    package_dir = tmp_path / "python"
    source_proto = _write_proto(package_dir.parent / "proto")
    output_dir = package_dir / "smg_grpc_proto" / "generated"
    existing_stub = output_dir / "common_pb2.py"
    existing_stub.parent.mkdir(parents=True, exist_ok=True)
    existing_stub.write_text("# existing stub\n")
    (output_dir / "__init__.py").write_text('"""existing init"""\n')

    fake_protoc = types.SimpleNamespace(main=lambda args: 1)
    fake_grpc_tools = types.SimpleNamespace(__file__=str(tmp_path / "grpc_tools" / "__init__.py"))
    monkeypatch.setattr(_proto_build, "_load_grpc_tools", lambda: (fake_grpc_tools, fake_protoc))

    try:
        _proto_build.compile_grpc_protos(package_dir, source_proto_dir=source_proto.parent)
    except RuntimeError as exc:
        assert "protoc returned non-zero exit code" in str(exc)
    else:
        raise AssertionError("compile_grpc_protos should fail when protoc.main returns non-zero")

    assert existing_stub.read_text() == "# existing stub\n"
    assert (output_dir / "__init__.py").read_text() == '"""existing init"""\n'
