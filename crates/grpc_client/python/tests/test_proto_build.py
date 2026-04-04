from __future__ import annotations

import os
from pathlib import Path
import sys


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from smg_grpc_proto import _proto_build


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


def test_generated_stubs_are_current_requires_fresh_outputs(tmp_path: Path) -> None:
    package_dir = tmp_path / "python"
    source_proto = _write_proto(package_dir.parent / "proto")
    output_dir = package_dir / "smg_grpc_proto" / "generated"
    expected_paths = _proto_build.expected_generated_stub_paths(output_dir, [source_proto])

    _write_generated_stubs(output_dir, [source_proto])
    generated_time = source_proto.stat().st_mtime_ns + 5_000_000
    for path in expected_paths:
        os.utime(path, ns=(generated_time, generated_time))

    assert _proto_build.generated_stubs_are_current(package_dir, [source_proto])

    newer_source_time = generated_time + 5_000_000
    os.utime(source_proto, ns=(newer_source_time, newer_source_time))
    assert not _proto_build.generated_stubs_are_current(package_dir, [source_proto])


def test_ensure_generated_stubs_rebuilds_when_sources_change(tmp_path: Path, monkeypatch) -> None:
    package_dir = tmp_path / "python"
    source_proto = _write_proto(package_dir.parent / "proto")
    output_dir = package_dir / "smg_grpc_proto" / "generated"
    compile_calls: list[int] = []

    def fake_compile(
        package_dir_arg: Path | None = None,
        source_proto_dir: Path | None = None,
        source_proto_files: list[Path] | None = None,
    ) -> None:
        compile_calls.append(1)
        assert package_dir_arg is not None
        assert source_proto_files is not None
        _proto_build.sync_proto_sources(
            package_dir_arg,
            source_proto_dir=source_proto_dir,
            source_proto_files=source_proto_files,
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
