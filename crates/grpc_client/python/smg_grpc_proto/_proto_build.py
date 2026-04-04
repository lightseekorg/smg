"""Helpers for syncing proto sources and generating Python gRPC stubs."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import shutil
from typing import Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None


def resolve_proto_sources(package_dir: Path) -> tuple[Path, list[Path]]:
    """Return the best available proto source directory and its files."""
    proto_dir = package_dir / "smg_grpc_proto" / "proto"
    source_proto_dir = package_dir.parent / "proto"

    for candidate in (source_proto_dir, proto_dir):
        proto_files = sorted(candidate.glob("*.proto"))
        if proto_files:
            return candidate, proto_files

    raise FileNotFoundError(
        f"No .proto files found in {source_proto_dir} or {proto_dir}"
    )


def sync_proto_sources(
    package_dir: Path,
    source_proto_dir: Path | None = None,
    source_proto_files: list[Path] | None = None,
) -> list[Path]:
    """Populate package-local proto files from the best available source."""
    proto_dir = package_dir / "smg_grpc_proto" / "proto"
    if source_proto_dir is None or source_proto_files is None:
        source_proto_dir, source_proto_files = resolve_proto_sources(package_dir)

    if proto_dir.exists() and proto_dir.resolve() == source_proto_dir.resolve():
        return list(source_proto_files)

    proto_dir.mkdir(parents=True, exist_ok=True)
    for existing in proto_dir.glob("*.proto"):
        existing.unlink()

    synced_proto_files = []
    for proto_file in source_proto_files:
        target = proto_dir / proto_file.name
        shutil.copy2(proto_file, target)
        synced_proto_files.append(target)

    return synced_proto_files


def expected_generated_stub_paths(output_dir: Path, proto_files: list[Path]) -> list[Path]:
    """Return the generated files expected for the provided proto set."""
    expected_paths = [output_dir / "__init__.py"]
    for proto_file in proto_files:
        stem = proto_file.stem
        expected_paths.extend(
            [
                output_dir / f"{stem}_pb2.py",
                output_dir / f"{stem}_pb2.pyi",
                output_dir / f"{stem}_pb2_grpc.py",
            ]
        )
    return expected_paths


def generated_stubs_are_current(
    package_dir: Path,
    source_proto_files: list[Path] | None = None,
) -> bool:
    """Check whether the generated stubs exist and are newer than the protos."""
    if source_proto_files is None:
        _, source_proto_files = resolve_proto_sources(package_dir)

    output_dir = package_dir / "smg_grpc_proto" / "generated"
    expected_paths = expected_generated_stub_paths(output_dir, source_proto_files)
    if any(not path.exists() for path in expected_paths):
        return False

    newest_source_mtime = max(proto_file.stat().st_mtime_ns for proto_file in source_proto_files)
    oldest_generated_mtime = min(path.stat().st_mtime_ns for path in expected_paths)
    return oldest_generated_mtime >= newest_source_mtime


def _clear_generated_stubs(output_dir: Path) -> None:
    """Remove generated stub artifacts while preserving unrelated files."""
    if not output_dir.exists():
        return

    for pattern in ("*_pb2*.py", "*_pb2*.pyi", "*.pyc"):
        for path in output_dir.glob(pattern):
            path.unlink()

    for cache_dir in output_dir.rglob("__pycache__"):
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir)


@contextmanager
def _proto_compile_lock(package_dir: Path) -> Iterator[None]:
    """Serialize proto generation across parallel test processes."""
    generated_dir = package_dir / "smg_grpc_proto" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    lock_path = generated_dir / ".proto-build.lock"
    with lock_path.open("a+") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def compile_grpc_protos(
    package_dir: Path | None = None,
    source_proto_dir: Path | None = None,
    source_proto_files: list[Path] | None = None,
) -> None:
    """Generate Python gRPC stubs from the checked-in or packaged proto files."""
    package_dir = package_dir or Path(__file__).resolve().parents[1]
    if source_proto_dir is None or source_proto_files is None:
        source_proto_dir, source_proto_files = resolve_proto_sources(package_dir)

    proto_files = sync_proto_sources(
        package_dir,
        source_proto_dir=source_proto_dir,
        source_proto_files=source_proto_files,
    )
    proto_dir = package_dir / "smg_grpc_proto" / "proto"
    output_dir = package_dir / "smg_grpc_proto" / "generated"

    _clear_generated_stubs(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "__init__.py").write_text('"""Auto-generated protobuf stubs. Do not edit."""\n')

    try:
        import grpc_tools
        from grpc_tools import protoc
    except ImportError as exc:
        raise RuntimeError(
            "grpcio-tools not installed. Install with: pip install grpcio-tools"
        ) from exc

    well_known_protos = Path(grpc_tools.__file__).parent / "_proto"
    args = [
        "grpc_tools.protoc",
        f"--proto_path={proto_dir}",
        f"--proto_path={well_known_protos}",
        f"--python_out={output_dir}",
        f"--grpc_python_out={output_dir}",
        f"--pyi_out={output_dir}",
        *[str(proto_file) for proto_file in proto_files],
    ]

    print(f"Generating protobuf stubs from {len(proto_files)} proto files...")
    result = protoc.main(args)
    if result != 0:
        raise RuntimeError(f"protoc returned non-zero exit code: {result}")

    mypy_header = "# mypy: ignore-errors\n"
    for py_file in output_dir.glob("*_pb2*.py"):
        content = py_file.read_text()
        for proto_file in proto_files:
            module_name = proto_file.stem + "_pb2"
            content = content.replace(f"import {module_name}", f"from . import {module_name}")
        if not content.startswith("# mypy:"):
            content = mypy_header + content
        py_file.write_text(content)

    for pyi_file in output_dir.glob("*_pb2*.pyi"):
        content = pyi_file.read_text()
        if not content.startswith("# mypy:"):
            pyi_file.write_text(mypy_header + content)

    generated_count = len(list(output_dir.glob("*.py"))) + len(list(output_dir.glob("*.pyi")))
    print(f"Generated {generated_count} files (including type stubs)")


def ensure_generated_stubs(package_dir: Path | None = None, *, force: bool = False) -> None:
    """Compile stubs when missing or stale, with a lock for parallel test runs."""
    package_dir = package_dir or Path(__file__).resolve().parents[1]
    with _proto_compile_lock(package_dir):
        source_proto_dir, source_proto_files = resolve_proto_sources(package_dir)
        if not force and generated_stubs_are_current(
            package_dir,
            source_proto_files=source_proto_files,
        ):
            return

        compile_grpc_protos(
            package_dir,
            source_proto_dir=source_proto_dir,
            source_proto_files=source_proto_files,
        )
