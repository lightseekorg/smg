"""Helpers for syncing proto sources and generating Python gRPC stubs."""

from __future__ import annotations

import fcntl
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def _proto_files_in(source_proto_dir: Path) -> list[Path]:
    """List proto files from a specific source directory."""
    return sorted(source_proto_dir.glob("*.proto"))


def resolve_proto_sources(
    package_dir: Path,
    source_proto_dir: Path | None = None,
) -> tuple[Path, list[Path]]:
    """Return the proto source directory and its files."""
    if source_proto_dir is not None:
        proto_files = _proto_files_in(source_proto_dir)
        if not proto_files:
            raise FileNotFoundError(f"No .proto files found in {source_proto_dir}")
        return source_proto_dir, proto_files

    proto_dir = package_dir / "smg_grpc_proto" / "proto"
    repo_proto_dir = package_dir.parent / "proto"

    for candidate in (repo_proto_dir, proto_dir):
        proto_files = _proto_files_in(candidate)
        if proto_files:
            return candidate, proto_files

    raise FileNotFoundError(f"No .proto files found in {repo_proto_dir} or {proto_dir}")


def sync_proto_sources(
    package_dir: Path,
    source_proto_dir: Path | None = None,
) -> list[Path]:
    """Populate package-local proto files from the best available source."""
    proto_dir = package_dir / "smg_grpc_proto" / "proto"
    source_proto_dir, source_proto_files = resolve_proto_sources(
        package_dir,
        source_proto_dir=source_proto_dir,
    )

    if proto_dir.exists() and proto_dir.resolve() == source_proto_dir.resolve():
        return [proto_dir / proto_file.name for proto_file in source_proto_files]

    proto_dir.mkdir(parents=True, exist_ok=True)
    if not proto_dir.is_symlink():
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
    source_proto_dir: Path | None = None,
) -> bool:
    """Check whether the generated stubs exist and are newer than the protos."""
    _, source_proto_files = resolve_proto_sources(package_dir, source_proto_dir=source_proto_dir)

    output_dir = package_dir / "smg_grpc_proto" / "generated"
    expected_paths = expected_generated_stub_paths(output_dir, source_proto_files)
    if any(not path.exists() for path in expected_paths):
        return False

    actual_paths = {path for path in output_dir.iterdir() if path.is_file()}
    if actual_paths != set(expected_paths):
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


def _load_grpc_tools() -> tuple[object, object]:
    """Import grpc_tools lazily so tests can patch the loader."""
    try:
        import grpc_tools
        from grpc_tools import protoc
    except ImportError as exc:
        raise RuntimeError(
            "grpcio-tools not installed. Install with: pip install grpcio-tools"
        ) from exc

    return grpc_tools, protoc


def _normalize_generated_imports(content: str, proto_files: list[Path]) -> str:
    """Rewrite generated sibling proto imports to package-relative imports."""
    for proto_file in proto_files:
        module_name = proto_file.stem + "_pb2"
        content = content.replace(f"import {module_name}", f"from . import {module_name}")
    return content


def _rewrite_generated_stubs(output_dir: Path, proto_files: list[Path]) -> None:
    """Normalize imports and type-ignore headers in generated stub files."""
    mypy_header = "# mypy: ignore-errors\n"
    for py_file in output_dir.glob("*_pb2*.py"):
        content = py_file.read_text()
        content = _normalize_generated_imports(content, proto_files)
        if not content.startswith("# mypy:"):
            content = mypy_header + content
        py_file.write_text(content)

    for pyi_file in output_dir.glob("*_pb2*.pyi"):
        content = pyi_file.read_text()
        content = _normalize_generated_imports(content, proto_files)
        if not content.startswith("# mypy:"):
            content = mypy_header + content
        pyi_file.write_text(content)


def _replace_generated_output(temp_output_dir: Path, output_dir: Path) -> None:
    """Swap the freshly generated stub tree into place after successful compilation."""
    backup_dir = output_dir.parent / f"{output_dir.name}.bak"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)

    try:
        if output_dir.exists():
            output_dir.replace(backup_dir)
        temp_output_dir.replace(output_dir)
    except Exception:
        if not output_dir.exists() and backup_dir.exists():
            backup_dir.replace(output_dir)
        raise
    else:
        if backup_dir.exists():
            shutil.rmtree(backup_dir)


@contextmanager
def _proto_compile_lock(package_dir: Path) -> Iterator[None]:
    """Serialize proto generation across parallel test processes."""
    package_root = package_dir / "smg_grpc_proto"
    package_root.mkdir(parents=True, exist_ok=True)

    lock_path = package_root / ".proto-build.lock"
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def compile_grpc_protos(
    package_dir: Path | None = None,
    source_proto_dir: Path | None = None,
) -> None:
    """Generate Python gRPC stubs from the checked-in or packaged proto files."""
    package_dir = package_dir or Path(__file__).resolve().parents[1]
    source_proto_dir, source_proto_files = resolve_proto_sources(
        package_dir, source_proto_dir=source_proto_dir
    )

    proto_files = sync_proto_sources(package_dir, source_proto_dir=source_proto_dir)
    proto_dir = package_dir / "smg_grpc_proto" / "proto"
    output_dir = package_dir / "smg_grpc_proto" / "generated"
    package_root = package_dir / "smg_grpc_proto"
    temp_output_dir = Path(tempfile.mkdtemp(prefix="generated.", dir=package_root))

    try:
        grpc_tools, protoc = _load_grpc_tools()

        (temp_output_dir / "__init__.py").write_text(
            '"""Auto-generated protobuf stubs. Do not edit."""\n'
        )

        well_known_protos = Path(grpc_tools.__file__).parent / "_proto"
        args = [
            "grpc_tools.protoc",
            f"--proto_path={proto_dir}",
            f"--proto_path={well_known_protos}",
            f"--python_out={temp_output_dir}",
            f"--grpc_python_out={temp_output_dir}",
            f"--pyi_out={temp_output_dir}",
            *[str(proto_file) for proto_file in proto_files],
        ]

        print(f"Generating protobuf stubs from {len(proto_files)} proto files...")
        result = protoc.main(args)
        if result != 0:
            raise RuntimeError(f"protoc returned non-zero exit code: {result}")

        _rewrite_generated_stubs(temp_output_dir, proto_files)
        _replace_generated_output(temp_output_dir, output_dir)
    finally:
        if temp_output_dir.exists():
            shutil.rmtree(temp_output_dir)

    generated_count = len(list(output_dir.glob("*.py"))) + len(list(output_dir.glob("*.pyi")))
    print(f"Generated {generated_count} files (including type stubs)")


def ensure_generated_stubs(package_dir: Path | None = None, *, force: bool = False) -> None:
    """Compile stubs when missing or stale, with a lock for parallel test runs."""
    package_dir = package_dir or Path(__file__).resolve().parents[1]
    with _proto_compile_lock(package_dir):
        source_proto_dir, source_proto_files = resolve_proto_sources(package_dir)
        if not force and generated_stubs_are_current(
            package_dir,
            source_proto_dir=source_proto_dir,
        ):
            return

        compile_grpc_protos(package_dir, source_proto_dir=source_proto_dir)
