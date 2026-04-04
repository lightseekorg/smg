"""Helpers for syncing proto sources and generating Python gRPC stubs."""

from __future__ import annotations

from pathlib import Path
import shutil


def sync_proto_sources(package_dir: Path) -> list[Path]:
    """Populate package-local proto files from the repo source-of-truth."""
    proto_dir = package_dir / "smg_grpc_proto" / "proto"
    source_proto_dir = package_dir.parent / "proto"

    source_proto_files = sorted(source_proto_dir.glob("*.proto"))
    if not source_proto_files:
        raise FileNotFoundError(f"No source .proto files found in {source_proto_dir}")

    if proto_dir.exists() and proto_dir.resolve() == source_proto_dir.resolve():
        return [proto_dir / proto_file.name for proto_file in source_proto_files]

    proto_dir.mkdir(parents=True, exist_ok=True)
    for existing in proto_dir.glob("*.proto"):
        existing.unlink()

    synced_proto_files = []
    for proto_file in source_proto_files:
        target = proto_dir / proto_file.name
        shutil.copy2(proto_file, target)
        synced_proto_files.append(target)

    return synced_proto_files


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


def compile_grpc_protos(package_dir: Path | None = None) -> None:
    """Generate Python gRPC stubs from the checked-in .proto files."""
    package_dir = package_dir or Path(__file__).resolve().parents[1]
    proto_files = sync_proto_sources(package_dir)
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
