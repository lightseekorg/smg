"""
Custom setup.py to generate protobuf stubs at build time.
The generated files are NOT committed to git — they're created fresh during pip install.
"""

from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop


def compile_grpc_protos():
    """Generate Python gRPC stubs from .proto files."""
    package_dir = Path(__file__).parent
    proto_dir = package_dir / "smg_grpc_proto" / "proto"
    output_dir = package_dir / "smg_grpc_proto" / "generated"

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "__init__.py").write_text('"""Auto-generated protobuf stubs. Do not edit."""\n')

    proto_files = list(proto_dir.glob("*.proto"))
    if not proto_files:
        raise FileNotFoundError(f"No .proto files found in {proto_dir}")

    # Use grpc_tools.protoc Python API (same approach as vLLM)
    try:
        # Include well-known types (google/protobuf/*.proto) shipped with grpc_tools
        import grpc_tools
        from grpc_tools import protoc

        well_known_protos = Path(grpc_tools.__file__).parent / "_proto"

        args = [
            "grpc_tools.protoc",
            f"--proto_path={proto_dir}",
            f"--proto_path={well_known_protos}",
            f"--python_out={output_dir}",
            f"--grpc_python_out={output_dir}",
            f"--pyi_out={output_dir}",  # Generate type stubs
        ] + [str(f) for f in proto_files]

        print(f"Generating protobuf stubs from {len(proto_files)} proto files...")
        result = protoc.main(args)

        if result != 0:
            raise RuntimeError(f"protoc returned non-zero exit code: {result}")

    except ImportError:
        raise RuntimeError("grpcio-tools not installed. Install with: pip install grpcio-tools")

    # Fix imports in generated files (grpcio-tools generates absolute imports)
    # Also add mypy ignore-errors comment for generated code
    mypy_header = "# mypy: ignore-errors\n"

    for py_file in output_dir.glob("*_pb2*.py"):
        content = py_file.read_text()

        # Fix imports to be relative
        for proto_file in proto_files:
            module_name = proto_file.stem + "_pb2"
            content = content.replace(f"import {module_name}", f"from . import {module_name}")

        # Add mypy ignore-errors if not already present
        if not content.startswith("# mypy:"):
            content = mypy_header + content

        py_file.write_text(content)

    # Also add mypy header to .pyi files
    for pyi_file in output_dir.glob("*_pb2*.pyi"):
        content = pyi_file.read_text()
        if not content.startswith("# mypy:"):
            pyi_file.write_text(mypy_header + content)

    generated_count = len(list(output_dir.glob("*.py"))) + len(list(output_dir.glob("*.pyi")))
    print(f"Generated {generated_count} files (including type stubs)")


class BuildPyWithProto(build_py):
    """Custom build_py that generates protobuf stubs before building."""

    def run(self):
        compile_grpc_protos()
        super().run()


class DevelopWithProto(develop):
    """Custom develop that generates protobuf stubs for editable installs."""

    def run(self):
        compile_grpc_protos()
        super().run()


setup(
    cmdclass={
        "build_py": BuildPyWithProto,
        "develop": DevelopWithProto,
    }
)
