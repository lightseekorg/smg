"""
Custom setup.py to generate protobuf stubs at build time.
The generated files are NOT committed to git — they're created fresh during pip install.
"""
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPyWithProto(build_py):
    """Custom build_py that generates protobuf stubs before building."""

    def run(self):
        self.generate_proto_stubs()
        super().run()

    def generate_proto_stubs(self):
        """Generate Python gRPC stubs from .proto files."""
        package_dir = Path(__file__).parent
        proto_dir = package_dir / "smg_grpc_proto" / "proto"
        output_dir = package_dir / "smg_grpc_proto" / "generated"

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "__init__.py").write_text(
            '"""Auto-generated protobuf stubs. Do not edit."""\n'
        )

        proto_files = list(proto_dir.glob("*.proto"))
        if not proto_files:
            print("Warning: No .proto files found in", proto_dir)
            return

        cmd = [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"--proto_path={proto_dir}",
            f"--python_out={output_dir}",
            f"--grpc_python_out={output_dir}",
        ] + [str(f) for f in proto_files]

        print(f"Generating protobuf stubs: {' '.join(cmd)}")
        subprocess.check_call(cmd)

        # Fix imports in generated files (grpcio-tools generates absolute imports)
        for py_file in output_dir.glob("*_pb2*.py"):
            content = py_file.read_text()
            for proto_file in proto_files:
                module_name = proto_file.stem + "_pb2"
                content = content.replace(
                    f"import {module_name}", f"from . import {module_name}"
                )
            py_file.write_text(content)

        print(f"Generated {len(list(output_dir.glob('*.py')))} Python files")


setup(cmdclass={"build_py": BuildPyWithProto})
