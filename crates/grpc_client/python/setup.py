"""
Custom setup.py to generate protobuf stubs at build time.
The generated files are NOT committed to git — they're created fresh during pip install.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop


def _load_proto_build_helper():
    helper_path = Path(__file__).parent / "smg_grpc_proto" / "_proto_build.py"
    spec = spec_from_file_location("_smg_grpc_proto_build", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load proto build helper from {helper_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compile_grpc_protos() -> None:
    helper = _load_proto_build_helper()
    helper.ensure_generated_stubs(Path(__file__).parent, force=True)


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


try:
    from setuptools.command.editable_wheel import editable_wheel as _EditableWheelBase

    class EditableWheelWithProto(_EditableWheelBase):
        """PEP 660 editable install hook (used by uv and pip >= 21.3)."""

        def run(self):
            compile_grpc_protos()
            super().run()

    _editable_wheel_cmd: dict = {"editable_wheel": EditableWheelWithProto}
except ImportError:
    _editable_wheel_cmd = {}


setup(
    cmdclass={
        "build_py": BuildPyWithProto,
        "develop": DevelopWithProto,
        **_editable_wheel_cmd,
    }
)
