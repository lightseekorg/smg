import os
import warnings

from setuptools import setup

_rust_env = os.environ.get("SMG_BUILD_WITH_RUST", None)
with_rust = _rust_env is None or (_rust_env.lower() not in ["0", "false", "no"])

rust_extensions = []
if with_rust:
    from setuptools_rust import Binding, RustExtension

    rust_extensions.append(
        RustExtension(
            target="smg_rs",
            path="Cargo.toml",
            binding=Binding.PyO3,
        )
    )
else:
    warnings.warn("Building 'smg' without Rust support. Performance may be degraded.")

setup(
    rust_extensions=rust_extensions,
    zip_safe=False,
)
