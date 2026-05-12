#!/usr/bin/env python3
# Reference: https://github.com/flashinfer-ai/flashinfer/blob/v0.2.0/scripts/update_whl_index.py
"""Update the wheel index in the lightseekorg/whl repository.

Index layout (matches the existing repo structure):
  <whl-repo-root>/cu<cuda>/index.html              ← PEP 503 top-level index
  <whl-repo-root>/cu<cuda>/<package>/index.html    ← per-package wheel list
  <whl-repo-root>/rocm<rocm>/index.html            ← PEP 503 top-level index
  <whl-repo-root>/rocm<rocm>/<package>/index.html  ← per-package wheel list

Install example (PyTorch-style --extra-index-url):
  pip install smg \
    --extra-index-url https://lightseek.org/whl/cu129/
"""

import argparse
import hashlib
import pathlib

BASE_URL = "https://github.com/lightseekorg/whl/releases/download"


def _cuda_display(cuda_digits: str) -> str:
    """'129' -> '12.9', '130' -> '13.0'"""
    return f"{cuda_digits[:-1]}.{cuda_digits[-1]}"


def _platform_index(cuda: str | None, rocm: str | None) -> tuple[str, str]:
    if cuda:
        return f"cu{cuda}", f"CUDA {_cuda_display(cuda)}"
    if rocm:
        return f"rocm{rocm}", f"ROCm {rocm}"
    raise ValueError("Either cuda or rocm must be provided")


def compute_sha256(path: pathlib.Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _ensure_in_top_index(index_root: pathlib.Path, package: str) -> None:
    """Add package to a platform index if not already listed (PEP 503)."""
    top_index = index_root / "index.html"
    entry = f'<a href="{package}/">{package}</a><br>\n'
    if top_index.exists():
        if entry in top_index.read_text():
            return
        with top_index.open("a") as f:
            f.write(entry)
    else:
        top_index.write_text(f"<!DOCTYPE html>\n{entry}")
    print(f"  Added {package} to top-level index")


def update_index(
    package: str,
    cuda: str | None,
    rocm: str | None,
    release_tag: str,
    wheel_dir: str,
    whl_repo_dir: str,
) -> None:
    platform_dir, platform_display = _platform_index(cuda, rocm)
    index_root = pathlib.Path(whl_repo_dir) / platform_dir
    index_dir = index_root / package
    index_dir.mkdir(exist_ok=True, parents=True)

    # Keep the platform index up-to-date for --index-url support
    _ensure_in_top_index(index_root, package)

    index_file = index_dir / "index.html"
    if not index_file.exists():
        index_file.write_text(
            f"<!DOCTYPE html>\n<h1>{package} wheels for {platform_display}</h1>\n"
        )

    wheels = sorted(pathlib.Path(wheel_dir).glob("*.whl"))
    if not wheels:
        print(f"WARNING: no .whl files found in {wheel_dir}")
        return

    for path in wheels:
        sha256 = compute_sha256(path)
        full_url = f"{BASE_URL}/{release_tag}/{path.name}#sha256={sha256}"
        with index_file.open("a") as f:
            f.write(f'<a href="{full_url}">{path.name}</a><br>\n')
        print(f"  Indexed: {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Update wheel index for lightseekorg/whl")
    parser.add_argument(
        "--package",
        required=True,
        help="Package name",
    )
    platform = parser.add_mutually_exclusive_group(required=True)
    platform.add_argument(
        "--cuda",
        help="CUDA version digits (e.g. 129, 130)",
    )
    platform.add_argument(
        "--rocm",
        help="ROCm version (e.g. 7.2)",
    )
    parser.add_argument(
        "--release-tag",
        required=True,
        help="Release tag in lightseekorg/whl",
    )
    parser.add_argument(
        "--wheel-dir",
        default="wheelhouse",
        help="Directory containing .whl files (default: wheelhouse)",
    )
    parser.add_argument(
        "--whl-repo-dir",
        default=".",
        help="Root of the lightseekorg/whl checkout (default: .)",
    )
    args = parser.parse_args()
    update_index(
        args.package,
        args.cuda,
        args.rocm,
        args.release_tag,
        args.wheel_dir,
        args.whl_repo_dir,
    )


if __name__ == "__main__":
    main()
