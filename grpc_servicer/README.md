# smg-grpc-servicer

gRPC servicer implementations for LLM inference engines. Currently supports vLLM,
with future support for SGLang and TensorRT-LLM.

## Prerequisites

- Python >= 3.10
- vLLM (installed separately — from source, nightly, or PyPI)

## Installation

```bash
pip install smg-grpc-servicer
```

Or with vLLM's optional dependency:

```bash
pip install vllm[grpc]
```

## Usage

With `vllm serve`:

```bash
vllm serve meta-llama/Llama-2-7b-hf --grpc
```

Or directly:

```bash
python -m smg_grpc_servicer.vllm.server --model meta-llama/Llama-2-7b-hf --port 50051
```

## Local Development

Install both proto and servicer as editable packages:

```bash
pip install -e grpc_client/python/
pip install -e grpc_servicer/
```

No version concerns locally — editable installs always use the latest source.

## Release Process

This package is published to PyPI via GitHub Actions. To release:

1. Bump the version in `grpc_servicer/pyproject.toml`
2. Merge to `main`
3. CI automatically builds and uploads to PyPI

### When changing protos and servicer together

1. Make all changes (proto + servicer) in a single PR
2. Bump versions in both `grpc_client/python/pyproject.toml` and `grpc_servicer/pyproject.toml`
3. On merge, CI releases both — proto publishes first since it's a simpler build
4. The servicer pins proto loosely (`smg-grpc-proto >= 0.4.2`) so the previous
   version satisfies during the brief publish window

### When changing only the servicer

Bump only `grpc_servicer/pyproject.toml`. No proto release needed.

## Architecture

```
smg-grpc-servicer  ──depends on──>  vllm            (runtime, not declared in pyproject.toml)
smg-grpc-servicer  ──depends on──>  smg-grpc-proto  (hard dependency)
vllm               ──optional──>    smg-grpc-servicer (lazy import via vllm serve --grpc)
```

This avoids circular dependencies: vLLM only imports `smg-grpc-servicer` at runtime
when `--grpc` is passed, via a lazy import.
