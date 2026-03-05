# smg-grpc-servicer

gRPC servicer implementations for LLM inference engines. Currently supports vLLM,
with future support for SGLang and TensorRT-LLM.

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

## Architecture

```
smg-grpc-servicer  ──depends on──>  vllm            (hard dependency)
smg-grpc-servicer  ──depends on──>  smg-grpc-proto  (hard dependency)
vllm               ──optional──>    smg-grpc-servicer (lazy import via vllm serve --grpc)
```

This avoids circular dependencies: vLLM only imports `smg-grpc-servicer` at runtime
when `--grpc` is passed, via a lazy import.

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for local development setup, CI, and release workflows.
