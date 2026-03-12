# smg-grpc-servicer

gRPC servicer implementations for LLM inference engines. Supports vLLM and SGLang.

## Installation

For vLLM:

```bash
pip install smg-grpc-servicer[vllm]
```

For SGLang:

```bash
pip install smg-grpc-servicer[sglang]
```

## Usage

### vLLM

```bash
vllm serve meta-llama/Llama-2-7b-hf --grpc
```

### SGLang

```bash
sglang serve --model-path meta-llama/Llama-2-7b-hf --grpc-mode
```

## Architecture

```
smg-grpc-servicer[vllm]    ──optional dep──>  vllm     (lazy import)
smg-grpc-servicer[sglang]  ──optional dep──>  sglang   (lazy import)
smg-grpc-servicer           ──depends on──>  smg-grpc-proto  (hard dependency)
vllm                        ──optional──>    smg-grpc-servicer (via vllm serve --grpc)
sglang                      ──optional──>    smg-grpc-servicer (via --grpc-mode)
```

Backend dependencies are isolated via extras to avoid conflicts between vLLM and SGLang.

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for local development setup, CI, and release workflows.
