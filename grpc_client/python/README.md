# smg-grpc-proto

[![PyPI](https://img.shields.io/pypi/v/smg-grpc-proto)](https://pypi.org/project/smg-grpc-proto/)
[![Python](https://img.shields.io/pypi/pyversions/smg-grpc-proto)](https://pypi.org/project/smg-grpc-proto/)

Protocol Buffer definitions for [SMG](https://github.com/lightseekorg/smg) (Shepherd Model Gateway) gRPC services.

This package provides pre-compiled Python gRPC stubs for:
- **SGLang** scheduler service (`sglang_scheduler.proto`)
- **vLLM** engine service (`vllm_engine.proto`)
- **TensorRT-LLM** service (`trtllm_service.proto`)

## Installation

```bash
pip install smg-grpc-proto
```

Requires `grpcio>=1.78.0` and `protobuf>=5.26.0`.

## Usage

```python
from smg_grpc_proto import sglang_scheduler_pb2, sglang_scheduler_pb2_grpc
from smg_grpc_proto import vllm_engine_pb2, vllm_engine_pb2_grpc
from smg_grpc_proto import trtllm_service_pb2, trtllm_service_pb2_grpc
```

## Proto Source

The proto source files live in [`grpc_client/proto/`](https://github.com/lightseekorg/smg/tree/main/grpc_client/proto) in the SMG repository. Python stubs are generated at build time using `grpcio-tools` and shipped in the wheel.

## Development

To install in editable mode from the repo root:

```bash
pip install -e grpc_client/python/
```

For CI or environments where symlinks don't work:

```bash
mkdir -p grpc_client/python/smg_grpc_proto/proto
cp grpc_client/proto/*.proto grpc_client/python/smg_grpc_proto/proto/
pip install -e grpc_client/python/
```

## License

Apache-2.0
