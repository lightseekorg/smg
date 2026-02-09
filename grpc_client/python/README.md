# smg-grpc-proto

Protocol Buffer definitions for SMG (Shepherd Model Gateway) gRPC services.

This package provides the Python gRPC stubs for:
- **SGLang** scheduler service
- **vLLM** engine service
- **TRT-LLM** service

## Installation

```bash
pip install smg-grpc-proto
```

## Usage

```python
from smg_grpc_proto import sglang_scheduler_pb2, sglang_scheduler_pb2_grpc
from smg_grpc_proto import vllm_engine_pb2, vllm_engine_pb2_grpc
from smg_grpc_proto import trtllm_service_pb2, trtllm_service_pb2_grpc
```

## Development

The proto files are located in `grpc_client/proto/` in the SMG repository. Python stubs are generated at build time using `grpcio-tools`.

To build locally:

```bash
# From repo root
mkdir -p grpc_client/python/smg_grpc_proto/proto
cp grpc_client/proto/*.proto grpc_client/python/smg_grpc_proto/proto/
pip install -e grpc_client/python/
```

## License

Apache-2.0
