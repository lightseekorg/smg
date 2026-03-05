"""vLLM gRPC servicer — implements VllmEngine proto service on top of AsyncLLM."""

from smg_grpc_servicer.vllm.servicer import VllmEngineServicer
from smg_grpc_servicer.vllm.server import serve_grpc

__all__ = ["VllmEngineServicer", "serve_grpc"]
