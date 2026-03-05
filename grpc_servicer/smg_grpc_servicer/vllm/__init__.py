"""vLLM gRPC servicer — implements VllmEngine proto service on top of AsyncLLM."""

from smg_grpc_servicer.vllm.server import serve_grpc
from smg_grpc_servicer.vllm.servicer import VllmEngineServicer

__all__ = ["VllmEngineServicer", "serve_grpc"]
