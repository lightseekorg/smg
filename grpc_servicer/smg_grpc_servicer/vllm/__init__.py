"""vLLM gRPC servicer — implements VllmEngine proto service on top of AsyncLLM."""

from smg_grpc_servicer.vllm.servicer import VllmEngineServicer

__all__ = ["VllmEngineServicer"]
