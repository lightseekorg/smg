"""TokenSpeed gRPC servicer — wraps :class:`AsyncLLM` behind the gRPC wire."""

from smg_grpc_servicer.tokenspeed.servicer import TokenSpeedSchedulerServicer

__all__ = ["TokenSpeedSchedulerServicer"]
