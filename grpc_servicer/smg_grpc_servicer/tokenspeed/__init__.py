"""TokenSpeed gRPC servicer implementation.

Mirrors smg_grpc_servicer.vllm / smg_grpc_servicer.sglang. Wraps TokenSpeed's
AsyncLLM (main-process async frontend) behind the SGLang gRPC service so the
existing Rust router (which auto-detects the SGLang proto) can route traffic
to TokenSpeed without needing a new client.
"""

from smg_grpc_servicer.tokenspeed.servicer import TokenSpeedSchedulerServicer

__all__ = ["TokenSpeedSchedulerServicer"]
