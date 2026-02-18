"""SMG gRPC Proto - Protocol definitions for SGLang, vLLM, and TRT-LLM."""

__version__ = "0.3.3"

# Re-export generated modules for convenient access
# These imports will work after the package is built (stubs generated at build time)
try:
    from smg_grpc_proto.generated import (
        sglang_scheduler_pb2,
        sglang_scheduler_pb2_grpc,
        trtllm_service_pb2,
        trtllm_service_pb2_grpc,
        vllm_engine_pb2,
        vllm_engine_pb2_grpc,
    )

    __all__ = [
        "sglang_scheduler_pb2",
        "sglang_scheduler_pb2_grpc",
        "vllm_engine_pb2",
        "vllm_engine_pb2_grpc",
        "trtllm_service_pb2",
        "trtllm_service_pb2_grpc",
    ]
except ImportError:
    # During development/build, generated modules may not exist yet
    __all__ = []
