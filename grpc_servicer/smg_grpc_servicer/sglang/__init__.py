"""SGLang gRPC servicer package."""

__all__ = ["SGLangSchedulerServicer"]


def __getattr__(name: str):
    if name == "SGLangSchedulerServicer":
        from smg_grpc_servicer.sglang.servicer import SGLangSchedulerServicer

        return SGLangSchedulerServicer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
