"""vLLM gRPC servicers -- VllmEngine proto service and standard health check."""

import logging

from smg_grpc_servicer.vllm.health_servicer import VllmHealthServicer
from smg_grpc_servicer.vllm.servicer import VllmEngineServicer

# Attach the top-level ``smg_grpc_servicer`` logger to the vllm logging
# hierarchy so that INFO/DEBUG messages from any submodule use the same
# handlers and format as native vllm loggers.
_vllm_logger = logging.getLogger("vllm")
_pkg_logger = logging.getLogger("smg_grpc_servicer")
_pkg_logger.handlers = list(_vllm_logger.handlers)
_pkg_logger.setLevel(_vllm_logger.level)
_pkg_logger.propagate = False

__all__ = ["VllmEngineServicer", "VllmHealthServicer"]
