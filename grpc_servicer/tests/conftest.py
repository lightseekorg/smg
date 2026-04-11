# SPDX-License-Identifier: Apache-2.0
import sys
from unittest.mock import MagicMock

import pytest
from grpc_health.v1 import health_pb2

# Stub out vllm and its submodules so vllm health_servicer can be imported
# without a full vLLM installation.  MagicMock-based stubs auto-satisfy
# any attribute access and from-import statements at collection time.
# This must run before any smg_grpc_servicer imports are resolved.
_VLLM_STUBS = [
    "vllm",
    "vllm.engine",
    "vllm.engine.protocol",
    "vllm.inputs",
    "vllm.inputs.engine",
    "vllm.logger",
    "vllm.logprobs",
    "vllm.multimodal",
    "vllm.multimodal.inputs",
    "vllm.outputs",
    "vllm.sampling_params",
    "vllm.v1",
    "vllm.v1.engine",
    "vllm.v1.engine.async_llm",
]
for _name in _VLLM_STUBS:
    if _name not in sys.modules:
        sys.modules[_name] = MagicMock()

# Stub out sglang and its submodules so health_servicer can be imported
# without a full SGLang installation.  MagicMock-based stubs auto-satisfy
# any attribute access and from-import statements at collection time.
# This must run before any smg_grpc_servicer imports are resolved.
_SGLANG_STUBS = [
    "sglang",
    "sglang.srt",
    "sglang.srt.configs",
    "sglang.srt.configs.model_config",
    "sglang.srt.disaggregation",
    "sglang.srt.disaggregation.kv_events",
    "sglang.srt.disaggregation.utils",
    "sglang.srt.managers",
    "sglang.srt.managers.data_parallel_controller",
    "sglang.srt.managers.disagg_service",
    "sglang.srt.managers.io_struct",
    "sglang.srt.managers.schedule_batch",
    "sglang.srt.managers.scheduler",
    "sglang.srt.observability",
    "sglang.srt.observability.req_time_stats",
    "sglang.srt.sampling",
    "sglang.srt.sampling.sampling_params",
    "sglang.srt.server_args",
    "sglang.srt.utils",
    "sglang.srt.utils.network",
    "sglang.srt.utils.torch_memory_saver_adapter",
    "sglang.utils",
]
for _name in _SGLANG_STUBS:
    if _name not in sys.modules:
        sys.modules[_name] = MagicMock()

SERVING = health_pb2.HealthCheckResponse.SERVING
NOT_SERVING = health_pb2.HealthCheckResponse.NOT_SERVING
SERVICE_UNKNOWN = health_pb2.HealthCheckResponse.SERVICE_UNKNOWN


@pytest.fixture
def grpc_context():
    return MagicMock(spec=["set_code", "set_details", "cancelled", "done"])


@pytest.fixture
def request_msg():
    msg = MagicMock()
    msg.service = ""
    return msg
