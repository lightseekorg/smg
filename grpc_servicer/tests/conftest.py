# SPDX-License-Identifier: Apache-2.0
import sys
from unittest.mock import MagicMock

import pytest
from grpc_health.v1 import health_pb2

# Stub out sglang and its submodules so health_servicer can be imported
# without a full SGLang installation.  MagicMock-based stubs auto-satisfy
# any attribute access and from-import statements at collection time.
# This must run before any smg_grpc_servicer imports are resolved.
_SGLANG_STUBS = [
    "sglang",
    "sglang.srt",
    "sglang.srt.disaggregation",
    "sglang.srt.disaggregation.kv_events",
    "sglang.srt.disaggregation.utils",
    "sglang.srt.managers",
    "sglang.srt.managers.io_struct",
    "sglang.srt.managers.schedule_batch",
    "sglang.srt.observability",
    "sglang.srt.observability.req_time_stats",
    "sglang.srt.sampling",
    "sglang.srt.sampling.sampling_params",
    "sglang.srt.server_args",
    "sglang.srt.utils",
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
