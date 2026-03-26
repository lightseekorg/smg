# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock

import pytest
from grpc_health.v1 import health_pb2

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
