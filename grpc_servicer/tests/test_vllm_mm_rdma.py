"""Integration tests for the vLLM servicer's RDMA multimodal wiring.

Exercises the servicer's tensor-deserialization seam: `remote` (RDMA) payloads
route to the NIXL puller with the model dtype, inline payloads deserialize
directly, and bfloat16 is a supported wire dtype. Skipped where vLLM isn't
installed (the servicer module imports vLLM at import time).

Run with: pytest grpc_servicer/tests/test_vllm_mm_rdma.py -v
"""

import struct
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("smg_grpc_proto")

from smg_grpc_proto import vllm_engine_pb2  # noqa: E402
from smg_grpc_proto.generated import common_pb2  # noqa: E402
from smg_grpc_servicer.vllm import servicer as vllm_servicer  # noqa: E402


class _FakePuller:
    def __init__(self, ready: bool = True):
        self.ready = ready
        self.calls = []

    def feature_from_remote(self, td, *, explicit_room, cast_to):
        self.calls.append((td, explicit_room, cast_to))
        return torch.zeros(1, dtype=cast_to)


def _servicer(puller, dtype):
    """Servicer instance bypassing __init__ (no real engine needed)."""
    s = vllm_servicer.VllmEngineServicer.__new__(vllm_servicer.VllmEngineServicer)
    s._rdma_pixel_puller = puller
    s.engine = types.SimpleNamespace(model_config=types.SimpleNamespace(dtype=dtype))
    return s


def test_bfloat16_is_a_supported_wire_dtype():
    assert vllm_servicer._PROTO_DTYPE_MAP["bfloat16"] is torch.bfloat16


def test_remote_payload_routes_to_puller_with_model_dtype():
    puller = _FakePuller()
    s = _servicer(puller, torch.bfloat16)
    td = vllm_engine_pb2.TensorData(
        shape=[1],
        dtype="float32",
        remote=common_pb2.RemoteTensorHandle(transport="nixl", descriptor=b"x", nbytes=4),
    )

    out = s._tensor_from_proto(td)

    assert len(puller.calls) == 1
    called_td, explicit_room, cast_to = puller.calls[0]
    assert called_td is td
    assert explicit_room is None  # vLLM has no EPD bootstrap_room
    assert cast_to is torch.bfloat16
    assert out.dtype is torch.bfloat16


def test_inline_bfloat16_payload_deserializes():
    s = _servicer(_FakePuller(), torch.bfloat16)
    # Two bf16 1.0 values (0x3F80 little-endian), carried inline.
    payload = (0x3F80).to_bytes(2, "little") * 2
    td = vllm_engine_pb2.TensorData(shape=[2], dtype="bfloat16", inline=payload)

    out = s._tensor_from_proto(td)

    assert out.dtype is torch.bfloat16
    assert list(out.shape) == [2]
    assert out.tolist() == [1.0, 1.0]


def test_inline_payload_does_not_touch_puller():
    puller = _FakePuller()
    s = _servicer(puller, torch.float32)
    td = vllm_engine_pb2.TensorData(shape=[1], dtype="float32", inline=struct.pack("<f", 1.0))

    s._tensor_from_proto(td)

    assert puller.calls == []
