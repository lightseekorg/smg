"""Engine-free tests for the shared RDMA pixel puller wire format.

Covers the `SMGRDMA1` descriptor parse (shared by the TokenSpeed and vLLM
pullers) and the disabled-by-default readiness signal. No torch / nixl / vLLM
required — the module's heavy deps are imported lazily inside the pull path.

Run with: pytest grpc_servicer/tests/test_mm_rdma.py -v
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).parents[1] / "smg_grpc_servicer" / "mm_rdma.py"
_spec = importlib.util.spec_from_file_location("smg_mm_rdma_under_test", _MODULE_PATH)
mm_rdma = importlib.util.module_from_spec(_spec)
# Register before exec so dataclass annotation resolution can find the module.
sys.modules[_spec.name] = mm_rdma
_spec.loader.exec_module(mm_rdma)


def _descriptor(addr: int, gen: int, room: int, port: int, ip: str) -> bytes:
    return (
        b"SMGRDMA1"
        + addr.to_bytes(8, "little")
        + gen.to_bytes(8, "little")
        + room.to_bytes(8, "little", signed=True)
        + port.to_bytes(2, "little")
        + ip.encode()
    )


def _td(descriptor: bytes):
    """Minimal stand-in for a proto TensorData carrying a remote descriptor."""
    return types.SimpleNamespace(remote=types.SimpleNamespace(descriptor=descriptor))


class TestParseDescriptor:
    def test_round_trips_all_fields(self):
        desc = _descriptor(0xDEADBEEF, 7, 12345, 18515, "172.16.1.80")
        parsed = mm_rdma._parse_descriptor(_td(desc), explicit_room=None)
        assert parsed.remote_addr == 0xDEADBEEF
        assert parsed.expected_gen == 7
        assert parsed.room == 12345
        assert parsed.port == 18515
        assert parsed.ip == "172.16.1.80"

    def test_matching_explicit_room_accepted(self):
        desc = _descriptor(0x1000, 1, 99, 18515, "127.0.0.1")
        parsed = mm_rdma._parse_descriptor(_td(desc), explicit_room=99)
        assert parsed.room == 99

    def test_room_mismatch_rejected(self):
        desc = _descriptor(0x1000, 1, 99, 18515, "127.0.0.1")
        with pytest.raises(ValueError, match="room mismatch"):
            mm_rdma._parse_descriptor(_td(desc), explicit_room=100)

    def test_too_short_rejected(self):
        with pytest.raises(ValueError, match="too short"):
            mm_rdma._parse_descriptor(_td(b"SMGRDMA1\x00\x00"), explicit_room=None)

    def test_legacy_descriptor_needs_explicit_room(self):
        # A pre-SMGRDMA1 descriptor (no magic) carries no room inline.
        legacy = (
            (0x2000).to_bytes(8, "little")
            + (5).to_bytes(8, "little")
            + (18515).to_bytes(2, "little")
            + b"127.0.0.1"
        )
        parsed = mm_rdma._parse_descriptor(_td(legacy), explicit_room=42)
        assert parsed.room == 42
        assert parsed.remote_addr == 0x2000
        assert parsed.expected_gen == 5
        with pytest.raises(ValueError, match="legacy remote descriptor lacks room"):
            mm_rdma._parse_descriptor(_td(legacy), explicit_room=None)


class TestReadiness:
    def test_disabled_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("SMG_MM_PIXEL_RDMA", raising=False)
        puller = mm_rdma.RdmaPixelPuller(agent_name="test-agent", log_prefix="test")
        assert puller.ready is False

    def test_default_gateway_agent_name(self):
        assert mm_rdma.DEFAULT_GATEWAY_AGENT_NAME == "smg-gateway-encode"
