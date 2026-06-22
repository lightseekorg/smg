"""Unit tests for the shared multimodal /dev/shm tensor-transport reader.

Engine-agnostic: exercises ``mm_shm`` against a temp dir (no vLLM/TokenSpeed,
no real /dev/shm), so it runs anywhere. The handle is duck-typed.
"""

from types import SimpleNamespace

import pytest
from smg_grpc_servicer import mm_shm


def _handle(name: str, nbytes: int, offset: int = 0) -> SimpleNamespace:
    return SimpleNamespace(name=name, nbytes=nbytes, offset=offset)


def test_validated_shm_name_accepts_allowed_prefixes():
    assert mm_shm.validated_shm_name("smg-mm-123-456-0") == "smg-mm-123-456-0"
    # Leading slash is stripped (handles "/name" forms).
    assert mm_shm.validated_shm_name("/smg-mm-1") == "smg-mm-1"
    # Legacy prefix is still accepted for back-compat.
    assert mm_shm.validated_shm_name("smg-tokenspeed-9-0") == "smg-tokenspeed-9-0"


@pytest.mark.parametrize("bad", ["", "/", "a/b", "..", ".", "x\x00y"])
def test_validated_shm_name_rejects_traversal(bad):
    with pytest.raises(ValueError):
        mm_shm.validated_shm_name(bad)


@pytest.mark.parametrize("bad", ["passwd", "evil-file", "shm-mm-typo", "tokenspeed-x"])
def test_validated_shm_name_rejects_out_of_namespace(bad):
    with pytest.raises(ValueError):
        mm_shm.validated_shm_name(bad)


def test_roundtrip_reads_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(mm_shm, "UNLINK_MM_SHM_AFTER_READ", False)
    data = bytes(range(8))
    name = "smg-mm-test-roundtrip"
    (tmp_path / name).write_bytes(data)

    got = mm_shm.tensor_payload_bytes_from_shm(_handle(name, len(data)), shm_dir=str(tmp_path))

    assert got == data
    assert (tmp_path / name).exists()  # left in place when unlink is disabled


def test_roundtrip_honors_offset(tmp_path, monkeypatch):
    monkeypatch.setattr(mm_shm, "UNLINK_MM_SHM_AFTER_READ", False)
    name = "smg-mm-test-offset"
    (tmp_path / name).write_bytes(b"HEADER" + b"\xaa\xbb\xcc\xdd")

    got = mm_shm.tensor_payload_bytes_from_shm(_handle(name, 4, offset=6), shm_dir=str(tmp_path))

    assert got == b"\xaa\xbb\xcc\xdd"


def test_unlinks_after_read_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr(mm_shm, "UNLINK_MM_SHM_AFTER_READ", True)
    name = "smg-mm-test-unlink"
    (tmp_path / name).write_bytes(b"abcd")

    mm_shm.tensor_payload_bytes_from_shm(_handle(name, 4), shm_dir=str(tmp_path))

    assert not (tmp_path / name).exists()


def test_length_mismatch_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(mm_shm, "UNLINK_MM_SHM_AFTER_READ", False)
    name = "smg-mm-test-short"
    (tmp_path / name).write_bytes(b"ab")  # fewer bytes than the handle claims

    with pytest.raises(ValueError):
        mm_shm.tensor_payload_bytes_from_shm(_handle(name, 8), shm_dir=str(tmp_path))
