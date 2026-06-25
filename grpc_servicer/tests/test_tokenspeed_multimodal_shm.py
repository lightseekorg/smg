import os

import pytest
import torch
from smg_grpc_proto.generated import tokenspeed_scheduler_pb2
from smg_grpc_servicer.tokenspeed import servicer as servicer_module
from smg_grpc_servicer.tokenspeed.servicer import TokenSpeedSchedulerServicer
from tokenspeed.runtime.multimodal.shm_transport import ShmTensorHandle


def test_feature_from_proto_preserves_offset_shm_handle():
    tensor = tokenspeed_scheduler_pb2.TensorData(
        shape=[3, 4],
        dtype="float32",
        shm=tokenspeed_scheduler_pb2.ShmHandle(
            name="smg-tokenspeed-test",
            offset=128,
            nbytes=3 * 4 * 4,
            owner_id="smg:test",
        ),
    )

    feature = TokenSpeedSchedulerServicer._feature_from_proto(tensor)

    assert isinstance(feature, ShmTensorHandle)
    assert feature.shm_name == "smg-tokenspeed-test"
    assert feature.shape == (3, 4)
    assert feature.dtype is torch.float32
    assert feature.offset == 128
    assert feature.nbytes == 3 * 4 * 4


def test_feature_from_proto_rejects_offset_shm_length_mismatch():
    tensor = tokenspeed_scheduler_pb2.TensorData(
        shape=[3, 4],
        dtype="float32",
        shm=tokenspeed_scheduler_pb2.ShmHandle(
            name="smg-tokenspeed-test",
            offset=128,
            nbytes=4,
            owner_id="smg:test",
        ),
    )

    with pytest.raises(ValueError, match="byte length mismatch"):
        TokenSpeedSchedulerServicer._feature_from_proto(tensor)


def test_feature_from_proto_cast_defers_shared_shm_unlink(monkeypatch):
    monkeypatch.setattr(servicer_module, "UNLINK_MM_SHM_AFTER_READ", True)
    name = f"smg-tokenspeed-test-cast-{os.getpid()}"
    path = os.path.join("/dev/shm", name)
    raw = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32).numpy().tobytes()
    with open(path, "wb") as f:
        f.write(raw)

    deferred_unlinks: set[str] = set()
    first = tokenspeed_scheduler_pb2.TensorData(
        shape=[2],
        dtype="float32",
        shm=tokenspeed_scheduler_pb2.ShmHandle(
            name=name,
            offset=0,
            nbytes=2 * 4,
            owner_id="smg:test",
        ),
    )
    second = tokenspeed_scheduler_pb2.TensorData(
        shape=[2],
        dtype="float32",
        shm=tokenspeed_scheduler_pb2.ShmHandle(
            name=name,
            offset=2 * 4,
            nbytes=2 * 4,
            owner_id="smg:test",
        ),
    )

    try:
        first_feature = TokenSpeedSchedulerServicer._feature_from_proto(
            first,
            cast_to=torch.float16,
            deferred_unlink_names=deferred_unlinks,
        )
        assert first_feature.dtype == torch.float16
        assert os.path.exists(path)

        second_feature = TokenSpeedSchedulerServicer._feature_from_proto(
            second,
            cast_to=torch.float16,
            deferred_unlink_names=deferred_unlinks,
        )
        assert second_feature.dtype == torch.float16
        assert deferred_unlinks == {name}

        TokenSpeedSchedulerServicer._unlink_deferred_shm(deferred_unlinks)
        assert not os.path.exists(path)
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def test_offsets_from_proto_placeholders_validates_and_builds_offsets_once():
    placeholders = [
        tokenspeed_scheduler_pb2.PlaceholderRange(offset=10, length=3),
        tokenspeed_scheduler_pb2.PlaceholderRange(offset=20, length=1),
    ]

    assert TokenSpeedSchedulerServicer._offsets_from_proto_placeholders(placeholders) == (
        [(10, 12), (20, 20)],
        4,
        [12, 20],
        [0, 3, 4],
    )


def test_offsets_from_proto_placeholders_single_placeholder_fast_path():
    placeholders = [
        tokenspeed_scheduler_pb2.PlaceholderRange(offset=10, length=3),
    ]

    assert TokenSpeedSchedulerServicer._offsets_from_proto_placeholders(placeholders) == (
        [(10, 12)],
        3,
        [12],
        [0, 3],
    )


def test_offsets_from_proto_placeholders_rejects_empty_and_non_positive_lengths():
    with pytest.raises(ValueError, match="no placeholders"):
        TokenSpeedSchedulerServicer._offsets_from_proto_placeholders([])

    with pytest.raises(ValueError, match="length must be > 0"):
        TokenSpeedSchedulerServicer._offsets_from_proto_placeholders(
            [tokenspeed_scheduler_pb2.PlaceholderRange(offset=10, length=0)]
        )
