import os

import pytest
from smg_grpc_proto.generated import tokenspeed_scheduler_pb2

torch = pytest.importorskip("torch")
shm_transport = pytest.importorskip("tokenspeed.runtime.multimodal.shm_transport")
servicer_module = pytest.importorskip("smg_grpc_servicer.tokenspeed.servicer")
ShmTensorHandle = shm_transport.ShmTensorHandle
TokenSpeedSchedulerServicer = servicer_module.TokenSpeedSchedulerServicer


def _require_writable_dev_shm():
    if not os.path.isdir("/dev/shm") or not os.access("/dev/shm", os.W_OK):
        pytest.skip("/dev/shm is not available or writable")


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
    _require_writable_dev_shm()
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


def test_mm_inputs_rejects_model_specific_reusing_preserved_encoder_shm(monkeypatch):
    _require_writable_dev_shm()
    monkeypatch.setattr(servicer_module, "UNLINK_MM_SHM_AFTER_READ", True)
    name = f"smg-tokenspeed-test-preserved-{os.getpid()}"
    path = os.path.join("/dev/shm", name)
    with open(path, "wb") as f:
        f.write(torch.tensor([1.0], dtype=torch.float32).numpy().tobytes())

    mm_inputs = tokenspeed_scheduler_pb2.MultimodalInputs(
        items=[
            tokenspeed_scheduler_pb2.MultimodalItem(
                modality=tokenspeed_scheduler_pb2.IMAGE,
                content_hash=b"hash",
                encoder_input=tokenspeed_scheduler_pb2.TensorData(
                    shape=[1],
                    dtype="float32",
                    shm=tokenspeed_scheduler_pb2.ShmHandle(
                        name=name,
                        offset=0,
                        nbytes=4,
                        owner_id="smg:test",
                    ),
                ),
                model_specific_tensors={
                    "image_grid_thw": tokenspeed_scheduler_pb2.TensorData(
                        shape=[1],
                        dtype="uint32",
                        shm=tokenspeed_scheduler_pb2.ShmHandle(
                            name=name,
                            offset=0,
                            nbytes=4,
                            owner_id="smg:test",
                        ),
                    ),
                },
                placeholders=[
                    tokenspeed_scheduler_pb2.PlaceholderRange(offset=0, length=1),
                ],
            ),
        ]
    )
    servicer = object.__new__(TokenSpeedSchedulerServicer)

    with pytest.raises(ValueError, match="must not share SHM segments"):
        servicer._mm_inputs_from_itemized_proto(mm_inputs)
    assert not os.path.exists(path)


def test_tensor_from_proto_rejects_shm_length_mismatch_before_read():
    tensor = tokenspeed_scheduler_pb2.TensorData(
        shape=[3, 4],
        dtype="float32",
        shm=tokenspeed_scheduler_pb2.ShmHandle(
            name="smg-tokenspeed-test-does-not-need-to-exist",
            offset=0,
            nbytes=4,
            owner_id="smg:test",
        ),
    )

    with pytest.raises(ValueError, match="byte length mismatch"):
        TokenSpeedSchedulerServicer._tensor_from_proto(tensor)


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
