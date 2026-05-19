"""Verify generated proto stubs export the expected KV cache event symbols.

These tests do not require torch, vLLM, or a GPU — they only check that
the protobuf stubs were generated correctly and contain the expected message
types, fields, and RPC descriptors for the KV cache event pipeline.
"""


def test_common_pb2_kv_event_message_types_present():
    """All KV cache message types exist and expose their expected fields."""
    from smg_grpc_proto.generated import common_pb2

    # SubscribeKvEventsRequest
    req = common_pb2.SubscribeKvEventsRequest()
    assert hasattr(req, "start_sequence_number")

    # KvEventBatch
    batch = common_pb2.KvEventBatch()
    assert hasattr(batch, "sequence_number")
    assert hasattr(batch, "timestamp")
    assert hasattr(batch, "events")
    assert hasattr(batch, "dp_rank")

    # KvCacheEvent oneof variants can each be set
    stored_event = common_pb2.KvCacheEvent(stored=common_pb2.KvBlocksStored())
    assert stored_event.HasField("stored")

    removed_event = common_pb2.KvCacheEvent(removed=common_pb2.KvBlocksRemoved())
    assert removed_event.HasField("removed")

    cleared_event = common_pb2.KvCacheEvent(cleared=common_pb2.KvCacheCleared())
    assert cleared_event.HasField("cleared")

    # KvBlocksStored
    stored = common_pb2.KvBlocksStored()
    assert hasattr(stored, "blocks")
    assert hasattr(stored, "parent_block_hash")

    # KvBlock
    block = common_pb2.KvBlock()
    assert hasattr(block, "block_hash")
    assert hasattr(block, "token_ids")
    assert hasattr(block, "block_size")
    assert hasattr(block, "lora_id")

    # KvBlocksRemoved
    removed = common_pb2.KvBlocksRemoved()
    assert hasattr(removed, "block_hashes")

    # KvCacheCleared (empty message)
    assert common_pb2.KvCacheCleared() is not None


def test_kv_event_batch_serialization_round_trip():
    """KvEventBatch serializes and deserializes without data loss."""
    from smg_grpc_proto.generated import common_pb2

    original = common_pb2.KvEventBatch(
        sequence_number=42,
        timestamp=1_234_567_890.5,
        dp_rank=1,
        events=[
            common_pb2.KvCacheEvent(
                stored=common_pb2.KvBlocksStored(
                    parent_block_hash=99,
                    blocks=[
                        common_pb2.KvBlock(
                            block_hash=101,
                            token_ids=[1, 2, 3, 4],
                            block_size=4,
                            lora_id=7,
                        ),
                    ],
                )
            ),
            common_pb2.KvCacheEvent(
                removed=common_pb2.KvBlocksRemoved(block_hashes=[101, 102]),
            ),
            common_pb2.KvCacheEvent(cleared=common_pb2.KvCacheCleared()),
        ],
    )

    wire = original.SerializeToString()
    decoded = common_pb2.KvEventBatch()
    decoded.ParseFromString(wire)

    assert decoded.sequence_number == 42
    assert decoded.timestamp == 1_234_567_890.5
    assert decoded.dp_rank == 1
    assert len(decoded.events) == 3

    stored = decoded.events[0]
    assert stored.HasField("stored")
    assert stored.stored.parent_block_hash == 99
    assert len(stored.stored.blocks) == 1
    assert stored.stored.blocks[0].block_hash == 101
    assert list(stored.stored.blocks[0].token_ids) == [1, 2, 3, 4]
    assert stored.stored.blocks[0].block_size == 4
    assert stored.stored.blocks[0].lora_id == 7

    removed = decoded.events[1]
    assert removed.HasField("removed")
    assert list(removed.removed.block_hashes) == [101, 102]

    assert decoded.events[2].HasField("cleared")


def test_subscribe_kv_events_request_sequence_number_default():
    """start_sequence_number defaults to 0 (replay from current state)."""
    from smg_grpc_proto.generated import common_pb2

    req = common_pb2.SubscribeKvEventsRequest()
    assert req.start_sequence_number == 0

    req_with_seq = common_pb2.SubscribeKvEventsRequest(start_sequence_number=55)
    assert req_with_seq.start_sequence_number == 55


def test_vllm_engine_grpc_stub_exposes_subscribe_kv_events():
    """SubscribeKvEvents RPC is wired up in the generated gRPC bindings.

    The stub sets SubscribeKvEvents as an instance attribute (in __init__),
    so we verify it via the experimental VllmEngine utility class, which
    exposes all RPCs as static methods, and via the servicer base class.
    """
    from smg_grpc_proto.generated import vllm_engine_pb2_grpc

    assert hasattr(vllm_engine_pb2_grpc, "VllmEngineStub")
    assert hasattr(vllm_engine_pb2_grpc, "VllmEngineServicer")
    # Servicer base class defines SubscribeKvEvents as a class method
    assert hasattr(vllm_engine_pb2_grpc.VllmEngineServicer, "SubscribeKvEvents")
    # VllmEngine utility class exposes all RPCs as static methods
    assert hasattr(vllm_engine_pb2_grpc.VllmEngine, "SubscribeKvEvents")
